# 公平对比的原则与口径

这份文档说明本仓库怎么让 DeepInfer、SGLang、vLLM 三家的数字可以放在一起比，以及哪些看起来顺手的做法会让对比失真。

## 1. 为什么不能直接拿 DeepInfer 的 `benchmark.py` / `run_sweep.sh` 结果去比

`examples/benchmark.py` 是 DeepInfer 用来分析自身稳态性能的工具，它的口径和 sglang/vllm 的 `bench_serving` 不一样：

| benchmark.py 的做法 | 源码位置 | 对比时的问题 |
|---|---|---|
| `DEEPINFER_FORCE_LOAD_BALANCE=1` | `benchmark.py` `setup_llm()`；生效点 `prepare_tensor_input_base.cpp` 里的 `gate_force_random` | **MoE gate 改成随机路由**，专家负载被人为摊平，跑的不是模型真实路由。EP 下专家不均衡本来就是主要开销之一，这样测出来偏乐观 |
| decode 吞吐 = `SPS × 请求数 × (1+nextn)` | `examples/README.md`："full-acceptance accounting" | **按投机草稿全部被接受来算**，nextn=5 时最多虚高 6 倍；sweep 出图再乘一个**假定**的接受率（如 0.75），也不是实测值 |
| prompt 是随机 token，`ignore_eos=True` | `benchmark.py` `add_requests()` | 随机文本上投机解码的接受率没有意义 |
| 32k 上下文里只有 4096 个新 token，前缀从一个源请求 fork 出来 | `benchmark.py` | 相当于 87.5% 前缀命中，对方若没有同样的前缀结构，prefill 计算量差 8 倍 |
| prefill、decode 分开测，只取稳态窗口 | `examples/README.md` | 不含排队、调度、TTFT，是"引擎内核"指标，不是服务指标 |

结论：`benchmark.py` 可以照样跑，用来看 DeepInfer 内部 prefill/decode 的稳态性能，但**不进对比表**。

另外 `offline_inference.py` 在 NVIDIA 上默认开 `DEEPINFER_BATCH_INVARIANT=1`（为逐位可复现牺牲性能），压测时也不能开。

## 2. 本仓库的测法

### 2.1 统一负载：闭环固定并发

- **闭环并发 C**：始终保持 C 个请求在飞，一个结束立刻补一个。SGLang/vLLM 侧是 C 个线程的流式 client（`bench/client_openai.py`）；DeepInfer 侧直接驱动 `LLMEngine`，一个结束就 `add_request` 补一个（`bench/run_deepinfer.py`）。两边语义相同。
- **输入逐 token 相同**：`bench/make_dataset.py` 预先生成 token ids，三家都直接吃 token ids，绕开各家 tokenizer / chat template 的差异。
- **输出长度固定**：`max_tokens = output_len` + `ignore_eos=True`，结果里有 `num_output_len_mismatch` 检查是否真的生成了这么多。
- **采样参数统一**：默认 `temperature=0, top_p=1`。
- **无前缀复用**：每个请求从文本流的不同位置截取，请求之间没有公共前缀；每个并发点用不同 seed 的数据集，避免同一个 server 跨扫描点命中上一个点的缓存；预热请求用逆序 token，不和正式请求共享前缀。
- **先预热再计时**：每个点先跑一轮短输出请求，触发 cuda graph / autotune / JIT 等一次性开销。
- **请求数 ≥ 4×并发**：保证中间有足够长的稳态段；另外单独报告去掉首尾各 10% 时间后的"稳态吞吐"。

### 2.2 统一指标

| 指标 | 定义 |
|---|---|
| 输出吞吐 / GPU | 实际输出 token 总数 ÷ 测量窗口时长 ÷ 占用 GPU 数。**只数真实输出的 token**，不做"全部接受"折算 |
| 稳态输出吞吐 / GPU | 去掉窗口首尾各 10% 时间后，窗口内收到的 token 数 ÷ 时长 ÷ GPU 数 |
| TTFT | 请求发出 → 收到第一个输出 token |
| TPOT | (最后一个 token 时刻 − 第一个 token 时刻) ÷ (输出 token 数 − 1) |
| 单用户速度（interactivity） | 1 / TPOT，tokens/s/user；和 InferenceX 的 interactivity 同义 |
| 平均每步 token 数 | (输出 token 数 − 1) ÷ (收到输出的次数 − 1)，即去掉 prefill 产出的首 token 后，每次收到输出平均带几个 token。DeepInfer 每次输出 = 一个引擎 step；server 流式下每个 chunk 基本也对应一个 decode step。不开投机解码时 ≈1，开了之后 ≈ 平均接受长度。三家同一口径。另外：DeepInfer 引擎日志里的 `avg_nextn_accept_rate`、vLLM/SGLang `/metrics` 里投机计数器的增量（存在 `server_metrics_delta`）可作旁证，但各家定义不同，不直接比 |

最终比较用 **吞吐-交互性 Pareto 曲线**：横轴单用户速度，纵轴每卡吞吐，扫并发 1/4/8/16/32/64/128 连成线（`bench/summarize.py` 出图）。单个点没有意义：任何系统都可以靠调并发在"更快"和"吞吐更高"之间移动。

### 2.3 GPU 数不同怎么办：按每卡归一化

官方参考配置用的卡数不同：SGLang TP4/EP4 用 4 张卡，vLLM TP2 用 2 张卡，DeepInfer 默认 EP8 用 8 张卡。吞吐一律按 **每卡** 报告。

但每卡吞吐也会随并行规模变化（EP 越大，每卡分到的专家越少、通信越多），所以分两层：

- **A 层：各自官方参考配置**。照抄官方给的命令，回答"各家按推荐方式部署，每张卡能干多少活"。
- **B 层：对齐配置**。尽量把能对齐的都对齐，回答"同样的卡、同样的条件下，谁的实现更快"：

| 维度 | 对齐方式 |
|---|---|
| 卡数 | 都用 8 卡（SGLang `tp8`、vLLM `tp8`、DeepInfer EP8）；另外 DeepInfer EP4 对 SGLang `ref`（同为 4 卡） |
| 权重 | 同一份 checkpoint（FP8），三家都不另外量化 |
| 投机解码 | 两组：全关（DeepInfer `--nextn 0`、SGLang `*-nospec`、vLLM 不传 spec）；全开且都用 DSpark 5-token block（DeepInfer `--nextn 5`、SGLang `tp8`、vLLM `tp8-spec`），开的那组必须用 sharegpt 真实文本。V4.1 没有 MTP 模块，DSpark 是唯一的投机方法；验证策略（如 vLLM 在 NVIDIA 上默认的 adaptive verification）各家可能不同，要记录 |
| prefill chunk | 16384（DeepInfer `--max-num-batched-tokens`、vLLM `tp8` 的 `--max-num-batched-tokens`；SGLang 用默认 chunked prefill，跑前在日志里确认实际值） |
| 最大 batch | 并发上限 ≥ 128（SGLang `tp8` cuda graph bs 256、vLLM `tp8` max-num-seqs 256；官方参考的 vLLM max-num-seqs 8 只适合低并发） |

### 2.4 必须记录、但没法完全对齐的差异

这些差异属于"各家实现的一部分"，不强行对齐，但结果里要写清楚：

- **KV cache / indexer 精度**：vLLM 参考命令用 `--kv-cache-dtype fp8`、indexer `mxfp4`；DeepInfer 和 SGLang 用各自默认值，跑前要从日志里查清楚并记下来。精度不同会影响速度，也可能影响输出质量。
- **engram CPU offload**：vLLM 参考命令把 engram 放 CPU，省显存但可能更慢。
- **HTTP 开销**：SGLang/vLLM 走 HTTP + 流式 + 反 tokenize，DeepInfer 是进程内离线引擎，没有这部分开销。大并发下占比很小，低并发（c=1）时会让 DeepInfer 的 TTFT/TPOT 略占便宜。
- **EPLB**：DeepInfer 默认不开（`--enable-eplb` 可开），对齐参考配置里 SGLang/vLLM 都没开 EPLB。

### 2.5 速度之外：先确认输出质量等价

只有三家输出质量相当，速度对比才有意义。至少要抽查同一批 prompt 的输出是否正常（不乱码、不重复），最好跑一个小评测集（如 GSM8K 子集）。DeepInfer 自己的评测见仓库 `eval/README.md`。

## 3. 每次跑之前的检查清单

- [ ] 机器上只有一个系统在占 GPU（跑 DeepInfer 前 `scripts/stop_servers.sh`）
- [ ] `nvidia-smi` 确认 GPU 空闲、没有残留进程
- [ ] 记录：镜像 tag + digest（`docker inspect`）、DeepInfer commit、驱动版本
- [ ] 结果里 `num_failed == 0`、`num_output_len_mismatch == 0`
- [ ] 同一个点至少跑两次，差异 < 3% 才算稳定；不稳定就三次取中位数
