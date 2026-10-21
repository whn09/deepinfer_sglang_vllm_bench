# Fair-comparison rules and metric definitions

How this repo makes DeepInfer, SGLang and vLLM numbers comparable, and which convenient shortcuts would quietly break the comparison.

## 1. Why DeepInfer's `benchmark.py` / `run_sweep.sh` numbers cannot be compared directly

`examples/benchmark.py` is DeepInfer's tool for analysing its own steady state. Its accounting differs from serving benchmarks such as `bench_serving`:

| What benchmark.py does | Where | Effect on a comparison |
|---|---|---|
| Sets `DEEPINFER_FORCE_LOAD_BALANCE=1` | `benchmark.py` `setup_llm()`; applied as `gate_force_random` in `prepare_tensor_input_base.cpp` | **The MoE gate routes randomly.** Expert load is flattened artificially instead of following the model's real routing. Under EP, expert imbalance is a major cost, so the numbers are optimistic |
| Decode throughput = `SPS × requests × (1+nextn)` | `examples/README.md`, "full-acceptance accounting" | **Counts every draft token as accepted.** Up to 6× inflation at nextn=5; the sweep plot then multiplies by an *assumed* acceptance (e.g. 0.75), not a measured one |
| Random-token prompts, `ignore_eos=True` | `benchmark.py` `add_requests()` | Speculative acceptance on random text is meaningless |
| 32k context of which only 4096 tokens are new; requests are forked from one cached source request | `benchmark.py` | Equivalent to an 87.5% prefix hit; without the same prefix structure on the other side, prefill work differs by 8× |
| Prefill and decode measured separately, steady-state windows only | `examples/README.md` | Excludes queueing, scheduling and TTFT; an engine-kernel metric, not a serving metric |

Run `benchmark.py` for DeepInfer-internal analysis if useful, but **keep it out of the comparison tables**.

`offline_inference.py` also enables `DEEPINFER_BATCH_INVARIANT=1` on NVIDIA by default (bitwise reproducibility at a performance cost); never enable it for benchmarking.

## 2. How this repo measures (fixed-length workload)

### 2.1 Same load: closed-loop fixed concurrency

- **Closed loop at concurrency C**: C requests are always in flight; when one finishes the next is issued. SGLang/vLLM use a C-thread streaming client (`bench/client_openai.py`); DeepInfer drives `LLMEngine` directly and calls `add_request` as soon as a request finishes (`bench/run_deepinfer.py`). Same semantics on both sides.
- **Identical inputs, token for token**: `bench/make_dataset.py` pre-generates token ids and all three stacks consume token ids, bypassing tokenizer / chat-template differences.
- **Fixed output length**: `max_tokens = output_len` plus `ignore_eos=True`; `num_output_len_mismatch` in the results checks that the length was actually produced.
- **Same sampling**: `temperature=0, top_p=1` by default.
- **No prefix reuse**: each request is cut from a different offset of the text stream, so requests share no prefix; each concurrency point uses a dataset with a different seed so a server cannot hit cache left by the previous point; warmup requests use reversed token order so they share no prefix with measured requests.
- **Warm up before timing**: each point first runs a round of short-output requests to trigger CUDA graph capture, autotuning and JIT.
- **Requests ≥ 4× concurrency**: leaves a long steady segment; a "steady" throughput with the first and last 10% of the window trimmed is reported separately.

### 2.2 Same metrics

| Metric | Definition |
|---|---|
| Output throughput / GPU | Output tokens actually produced ÷ measurement window ÷ GPUs used. **Only real output tokens count** — no full-acceptance accounting |
| Steady output throughput / GPU | Tokens received inside the window with its first and last 10% trimmed ÷ that duration ÷ GPUs |
| TTFT | Request sent → first output token received |
| TPOT | (last token time − first token time) ÷ (output tokens − 1) |
| Interactivity | 1 / TPOT, tokens/s/user — same meaning as on InferenceX |
| Tokens per step (`tokens_per_chunk`) | (output tokens − 1) ÷ (output deliveries − 1): after the first token from prefill, how many tokens each delivery carries. For DeepInfer one delivery = one engine step; for a streaming server one chunk ≈ one decode step. ≈1 without speculation, ≈ mean acceptance length with it. Same definition for all three. DeepInfer's `avg_nextn_accept_rate` log line and the servers' `/metrics` speculative counters (stored in `server_metrics_delta`) are corroborating evidence only; their definitions differ |

The final comparison is a **throughput-vs-interactivity Pareto curve**: interactivity on x, throughput per GPU on y, concurrency 1/4/8/16/32/64/128 joined into a line (`bench/summarize.py`). A single point means nothing — any system can trade speed for throughput by changing concurrency.

### 2.3 Different GPU counts: normalise per GPU

The official reference configs use different GPU counts: SGLang TP4/EP4 on 4 GPUs, vLLM TP2 on 2 GPUs, DeepInfer EP8 on 8 GPUs. Throughput is always reported **per GPU**.

Per-GPU throughput still moves with parallel size (larger EP means fewer experts per GPU and more communication), so there are two tiers:

- **Tier A — each stack's official reference config**, copied verbatim: "deployed the recommended way, how much work does each GPU do?"
- **Tier B — aligned configs**, aligning everything that can be aligned: "same GPUs, same conditions, whose implementation is faster?"

| Dimension | Alignment |
|---|---|
| GPUs | **4 for the main comparison** — the GPU count InferenceX uses for DeepSeek-V4.1-Flash on B200 (SGLang `tp4`, vLLM `tp4`, DeepInfer EP4). 8 GPUs (SGLang `tp8`, vLLM `tp8`, DeepInfer EP8) as a supplement |
| Weights | The same checkpoint (FP8 with FP4 experts); no extra quantization anywhere |
| Speculation | Two groups: all off (DeepInfer `--nextn 0`, SGLang `*-nospec`, vLLM without spec), or all DSpark with a 5-token block (DeepInfer `--nextn 5`, SGLang `tp4`, vLLM `tp4-spec`). Speculative runs use ShareGPT text only. V4.1 has no MTP module; DSpark is the only speculative method. Verification policy may differ (e.g. vLLM's adaptive verification, on by default on NVIDIA) — record it |
| Prefill chunk | 16384 (DeepInfer `--max-num-batched-tokens`, vLLM `tp4`/`tp8` `--max-num-batched-tokens`; SGLang's default is 16384 too, confirmed in its log) |
| Max batch | Concurrency up to ≥ 128 (SGLang `tp4`/`tp8` CUDA graph bs 256, vLLM `tp4`/`tp8` max-num-seqs 256; the reference vLLM max-num-seqs 8 only suits low concurrency) |

- **Tier C — same parallel strategy (ablation)**: SGLang and vLLM also run DeepInfer's strategy, attention DP + MoE EP + DeepEP (both release images ship a DeepEP build). It splits a gap into "parallel strategy" vs "implementation of the same strategy". Tiers A/B keep each stack's recommended config: DeepEP is intrinsic to DeepInfer's design, while InferenceX's own B200 recipes do not use DeepEP either (SGLang `moe-a2a-backend: none`, vLLM DEP with MegaMoE), and forcing one stack's internals onto another is not "fairer".

### 2.4 Scheduling knobs trade TTFT for decode throughput: always report TTFT

DeepInfer's `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` (engine default 100) admits a waiting prompt on an otherwise idle rank only every N decode steps. Measured on 4 GPUs, DSpark 5, `sg-1k-1k`: interval 100 vs 2 gives c=128 3003 vs 2658 tok/s/GPU and TPOT 7.6 vs 10.3 ms, but TTFT 1491 vs 172 ms (c=1: 552 vs 73 ms). Any engine can buy decode throughput with TTFT this way (SGLang's `--prefill-decode-interval`, vLLM's chunked-prefill budget play the same role), so:

- Every comparison table carries TTFT next to throughput and TPOT.
- The headline comparison is at matched TTFT (or under a TTFT SLO), not at matched concurrency alone.
- `bench/run_deepinfer.py` defaults to `--prefill-any-interval 2` (serving-like) and records it in `engine_args`; interval 100 is kept as a reference point.

### 2.5 Differences to record but not force

These are part of each implementation; do not force them equal, but state them next to the results:

- **KV cache / indexer precision**: the vLLM reference uses `--kv-cache-dtype fp8` and an `mxfp4` indexer; DeepInfer and SGLang use their own defaults — read them from the startup logs and record them. Precision affects speed and may affect output quality.
- **Engram CPU offload**: the vLLM reference keeps engram on CPU — saves HBM, may cost speed.
- **HTTP overhead**: SGLang/vLLM go through HTTP + streaming + detokenization; DeepInfer runs in-process. Negligible at high concurrency; at c=1 it slightly favours DeepInfer's TTFT/TPOT.
- **EPLB**: off by default in DeepInfer (`--enable-eplb` turns it on); neither SGLang nor vLLM enables EPLB in the aligned configs.

### 2.6 Beyond speed: check output quality first

A speed comparison only means something if output quality is equivalent. At minimum spot-check outputs for the same prompts (no garbage, no repetition loops); better, run a small eval set (e.g. a GSM8K subset). DeepInfer's own evals: `eval/README.md` in its repo.

## 3. Comparing with InferenceX: synthetic acceptance length

InferenceX's published AgentX throughput numbers are **not** measured with real speculative verification. Each model, thinking mode and draft length has one committed "golden" acceptance length (AL), measured on the SPEED-Bench Qualitative coding split, and throughput runs inject it (`infx/golden_al_distribution/` in `SemiAnalysisAI/InferenceX`):

| `deepseek-v4.1-flash`, DSpark | draft 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| thinking off | 1.88 | 2.62 | 3.27 | 3.73 | 4.07 |
| thinking on | 1.81 | 2.43 | 2.90 | 3.26 | **3.51** |

AL includes the verification token. AgentX runs with thinking on, so DSpark 5 uses **3.51**. How each stack injects it:

- vLLM: `"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.51` in `--speculative-config`
- SGLang: `SGLANG_SIMULATE_ACC_LEN=3.51`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`
- DeepInfer: no such switch today — a patch is planned in #5

Rules for this repo:

- Any number compared with InferenceX must use the same golden AL.
- Report real-acceptance runs separately; they show how good each stack's drafting actually is on that traffic.
- Never mix synthetic-AL and real-AL points in one curve.

## 4. Checklist before every run

- [ ] Only one stack holds the GPUs (`scripts/stop_servers.sh` before DeepInfer)
- [ ] `nvidia-smi` shows idle GPUs and no leftover processes
- [ ] Record image tag + digest (`docker inspect`), DeepInfer commit, driver version
- [ ] Results show `num_failed == 0` and `num_output_len_mismatch == 0`
- [ ] Run each point at least twice; < 3% difference counts as stable, otherwise take the median of three

<details><summary>中文</summary>

# 公平对比的规则与指标定义

这份文档说明本仓库怎么让 DeepInfer、SGLang、vLLM 三家的数字可以放在一起比，以及哪些看起来顺手的做法会悄悄让对比失真。

## 1. 为什么不能直接拿 DeepInfer 的 `benchmark.py` / `run_sweep.sh` 结果去比

`examples/benchmark.py` 是 DeepInfer 用来分析自身稳态性能的工具，口径和 `bench_serving` 这类服务端 benchmark 不一样：

| benchmark.py 的做法 | 源码位置 | 对比时的问题 |
|---|---|---|
| 设置 `DEEPINFER_FORCE_LOAD_BALANCE=1` | `benchmark.py` `setup_llm()`；生效点是 `prepare_tensor_input_base.cpp` 里的 `gate_force_random` | **MoE gate 改成随机路由**，专家负载被人为摊平，跑的不是模型真实路由。EP 下专家不均衡本来就是主要开销之一，这样测出来偏乐观 |
| decode 吞吐 = `SPS × 请求数 × (1+nextn)` | `examples/README.md`："full-acceptance accounting" | **按投机草稿全部被接受来算**，nextn=5 时最多虚高 6 倍；sweep 出图再乘一个**假定**的接受率（如 0.75），也不是实测值 |
| prompt 是随机 token，`ignore_eos=True` | `benchmark.py` `add_requests()` | 随机文本上投机解码的接受率没有意义 |
| 32k 上下文里只有 4096 个新 token，请求从一个缓存好的源请求 fork 出来 | `benchmark.py` | 相当于 87.5% 前缀命中，对方没有同样的前缀结构时，prefill 计算量差 8 倍 |
| prefill、decode 分开测，只取稳态窗口 | `examples/README.md` | 不含排队、调度、TTFT，是"引擎内核"指标，不是服务指标 |

`benchmark.py` 可以照样跑，用来分析 DeepInfer 内部，但**不进对比表**。

另外 `offline_inference.py` 在 NVIDIA 上默认开 `DEEPINFER_BATCH_INVARIANT=1`（为逐位可复现牺牲性能），压测时绝不能开。

## 2. 本仓库的测法（定长负载）

### 2.1 统一负载：闭环固定并发

- **闭环并发 C**：始终保持 C 个请求在飞，一个结束立刻补一个。SGLang/vLLM 侧是 C 个线程的流式 client（`bench/client_openai.py`）；DeepInfer 侧直接驱动 `LLMEngine`，一个结束就 `add_request` 补一个（`bench/run_deepinfer.py`）。两边语义相同。
- **输入逐 token 相同**：`bench/make_dataset.py` 预先生成 token ids，三家都直接吃 token ids，绕开各家 tokenizer / chat template 的差异。
- **输出长度固定**：`max_tokens = output_len` + `ignore_eos=True`，结果里的 `num_output_len_mismatch` 检查是否真的生成了这么多。
- **采样参数统一**：默认 `temperature=0, top_p=1`。
- **无前缀复用**：每个请求从文本流的不同位置截取，请求之间没有公共前缀；每个并发点用不同 seed 的数据集，避免同一个 server 命中上一个点留下的缓存；预热请求用逆序 token，不和正式请求共享前缀。
- **先预热再计时**：每个点先跑一轮短输出请求，触发 cuda graph 捕获、autotune、JIT 等一次性开销。
- **请求数 ≥ 4×并发**：保证中间有足够长的稳态段；另外单独报告去掉首尾各 10% 时间后的"稳态吞吐"。

### 2.2 统一指标

| 指标 | 定义 |
|---|---|
| 输出吞吐 / GPU | 实际输出的 token 数 ÷ 测量窗口时长 ÷ 占用 GPU 数。**只数真实输出的 token**，不做"全部接受"折算 |
| 稳态输出吞吐 / GPU | 去掉窗口首尾各 10% 时间后，窗口内收到的 token 数 ÷ 时长 ÷ GPU 数 |
| TTFT | 请求发出 → 收到第一个输出 token |
| TPOT | (最后一个 token 时刻 − 第一个 token 时刻) ÷ (输出 token 数 − 1) |
| 交互性（interactivity） | 1 / TPOT，tokens/s/user，和 InferenceX 的 interactivity 同义 |
| 每步 token 数（`tokens_per_chunk`） | (输出 token 数 − 1) ÷ (收到输出的次数 − 1)：去掉 prefill 产出的首 token 后，每次收到输出平均带几个 token。DeepInfer 每次输出 = 一个引擎 step；流式 server 每个 chunk 约等于一个 decode step。不开投机解码时 ≈1，开了以后 ≈ 平均接受长度。三家同一口径。DeepInfer 日志里的 `avg_nextn_accept_rate`、server `/metrics` 里的投机计数器（存在 `server_metrics_delta`）只作旁证，各家定义不同 |

最终比较用**吞吐-交互性 Pareto 曲线**：横轴交互性，纵轴每卡吞吐，并发 1/4/8/16/32/64/128 连成线（`bench/summarize.py` 出图）。单个点没有意义：任何系统都可以靠调并发在"更快"和"吞吐更高"之间移动。

### 2.3 GPU 数不同：按每卡归一化

官方参考配置用的卡数不同：SGLang TP4/EP4 用 4 卡，vLLM TP2 用 2 卡，DeepInfer EP8 用 8 卡。吞吐一律按**每卡**报告。

但每卡吞吐也会随并行规模变化（EP 越大，每卡分到的专家越少、通信越多），所以分两层：

- **A 层：各自的官方参考配置**，原样照抄，回答"各家按推荐方式部署，每张卡能干多少活"。
- **B 层：对齐配置**，把能对齐的都对齐，回答"同样的卡、同样的条件下，谁的实现更快"。

| 维度 | 对齐方式 |
|---|---|
| 卡数 | **主线对比用 4 卡**，即 InferenceX 在 B200 上测 DeepSeek-V4.1-Flash 用的卡数（SGLang `tp4`、vLLM `tp4`、DeepInfer EP4）。8 卡（SGLang `tp8`、vLLM `tp8`、DeepInfer EP8）作补充 |
| 权重 | 同一份 checkpoint（FP8 加 FP4 专家），三家都不另外量化 |
| 投机解码 | 两组：全关（DeepInfer `--nextn 0`、SGLang `*-nospec`、vLLM 不开 spec），或全用 DSpark 5-token block（DeepInfer `--nextn 5`、SGLang `tp4`、vLLM `tp4-spec`）。开投机解码时只用 ShareGPT 文本。V4.1 没有 MTP 模块，DSpark 是唯一的投机方法。验证策略可能不同（如 vLLM 在 NVIDIA 上默认开的 adaptive verification），要记录 |
| prefill chunk | 16384（DeepInfer `--max-num-batched-tokens`、vLLM `tp4`/`tp8` 的 `--max-num-batched-tokens`；SGLang 默认值也是 16384，日志里已确认） |
| 最大 batch | 并发上限 ≥ 128（SGLang `tp4`/`tp8` cuda graph bs 256、vLLM `tp4`/`tp8` max-num-seqs 256；官方参考的 vLLM max-num-seqs 8 只适合低并发） |

- **C 层：相同并行策略（消融）**：让 SGLang 和 vLLM 也用 DeepInfer 的并行策略，即 attention DP + MoE EP + DeepEP（两家的正式版镜像里都带了 DeepEP）。用来把差距拆成"并行策略不同"和"同样策略下实现的快慢"两部分。A/B 层仍然用各家推荐的配置：DeepEP 是 DeepInfer 设计里固有的一部分，而 InferenceX 自己的 B200 recipe 也没有用 DeepEP（SGLang 用 `moe-a2a-backend: none`，vLLM DEP 用 MegaMoE）；把一家的内部实现强加给另一家，并不会更公平。

### 2.4 调度参数会拿 TTFT 换 decode 吞吐：必须报告 TTFT

DeepInfer 的 `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL`（引擎默认 100）在其他 rank 空闲时，每 N 个 decode step 才接纳一次等待中的 prompt。4 卡、DSpark 5、`sg-1k-1k` 实测：interval 100 对 2，c=128 时每卡 3003 对 2658 tok/s，TPOT 7.6 对 10.3ms，但 TTFT 是 1491 对 172ms（c=1 时 552 对 73ms）。任何引擎都可以这样拿 TTFT 换 decode 吞吐（SGLang 的 `--prefill-decode-interval`、vLLM 的 chunked prefill 预算起同样的作用），所以：

- 每张对比表都要在吞吐和 TPOT 旁边带上 TTFT。
- 主结论要在 TTFT 相当（或满足 TTFT SLO）的条件下给出，不能只按相同并发比。
- `bench/run_deepinfer.py` 默认 `--prefill-any-interval 2`（接近服务端行为），并记录在 `engine_args` 里；interval 100 保留作参考点。

### 2.5 必须记录、但不强行对齐的差异

这些属于各家实现的一部分，不强行对齐，但要写在结果旁边：

- **KV cache / indexer 精度**：vLLM 参考命令用 `--kv-cache-dtype fp8`、indexer `mxfp4`；DeepInfer 和 SGLang 用各自默认值，从启动日志里查清楚并记下来。精度会影响速度，也可能影响输出质量。
- **engram CPU 卸载**：vLLM 参考命令把 engram 放 CPU，省显存但可能更慢。
- **HTTP 开销**：SGLang/vLLM 走 HTTP + 流式 + 反 tokenize，DeepInfer 是进程内运行。大并发下可以忽略；c=1 时会让 DeepInfer 的 TTFT/TPOT 略占便宜。
- **EPLB**：DeepInfer 默认不开（`--enable-eplb` 可开），对齐配置里 SGLang/vLLM 都没开 EPLB。

### 2.6 速度之外：先确认输出质量等价

只有输出质量相当，速度对比才有意义。至少抽查同一批 prompt 的输出（不乱码、不重复循环）；最好跑一个小评测集（如 GSM8K 子集）。DeepInfer 自己的评测见它仓库里的 `eval/README.md`。

## 3. 和 InferenceX 对比：合成接受长度

InferenceX 公开的 AgentX 吞吐数字**不是**用真实投机验证测出来的。每个模型、thinking 模式和草稿长度都有一个固定的 "golden" 接受长度（AL），在 SPEED-Bench Qualitative 的 coding 子集上测得，吞吐测试直接注入这个值（`SemiAnalysisAI/InferenceX` 的 `infx/golden_al_distribution/`）。数值见上方英文表格。

AL 包含验证 token。AgentX 开 thinking，所以 DSpark 5 用 **3.51**。各家的注入方式：

- vLLM：`--speculative-config` 里加 `"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 3.51`
- SGLang：`SGLANG_SIMULATE_ACC_LEN=3.51`、`SGLANG_SIMULATE_ACC_METHOD=match-expected`、`SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`
- DeepInfer：目前没有这个开关，计划在 #5 里打补丁

本仓库的规则：

- 凡是要和 InferenceX 比的数字，必须用同一个 golden AL。
- 真实接受的结果单独报告，它反映各家在这种流量上实际的草稿质量。
- 合成 AL 和真实 AL 的点不能画在同一条曲线上。

## 4. 每次跑之前的检查清单

- 机器上只有一个系统在占 GPU（跑 DeepInfer 前先 `scripts/stop_servers.sh`）
- `nvidia-smi` 确认 GPU 空闲、没有残留进程
- 记录镜像 tag + digest（`docker inspect`）、DeepInfer commit、驱动版本
- 结果里 `num_failed == 0`、`num_output_len_mismatch == 0`
- 同一个点至少跑两次，差异 < 3% 算稳定，否则跑三次取中位数

</details>
