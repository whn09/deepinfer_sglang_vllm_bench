# DeepInfer vs SGLang vs vLLM：DeepSeek-V4.1-Flash 在 B200 上的对比评测

在同一台 8×B200 机器上，用同一份 `DeepSeek-V4.1-Flash-1017` 权重、同一批输入，对比三个推理框架的吞吐和延迟。

> **状态：** 框架和文档已完成，**还没有实测数据**。脚本还没在 B200 上端到端跑通过，第一次跑时请留意报错。

## 目录

```
bench/
  common.py          数据集/记录格式 + 指标计算（三家共用）
  make_dataset.py    生成定长请求文件（token ids；random 或 sharegpt 真实文本）
  client_openai.py   SGLang / vLLM 压测 client（/v1/completions 流式，闭环固定并发）
  run_deepinfer.py   DeepInfer 压测（直接驱动 LLMEngine，闭环固定并发，口径同上）
  summarize.py       汇总成 markdown 表 + 吞吐-交互性 Pareto 图
scripts/
  env.sh             所有路径和 DeepInfer 环境变量（换机器只改这里）
  scenarios.sh       场景（输入/输出长度）和并发点定义
  make_datasets.sh   为一个场景生成每个并发点的数据集
  serve_sglang.sh    起 SGLang server（ref / ref-nospec / tp8 / tp8-nospec）
  serve_vllm.sh      起 vLLM server（ref / tp8）
  sweep_server.sh    对已起好的 server 扫一个场景
  sweep_deepinfer.sh DeepInfer 扫一个场景
  run_agentx.sh      用 InferenceX AgentX 压 SGLang / vLLM
  stop_servers.sh    停 server 容器、释放 GPU
  pull_results.sh    （本地运行）把结果拉回本仓库 results/，可选清理远端
docs/
  setup_deepinfer.md   DeepInfer 编译与模型下载（含所有踩坑）
  fairness.md          公平对比的原则与口径 ← 先读这个
  reference_configs.md 官方参考命令与三家配置差异
  agentx.md            InferenceX AgentX 评测，以及为什么它不能直接测 DeepInfer
```

## 两条评测线

| | 定长闭环场景（主线） | InferenceX AgentX |
|---|---|---|
| 参与者 | DeepInfer、SGLang、vLLM | SGLang、vLLM |
| 负载 | 固定输入/输出长度，无前缀复用，闭环并发 1~128 | 真实编程 agent 多轮轨迹，跨请求前缀复用是核心 |
| 用途 | 三家引擎本身的快慢 | SGLang vs vLLM；拿 InferenceX 公布的数字校准本机环境 |
| 为什么 | DeepInfer 开源版没有 HTTP server、没有跨请求 KV 复用 | 见 `docs/agentx.md` |

## 快速开始

B200 机器不固定，所有东西都放在 `/opt/dlami/nvme/bench_tmp`（`$WORK`），跑完拉回本地再删。

```bash
# 0. 本地：把仓库同步到 B200 的临时目录
rsync -av --exclude results --exclude .git ./ B200-1:/opt/dlami/nvme/bench_tmp/repo/

# 1. B200 上：DeepInfer 已按 docs/setup_deepinfer.md 编译好、模型已下载
ssh B200-1
cd /opt/dlami/nvme/bench_tmp/repo && source scripts/env.sh

# 2. DeepInfer（先确认没有 server 在占卡）
bash scripts/stop_servers.sh
bash scripts/sweep_deepinfer.sh deepinfer-ep8-nospec 8 0 sg-1k-1k
bash scripts/sweep_deepinfer.sh deepinfer-ep8-dspark5 8 5 sg-1k-1k

# 3. SGLang
bash scripts/serve_sglang.sh tp8-nospec
bash scripts/sweep_server.sh sglang sglang-tp8-nospec 8 sg-1k-1k
bash scripts/serve_sglang.sh tp8
bash scripts/sweep_server.sh sglang sglang-tp8 8 sg-1k-1k
bash scripts/stop_servers.sh

# 4. vLLM
bash scripts/serve_vllm.sh tp8
bash scripts/sweep_server.sh vllm vllm-tp8 8 sg-1k-1k
bash scripts/serve_vllm.sh tp8-spec
bash scripts/sweep_server.sh vllm vllm-tp8-spec 8 sg-1k-1k
bash scripts/stop_servers.sh

# 5. 汇总
python bench/summarize.py $RESULTS_DIR/results.jsonl

# 6. 本地：拉回结果并清理远端
bash scripts/pull_results.sh B200-1 --clean
```

官方参考配置（A 层）把上面的 `tp8` 换成 `ref`：SGLang `ref` 用 4 卡，对应 DeepInfer `deepinfer-ep4-*`（`sweep_deepinfer.sh ... 4 ...`）；vLLM `ref` 用 2 卡、max-num-seqs 8，只适合低并发，可以用 `CONCURRENCIES="1 2 4 8"` 限定并发点。

## 跑之前要先确认的事

- **SGLang/vLLM 镜像能不能起**：`lmsysorg/sglang:dev-dsv41`、`vllm/vllm-openai:latest` 是否支持当前 checkpoint，第一次起服务看日志。
- **vLLM 的 DSpark 能不能起**：`*-spec` 档用 `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`，NVIDIA 上默认开 adaptive verification；起不来先试 `VLLM_ADAPTIVE_VERIFICATION=false`（见 `scripts/serve_vllm.sh`）。
- **各家 KV cache 精度**：从启动日志里查清楚并记录，见 `docs/fairness.md` §2.4。
- **SGLang 投机解码接受率的指标名**：`client_openai.py` 会把 `/metrics` 里名字带 spec/accept/draft 的计数器增量原样存进 `server_metrics_delta`，需要人工核对。
