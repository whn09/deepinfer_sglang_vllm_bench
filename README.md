# DeepInfer vs SGLang vs vLLM: DeepSeek-V4.1-Flash on B200

Compare three inference stacks serving `DeepSeek-V4.1-Flash` on one 8×B200 box — same weights, same inputs, same metrics — in a way that can be checked against the public [InferenceX](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&i_prec=fp8) benchmark.

**Roadmap and status: [#8](https://github.com/whn09/deepinfer_sglang_vllm_bench/issues/8).** So far only the DeepInfer smoke test has run (`results/20261021_smoke_deepinfer/`); the SGLang/vLLM paths have not been exercised yet.

## Two workloads

| | InferenceX AgentX (primary) | Fixed-length closed loop |
|---|---|---|
| Who | SGLang, vLLM; DeepInfer after #5 | DeepInfer, SGLang, vLLM |
| Load | Real coding-agent traces: multi-turn, long context, cross-request prefix reuse | Fixed input/output length, no prefix reuse, closed-loop concurrency 1–128 |
| Why | Comparable to the published B200 points | Controlled engine-level comparison; the only one DeepInfer can join today |
| Docs | `docs/agentx.md` | `docs/fairness.md` |

DeepInfer's open-source release is offline-only: no HTTP server, no cross-request KV cache, no DP load balancing. Getting it onto AgentX is tracked in #5.

## Layout

```
bench/
  common.py          dataset / record format + metric computation (shared)
  make_dataset.py    fixed-length request files as token ids (random or ShareGPT text)
  client_openai.py   SGLang / vLLM client: streaming /v1/completions, closed-loop concurrency
  run_deepinfer.py   DeepInfer driver: LLMEngine directly, same closed-loop semantics
  summarize.py       markdown tables + throughput-vs-interactivity Pareto plots
scripts/
  env.sh             every path and the DeepInfer environment (edit this when the host changes)
  scenarios.sh       scenarios (input/output length) and concurrency points
  make_datasets.sh   per-concurrency datasets for one scenario
  serve_sglang.sh    SGLang server (ref / ref-nospec / tp8 / tp8-nospec)
  serve_vllm.sh      vLLM server (ref / ref-spec / tp8 / tp8-spec)
  sweep_server.sh    sweep one scenario against a running server
  sweep_deepinfer.sh sweep one scenario on DeepInfer
  run_agentx.sh      InferenceX AgentX against SGLang / vLLM
  stop_servers.sh    stop server containers, free the GPUs
  pull_results.sh    (run locally) pull results into results/, optionally wipe the remote temp dir
docs/
  setup_deepinfer.md   DeepInfer build and model download, with every pitfall we hit
  fairness.md          fair-comparison rules and metric definitions  <- read first
  reference_configs.md official reference commands and how the three configs differ
  agentx.md            InferenceX AgentX and what it takes to run DeepInfer on it
results/               pulled-back results, one directory per run
```

## Quick start (fixed-length)

The B200 host is not fixed. Everything lives under `/opt/dlami/nvme/bench_tmp` (`$WORK`); pull the results back and delete it afterwards.

```bash
# 0. local: sync the repo to the host's temp dir
rsync -av --exclude results --exclude .git ./ B200-1:/opt/dlami/nvme/bench_tmp/repo/

# 1. on the host: DeepInfer built per docs/setup_deepinfer.md, model downloaded
ssh B200-1
cd /opt/dlami/nvme/bench_tmp/repo && source scripts/env.sh

# 2. DeepInfer (no server may hold the GPUs) — main comparison is 4 GPUs, like InferenceX
bash scripts/stop_servers.sh
bash scripts/sweep_deepinfer.sh deepinfer-ep4-nospec  4 0 sg-1k-1k
bash scripts/sweep_deepinfer.sh deepinfer-ep4-dspark5 4 5 sg-1k-1k

# 3. SGLang
bash scripts/serve_sglang.sh tp4-nospec && bash scripts/sweep_server.sh sglang sglang-tp4-nospec 4 sg-1k-1k
bash scripts/serve_sglang.sh tp4        && bash scripts/sweep_server.sh sglang sglang-tp4        4 sg-1k-1k
bash scripts/stop_servers.sh

# 4. vLLM
bash scripts/serve_vllm.sh tp4      && bash scripts/sweep_server.sh vllm vllm-tp4      4 sg-1k-1k
bash scripts/serve_vllm.sh tp4-spec && bash scripts/sweep_server.sh vllm vllm-tp4-spec 4 sg-1k-1k
bash scripts/stop_servers.sh

# 5. summarize
python bench/summarize.py $RESULTS_DIR/results.jsonl

# 6. local: pull results back and wipe the remote temp dir
bash scripts/pull_results.sh B200-1 --clean
```

The 8-GPU supplement uses `tp8` and `deepinfer-ep8-*` (`sweep_deepinfer.sh ... 8 ...`). The official reference configs (tier A) use `ref`: SGLang `ref` is TP4/EP4 with CUDA graphs only up to bs 64; vLLM `ref` uses 2 GPUs with max-num-seqs 8, so restrict it to low concurrency with `CONCURRENCIES="1 2 4 8"`.

AgentX: see `docs/agentx.md`.

<details><summary>中文</summary>

# DeepInfer vs SGLang vs vLLM：DeepSeek-V4.1-Flash 在 B200 上的对比

在一台 8×B200 机器上对比三个推理框架运行 `DeepSeek-V4.1-Flash` 的表现：同一份权重、同一批输入、同一套指标，并且结果能和公开的 [InferenceX](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&i_prec=fp8) benchmark 核对。

**Roadmap 和进度：[#8](https://github.com/whn09/deepinfer_sglang_vllm_bench/issues/8)。** 目前只跑过 DeepInfer 的冒烟测试（`results/20261021_smoke_deepinfer/`），SGLang/vLLM 这边还没跑过。

## 两种负载

| | InferenceX AgentX（主线） | 定长闭环 |
|---|---|---|
| 参与者 | SGLang、vLLM；DeepInfer 在 #5 完成之后 | DeepInfer、SGLang、vLLM |
| 负载 | 真实编程 agent 轨迹：多轮、长上下文、跨请求前缀复用 | 固定输入/输出长度、无前缀复用、闭环并发 1–128 |
| 用途 | 能和公开的 B200 数据点比 | 受控的引擎级对比；也是 DeepInfer 目前唯一能参加的 |
| 文档 | `docs/agentx.md` | `docs/fairness.md` |

DeepInfer 开源版只能离线推理：没有 HTTP server、没有跨请求 KV cache、没有 DP 负载均衡。让它跑 AgentX 的工作在 #5 跟踪。

## 目录

```
bench/
  common.py          数据集/记录格式 + 指标计算（三家共用）
  make_dataset.py    生成定长请求文件，存 token ids（random 或 ShareGPT 真实文本）
  client_openai.py   SGLang / vLLM 压测 client：/v1/completions 流式，闭环固定并发
  run_deepinfer.py   DeepInfer 压测：直接驱动 LLMEngine，闭环语义相同
  summarize.py       markdown 表 + 吞吐-交互性 Pareto 图
scripts/
  env.sh             所有路径和 DeepInfer 环境变量（换机器只改这里）
  scenarios.sh       场景（输入/输出长度）和并发点
  make_datasets.sh   为一个场景生成每个并发点的数据集
  serve_sglang.sh    起 SGLang server（ref / ref-nospec / tp8 / tp8-nospec）
  serve_vllm.sh      起 vLLM server（ref / ref-spec / tp8 / tp8-spec）
  sweep_server.sh    对已起好的 server 扫一个场景
  sweep_deepinfer.sh DeepInfer 扫一个场景
  run_agentx.sh      用 InferenceX AgentX 压 SGLang / vLLM
  stop_servers.sh    停 server 容器、释放 GPU
  pull_results.sh    （本地运行）把结果拉回 results/，可选清理远端临时目录
docs/
  setup_deepinfer.md   DeepInfer 编译与模型下载（含所有踩过的坑）
  fairness.md          公平对比的规则和指标定义  <- 先读这个
  reference_configs.md 官方参考命令和三家配置的差异
  agentx.md            InferenceX AgentX，以及 DeepInfer 要跑它还缺什么
results/               拉回来的结果，每次运行一个目录
```

## 快速开始（定长）

B200 机器不固定。所有东西都放在 `/opt/dlami/nvme/bench_tmp`（`$WORK`），跑完把结果拉回本地，再删掉远端目录。命令见上方英文部分，各步骤含义：

0. 本地：把仓库同步到 B200 的临时目录
1. B200 上：DeepInfer 已按 `docs/setup_deepinfer.md` 编译好、模型已下载
2. DeepInfer（先确认没有 server 占着 GPU）
3. SGLang
4. vLLM
5. 汇总
6. 本地：拉回结果并清理远端临时目录

主线对比用 4 卡（和 InferenceX 一致），命令见上方英文部分。8 卡补充用 `tp8` 和 `deepinfer-ep8-*`（`sweep_deepinfer.sh ... 8 ...`）。官方参考配置（A 层）用 `ref`：SGLang `ref` 是 TP4/EP4，但 cuda graph 只到 bs 64；vLLM `ref` 用 2 卡、max-num-seqs 8，只适合低并发，用 `CONCURRENCIES="1 2 4 8"` 限定并发点。

AgentX：见 `docs/agentx.md`。

</details>
