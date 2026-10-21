# InferenceX AgentX

[InferenceX](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&i_prec=fp8) is SemiAnalysis's open inference benchmark. Its **AgentX** scenario replays real Claude Code sessions ([agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness), a fork of NVIDIA aiperf). It is the primary workload of this repo because it lets us check our numbers against the published B200 points. Plan: #4 (SGLang, vLLM) and #5 (DeepInfer).

## 1. What AgentX measures

- Corpus: `semianalysisai/cc-traces-weka-062126` (public HF dataset, coding sessions with subagents). The traffic **shape** is real (prompt sizes, prefix sharing, subagent fan-out, inter-turn timing); prompt **content** is synthesized to exact token counts.
- Multi-turn long context: every turn carries the whole session so far, so **cross-request KV prefix reuse** is the core of the load.
- Streaming OpenAI `/v1/chat/completions`, `ignore_eos=true`, `--cache-bust first_turn_prefix` so distinct sessions cannot borrow each other's cache.
- Each concurrency point profiles for 3600 s by default, plus up to 1800 s of cache warmup; the result JSON carries `submission_valid`.
- Metrics: throughput per GPU vs interactivity — the same axes as this repo's fixed-length Pareto plots.

## 2. How InferenceX runs DeepSeek-V4.1-Flash on B200

From `SemiAnalysisAI/InferenceX` (`inferencex-e2e/configs/nvidia-master.yaml` and `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/{sglang,vllm}/b200-fp4-mtp/agentic.yaml`):

| | SGLang | vLLM |
|---|---|---|
| Image (pinned nightly) | `lmsysorg/sglang:nightly-dev-cu13-20261014-684fea8a` | `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7` |
| Parallelism by concurrency | TP4 (c=1,4,8); TP4/EP4 (c=16,24); TP4/EP4 + DP-attention + HiCache DRAM offload + `sglang-router` (c=48–256) | TP4 (c=1–128); DEP2 (c=8–32) and DEP4 (c=64,128) behind `vllm-router` |
| Speculation | DSPARK, block 5 | `dspark`, 5 tokens, probabilistic draft, block rejection, adaptive verification off |
| Thinking | on, reasoning effort high | on |
| Acceptance | synthetic, golden AL 3.51 | synthetic, golden AL 3.51 |

Golden AL: see `docs/fairness.md` §3.

**Our deviations (decided 2026-10-21):** model `deepseek-ai/DeepSeek-V4.1-Flash-1017` instead of `deepseek-ai/DeepSeek-V4.1-Flash` (same format, one day newer), and the latest official release images pinned by version (SGLang `v0.5.22-cu130`, vLLM `v0.32.0`) instead of InferenceX's nightlies; a dedicated image (e.g. `lmsysorg/sglang:dev-dsv41`) only if a release cannot serve the model. Everything else follows the recipes. These deviations must be stated next to every number compared with InferenceX.

## 3. DeepInfer on AgentX

The open-source DeepInfer release lacks what AgentX needs (`README.md`, "not included in this release"; `llm_engine.py` docstring):

1. **No online server**: only offline `LLM` / `LLMEngine`, no OpenAI-compatible HTTP server.
2. **No KV cache system**: no cross-request prefix reuse.
3. **No DP load balancing**: no router with session affinity across data-parallel ranks.

(1) and (3) are straightforward to build. (2) is the real issue: each AgentX turn carries tens to hundreds of thousands of history tokens; SGLang (radix cache / HiCache) and vLLM (prefix caching) reuse the history's KV, while DeepInfer would re-prefill the whole history every turn. That would measure the missing feature, not the engine. #5 tracks the work: server, session-affinity routing, a prefix-reuse feasibility study, and a synthetic-acceptance patch. Until then, DeepInfer joins only the fixed-length comparison (`docs/fairness.md`).

## 4. Running it

```bash
source scripts/env.sh
bash scripts/serve_sglang.sh ref           # or serve_vllm.sh ref
bash scripts/run_agentx.sh sglang sglang-ref 8 32
```

- `run_agentx.sh` creates a dedicated venv at `$WORK/agentx-venv` and installs `agentx-harness==1.0.7` (it bundles a forked aiperf, so it must not share an environment). It needs Python 3.11–3.13; the `/opt/pytorch` venv that `env.sh` activates is 3.13, the system 3.14 is not supported.
- The corpus defaults to the date-pinned `semianalysis_cc_traces_weka_062126`; do not use the rolling `..._with_subagents` alias (runs on different drops are not comparable).
- Results land in `$RESULTS_DIR/agentx/<config>/c<concurrency>/`; check `submission_valid` in `profile_export_aiperf.json`.
- For a quick smoke, `AGENTX_EXTRA_ARGS="--benchmark-duration 600"` — but such a run is not a valid submission and cannot be compared with published numbers.
- The model's max context is 1M (`max_position_embeddings 1048576`) and the corpus has requests over 256k. If the server is configured for less, pass `--max-context-length <value>` or use the `_256k` corpus variant; otherwise over-length requests fail server-side and more than 1% of them makes `submission_valid=false`.
- The current `serve_*.sh` `ref` / `tp8` profiles are not yet the InferenceX recipes; porting them (`ix-*` profiles) is part of #4.

<details><summary>中文</summary>

# InferenceX AgentX

[InferenceX](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&i_prec=fp8) 是 SemiAnalysis 的开源推理基准。其中 **AgentX** 场景回放真实的 Claude Code 会话（[agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness)，基于 NVIDIA aiperf 的 fork）。它是本仓库的主负载，因为可以拿我们的数字和公开的 B200 数据点核对。计划：#4（SGLang、vLLM）和 #5（DeepInfer）。

## 1. AgentX 测什么

- 语料：`semianalysisai/cc-traces-weka-062126`（HF 公开数据集，带子 agent 的编程会话）。负载**形状**是真实的（prompt 长度、前缀共享、子 agent 扇出、轮次间隔），prompt **内容**是按 token 数精确合成的。
- 多轮长上下文：每一轮都带着会话到目前为止的全部内容，所以**跨请求的 KV 前缀复用**是这个负载的核心。
- 走 OpenAI `/v1/chat/completions` 流式接口，`ignore_eos=true`，`--cache-bust first_turn_prefix` 防止不同会话之间互相蹭缓存。
- 每个并发点默认测量 3600s，外加最长 1800s 缓存预热；结果 json 里有 `submission_valid` 字段。
- 指标：每卡吞吐 vs 交互性，和本仓库定长场景的 Pareto 图同一种坐标。

## 2. InferenceX 在 B200 上怎么跑 DeepSeek-V4.1-Flash

来自 `SemiAnalysisAI/InferenceX`（`inferencex-e2e/configs/nvidia-master.yaml` 和 `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/{sglang,vllm}/b200-fp4-mtp/agentic.yaml`），见上方英文表格：两家都钉了 nightly 镜像；按并发区间换并行方式（SGLang 高并发用 DP-attention + HiCache DRAM 卸载 + router，vLLM 用 DEP2/DEP4 + router）；DSpark 5；开 thinking；接受长度用合成的 golden AL 3.51。

golden AL 见 `docs/fairness.md` §3。

**我们的差异（2026-10-21 决定）：** 模型用 `deepseek-ai/DeepSeek-V4.1-Flash-1017` 而不是 `deepseek-ai/DeepSeek-V4.1-Flash`（格式相同，新一天的版本）；镜像用最新正式发行版并钉版本号（SGLang `v0.5.22-cu130`、vLLM `v0.32.0`），而不是 InferenceX 的 nightly；正式版跑不了这个模型时才用专用镜像（如 `lmsysorg/sglang:dev-dsv41`）。其余都按 recipe。凡是和 InferenceX 比的数字，旁边都要写明这些差异。

## 3. DeepInfer 跑 AgentX

DeepInfer 开源版缺 AgentX 需要的东西（`README.md` 写明 "not included in this release"；`llm_engine.py` 的 docstring）：

1. **没有在线服务**：只有离线 `LLM` / `LLMEngine`，没有 OpenAI 兼容的 HTTP server。
2. **没有 KV cache 系统**：不能跨请求复用前缀。
3. **没有 DP 负载均衡**：没有在各数据并行 rank 之间按会话亲和分发的 router。

(1) 和 (3) 补起来不难。(2) 才是真正的问题：AgentX 每一轮都带着几万到几十万 token 的历史，SGLang（radix cache / HiCache）和 vLLM（prefix caching）能复用历史的 KV，DeepInfer 每一轮都要把整段历史重新 prefill。这样测出来的是"缺不缺这个功能"，不是引擎本身。#5 跟踪这部分工作：server、会话亲和路由、前缀复用可行性分析、合成接受长度补丁。在那之前，DeepInfer 只参加定长对比（`docs/fairness.md`）。

## 4. 运行

命令见上方英文部分。

- `run_agentx.sh` 会在 `$WORK/agentx-venv` 建独立 venv，安装 `agentx-harness==1.0.7`（它带的是 fork 版 aiperf，不能和别的环境混装）。需要 Python 3.11–3.13：`env.sh` 激活的 `/opt/pytorch` venv 是 3.13，可以用；系统自带的 3.14 不行。
- 语料默认用钉日期的 `semianalysis_cc_traces_weka_062126`，不要用滚动别名 `..._with_subagents`（不同批次的结果不可比）。
- 结果在 `$RESULTS_DIR/agentx/<config>/c<并发>/`，看 `profile_export_aiperf.json` 里的 `submission_valid`。
- 快速冒烟可以用 `AGENTX_EXTRA_ARGS="--benchmark-duration 600"`，但这样不是有效提交，不能和公开数字比。
- 模型最大上下文是 1M（`max_position_embeddings 1048576`），语料里有超过 256k 的请求。如果 server 配的上下文更短，要么加 `--max-context-length <值>`，要么换 `_256k` 版本的语料；否则超长请求会在 server 端失败，超过 1% 就会 `submission_valid=false`。
- 目前 `serve_*.sh` 的 `ref` / `tp8` 档还不是 InferenceX 的 recipe；把它们移植成 `ix-*` 档是 #4 的一部分。

</details>
