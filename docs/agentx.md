# InferenceX AgentX 评测

[InferenceX](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&i_prec=fp8) 是 SemiAnalysis 的开源推理基准，其中 **AgentX** 场景用真实 Claude Code 编程会话的轨迹做负载（[agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness)，基于 NVIDIA aiperf 的 fork）。

## 1. AgentX 测的是什么

- 语料：`semianalysisai/cc-traces-weka-062126`（HF 公开数据集，带子 agent 的编程会话）。负载**形状**是真实的（prompt 长度、前缀共享、子 agent 扇出、轮次间思考时间），prompt **内容**是按 token 数精确合成的。
- 多轮长上下文：同一个会话每一轮都带着前面所有内容，**跨请求的 KV 前缀复用**是负载的核心。
- 走 OpenAI `/v1/chat/completions` 流式接口，`ignore_eos=true`，`--cache-bust first_turn_prefix` 防止不同会话之间互相蹭缓存。
- 每个并发点默认 profiling 3600s，外加最长 1800s 缓存预热；结果 json 里有 `submission_valid` 字段标明这次跑法是否合规。
- 指标：每卡吞吐 vs 单用户交互性（interactivity），和本仓库定长场景的 Pareto 图同一种坐标。

## 2. 为什么 AgentX 不能直接用来测 DeepInfer

DeepInfer 开源版自己写明了缺这两样（`README.md` "not included in this release"、`llm_engine.py` 的 docstring）：

1. **没有在线服务封装**：只有离线 `LLM` / `LLMEngine`，没有 OpenAI 兼容的 HTTP server。
2. **没有 KV cache 系统**：没有跨请求的前缀复用。

第 1 点可以自己补一个 server 解决。但第 2 点是根本问题：AgentX 的每一轮都带着上万到几十万 token 的历史，SGLang（radix cache）和 vLLM（prefix caching）能直接复用前面的 KV，DeepInfer 只能每一轮把整段历史重新 prefill。这样测出来比的是"有没有 KV cache 系统"，而不是推理引擎本身的快慢，对 DeepInfer 不公平，也不是你想要的结论。

## 3. 怎么用 AgentX

1. **SGLang vs vLLM**：两家都有 server 和前缀缓存，用 AgentX 比是完全合规的。
2. **校准本机环境**：用和 InferenceX 相同的语料、相同的配置在 B200 上跑 SGLang/vLLM，结果和 InferenceX 网站上公布的 B200 数字对一下。对得上，说明我们这台机器、这套镜像和参数是对的，定长场景里 SGLang/vLLM 的数字也就可信。
3. **DeepInfer 三家同台**：用本仓库的定长闭环场景（`docs/fairness.md`），三家都不复用前缀。

如果以后要让 DeepInfer 也跑 AgentX，需要先给它做：OpenAI chat server（套 `encoding/encoding.py` 的 V4.1 prompt 格式）+ 跨请求前缀 KV 复用。这超出本仓库范围。

## 4. 运行

```bash
source scripts/env.sh
bash scripts/serve_sglang.sh ref           # 或 serve_vllm.sh ref
bash scripts/run_agentx.sh sglang sglang-ref 8 32
```

- `run_agentx.sh` 会在 `$WORK/agentx-venv` 建独立 venv 安装 `agentx-harness==1.0.7`（它带的是 fork 版 aiperf，不能和别的环境混装）。需要 Python 3.11–3.13，`env.sh` 激活的 `/opt/pytorch` 是 3.13，可以用；系统自带的 3.14 不行。
- 语料默认钉日期版本 `semianalysis_cc_traces_weka_062126`，不要用滚动别名 `..._with_subagents`（换批次之后结果不可比）。
- 结果在 `$RESULTS_DIR/agentx/<config>/c<并发>/`，看 `profile_export_aiperf.json` 里的 `submission_valid`。
- 快速冒烟可以 `AGENTX_EXTRA_ARGS="--benchmark-duration 600"`，但这样结果不是有效提交，不能和 InferenceX 公布的数字比。
- 模型最大上下文是 1M（`max_position_embeddings 1048576`），语料里有超过 256k 的请求。如果 server 配的上下文更短，要么加 `--max-context-length <值>`，要么换 `_256k` 版本的语料，否则超长请求会在 server 端报错，超过 1% 会导致 `submission_valid=false`。
