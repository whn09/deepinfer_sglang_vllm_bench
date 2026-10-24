Two server defaults cost decode speed on AgentX-style traffic (long multi-turn prompts, B200, `--dp 4`, EPLB off, synthetic AL 3.51). Both are one-line changes.

**1. Redundant experts with EPLB off.** `setup_common_env(set_override_experts=True)` sets `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` to 32. They exist for EPLB rearrangement, and with EPLB off (the default now) they only add expert weight per rank.

| `examples/benchmark.py`, decode SPS (150K / 64K context) | 32 extra | 0 extra |
|---|---|---|
| 4 requests per rank | 46.4 | 48.7 (+4.9%) |
| 16 requests per rank | 36.3 | 37.7 (+3.9%) |

The MLP drops from 14.9 to 13.8 ms. On AgentX c=64, interactivity p50 is +4.6% and throughput +1.8%. Suggestion: when EPLB is off, set `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` to 0 (e.g. `os.environ.setdefault("DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS", "0")` before `setup_common_env` when `args.eplb_interval == 0`). That is what we measured; we did not test leaving the variable unset.

**2. `--prefill-any-interval 2`.** In the NORMAL engine a prompt step is collective, and no rank decodes in it. With an ANY point every 2 decode steps, almost every arriving request pauses all DP ranks' decode. At high concurrency this dominates per-user speed. AgentX, extra experts 0, 900 s:

| | interval 2 | 30 | 100 (engine default) |
|---|---|---|---|
| c=64 interactivity p50 / TTFT p50 | 190 / 0.14 s | 207 / 0.45 s | 227 / 1.02 s |
| c=256 tok/s/GPU / interactivity p50 / TTFT p50 | 211K / 55 / 0.70 s | – | 255K / 119 / 2.53 s |
| c=16 interactivity p50 / TTFT p50 | 322 / 0.10 s | – | 320 / 0.43 s (interval 60) |

It is a TTFT ↔ interactivity trade-off that depends on load. Suggestion: keep the flag, but document it that way. A load-aware default would be better still, e.g. scaling the ANY interval with the number of running sequences per rank, or skipping the ANY point while every rank still has decode work. 3600 s numbers for the whole curve: https://github.com/whn09/deepinfer_sglang_vllm_bench/blob/main/results/20261024_agentx_v3/summary.md

<details><summary>中文</summary>

在 AgentX 这类负载下（长的多轮 prompt，B200，`--dp 4`，EPLB 关闭，合成接受长度 3.51），server 有两个默认值拖慢了 decode，改起来都只需要一行。

**1. EPLB 关闭时的冗余专家：** `setup_common_env(set_override_experts=True)` 会把 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 设成 32。这些冗余专家是给 EPLB 重排用的；EPLB 关闭时（现在的默认），它们只会增加每个 rank 的专家权重。设成 0 后，decode SPS 在每 rank 4 个请求时 +4.9%，16 个请求时 +3.9%，MLP 从 14.9 ms 降到 13.8 ms；AgentX c=64 下 interactivity p50 +4.6%，吞吐 +1.8%。建议在 EPLB 关闭时把 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 设为 0（比如在 `args.eplb_interval == 0` 时，于 `setup_common_env` 之前执行 `os.environ.setdefault("DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS", "0")`）。我们实测的就是这种设法；不设置这个变量的情况没有测过。

**2. `--prefill-any-interval 2`：** NORMAL 引擎的 prompt step 是全组一起做的，这一步里所有 rank 都不 decode。每 2 个 decode step 就有一个 ANY 点，意味着几乎每来一个新请求，所有 DP rank 的 decode 都会被打断。在高并发下，这成了决定每用户速度的主要因素（数据见上表）。这是一个随负载变化的 TTFT 与 interactivity 之间的取舍。建议保留这个参数，但在文档里写清楚这一点；更好的做法是按负载自动调整，比如让 ANY 间隔随每 rank 运行中的请求数增长，或者在每个 rank 都还有 decode 任务时跳过 ANY 点。
</details>
