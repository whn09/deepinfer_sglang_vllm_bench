**EPLB rearrangement is hard-wired on in the server, and under online traffic it stalls all ranks for a large share of wall time**

`serve/openai_server.py:95` calls `setup_common_env(..., enable_eplb=True)`. That sets `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL=30` **and** `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE=1`, so experts are rearranged every 30 steps plus right after every prefill step. Offline benchmarks rarely prefill; a server under agentic/multi-turn traffic prefills for nearly every request, so the rearrangement fires almost continuously.

Setup: 8×B200, DeepSeek-V4.1-Flash, `--dp 4` (EP4), MTP 5, SemiAnalysis AgentX traces (multi-turn, ~100K-token prompts), 3600 s per point. Rank 0 times each `[EPLB] rearranged ... time=` between two `torch.cuda.synchronize()` calls (all_reduce + expert-weight shuffle), and every rank waits for it:

| concurrency | rearrangements | sum of `time=` | wall | share |
|---|---|---|---|---|
| 1 | 3778 | 1050 s | ~3750 s | 28% |
| 16 | 6025 | 1873 s | ~3900 s | 48% |
| 128 | 6216 | 2108 s | ~4125 s | 51% |

Each costs ~0.34 s (0.035 s + 0.30 s for the two layout groups), while 30 decode steps take only ~0.28 s (~9.3 ms/step).

Same server, same traces, concurrency 128, only the EPLB flags changed (A/B, 3600 s each; the EPLB-off run logged 0 rearrangements):

| c=128 | EPLB on (current) | EPLB off | change |
|---|---|---|---|
| total tok/s/GPU | 79,675 | 161,557 | **2.03×** |
| interactivity p50 (tok/s/user) | 24.6 | 111.5 | **4.54×** |
| TTFT p50 / p90 | 0.79 s / 1.67 s | 0.44 s / 1.19 s | |
| E2E latency p50 / p90 | 18.3 s / 83.0 s | 5.2 s / 23.9 s | |

With EPLB off, the server matches vLLM's published InferenceX B200 point for the same workload (0.97× throughput, 0.96× interactivity).

**Suggestion:** make it a server flag and default it off (or at least default `after_prefill` off for serving):
```python
p.add_argument("--eplb-interval", type=int, default=0, help="DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL; 0 disables EPLB rearrangement")
p.add_argument("--eplb-after-prefill", type=int, default=0, choices=[0, 1])
...
setup_common_env(..., enable_eplb=args.eplb_interval > 0)
os.environ["DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL"] = str(args.eplb_interval)
os.environ["DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE"] = str(args.eplb_after_prefill)
```
Rearranging asynchronously, or only when the measured imbalance exceeds a threshold, would also avoid paying the shuffle when the layout barely changes.

<details><summary>中文</summary>

server 写死了 `enable_eplb=True`，效果是每 30 步重排一次专家，并且每个 prefill 之后都重排。在线服务下几乎每个请求都有 prefill，所以重排几乎是持续发生的。在 8×B200、DP4/EP4、AgentX 轨迹上实测，rank 0 记录的重排耗时（两次 `cuda.synchronize` 之间、所有 rank 都在等）占墙钟时间 28%（c=1）到 51%（c=128）：每次约 0.34 s，而 30 个 decode step 才约 0.28 s。同一个 server、同一批轨迹、c=128，只改 EPLB 开关做 A/B（各跑 3600 s，关闭那次一次重排都没有）：总吞吐/GPU 提升 2.03×（79,675 → 161,557），interactivity p50 提升 4.54×（24.6 → 111.5 tok/s），TTFT p50 从 0.79 s 降到 0.44 s，E2E 延迟 p50 从 18.3 s 降到 5.2 s；关闭后与 InferenceX 公开的 vLLM B200 同负载结果基本持平（吞吐 0.97×，interactivity 0.96×）。建议改成 server 参数、默认关闭（至少默认关掉 after-prefill），或者改成异步重排，或者只在不均衡超过阈值时才重排。
</details>
