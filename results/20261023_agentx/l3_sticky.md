# Does the Mooncake Store L3 help the headline config? (#20)

Paired AgentX run, c=64, 900 s, sticky-balanced routing (the headline routing), fastokens. L3 on GPUs 0-3, L3 off on GPUs 4-7, at the same time. The image is `deepinfer:agentx-l3`: the L3 code with the circuit breaker and the TCP queue-limit env defaults, on the released 0.3.13.post1 wheel.

| | L3 off | L3 on |
|---|---|---|
| total tok/s/GPU | 92,648 | 93,167 (+0.6%) |
| interactivity p50 / mean | 178.0 / 156.9 | 185.3 / 164.1 |
| TTFT p50 / p90 s | 0.14 / 0.50 | 0.14 / 0.51 |
| prefix-cache token hit rate | 92.7% | 92.7% |
| L3 lookups / hits | – | 2782 / 0 |
| L3 puts (objects / GiB), failures | – | 70,202 / 40, 0 |
| queue-full / TRANSFER_FAIL / breaker trips | 0 / 0 / 0 | 0 / 0 / 0 |

**No.** With sticky routing each conversation stays on one DP rank, and the per-rank host pool (32 GB) holds everything AgentX re-reads within a run, so L3 is never consulted successfully. L3 matters when conversations move between ranks or instances. With least-loaded routing it lifts throughput 1.36× ([l3_mooncake_main.md](l3_mooncake_main.md)), to the level that sticky routing gets without it. It would also matter after a restart, across instances, or when the host pool is too small for the working set. None of these happen in this benchmark.

**The fix works on the released wheel.** With `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER` / `MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER` defaulted to 65535, 0.3.13.post1 has no queue-full rejections under this load. Without the defaults it failed within 30 s of the first requests (E1).

**900 s vs 3600 s.** These 900 s runs give 0.79× vLLM throughput, while the 3600 s headline gives 0.98× for the same server configuration. That is the known boundary effect of short runs (the first wave of cold prompts weighs more), so compare 900 s runs only with each other.

<details><summary>中文</summary>

# Mooncake Store L3 对主配置有帮助吗？（#20）

AgentX 配对测试：c=64，900 s，sticky-balanced 路由（主结果用的路由），开 fastokens。L3 on 跑在 GPU 0-3、L3 off 跑在 GPU 4-7，两组同时跑。镜像是 `deepinfer:agentx-l3`，即带熔断和 TCP 排队上限 env 默认值的 L3 代码，Mooncake 用发布版 0.3.13.post1 wheel。

**结论：没有帮助。** 吞吐 93.2K 对 92.6K（+0.6%），interactivity 和 TTFT 都在噪声范围内，token 命中率都是 92.7%，L3 命中 0 次。sticky 路由让每个对话固定在同一个 DP rank 上，每个 rank 32 GB 的 host 池已经装得下一次运行里 AgentX 会重复读取的全部前缀，所以 L3 从来没有成功派上用场。L3 有用的场景是对话会在 rank 或实例之间迁移：least-loaded 路由下它把吞吐提升到 1.36×，刚好补到 sticky 不开 L3 的水平。重启之后、多实例之间，或者 host 池装不下工作集时，它也会有用，但这些情况在这个 benchmark 里都不会出现。

**修复在发布版 wheel 上有效：** 把两个排队上限的环境变量默认设成 65535 之后，0.3.13.post1 在这个负载下没有出现任何 queue-full；不设的话，第一批请求进来 30 s 内就会失败（E1）。

**900 s 和 3600 s 不能直接比：** 同样的 server 配置，900 s 的测试是 vLLM 吞吐的 0.79×，3600 s 的主结果是 0.98×。这是短测试已知的边界效应（第一波冷 prompt 的权重更大），所以 900 s 的结果只能互相比较。
</details>
