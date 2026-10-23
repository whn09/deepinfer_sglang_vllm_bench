# Mooncake Store L3 under load: 0.3.13 vs Mooncake main (#20)

**Root cause of the E1 failure.** The Mooncake 0.3.13 TCP transport queues at most 1024 transfers per peer. A 167K-token AgentX request stores about 1300 pages, so one `batch_put_from_multi_buffers` call carries about 1300 buffers. The transfers over the limit are rejected (`TCP lane queue-full rejection`). The batch does not fail fast; it waits for the 60 s timeout and returns `TRANSFER_FAIL (-800)`, and the backlog then cascades. Mooncake main raises the default to 65535 (kvcache-ai/Mooncake#3710, 2026-09-01, one day after 0.3.13.post1).

**Reproduction without GPUs** (`bench/store_stress_mp.py`: 4 processes, one Store client each, a standalone `mooncake_client` holding the segment, cross-process gets, 60 s):

| Mooncake | objects per call | put fail | get fail | put GB/s | get GB/s |
|---|---|---|---|---|---|
| 0.3.13.post1 wheel | 16 | 0 | 0 | 0.99 | 0.99 |
| 0.3.13.post1 wheel | 96 | 0 | 0 | 1.10 | 1.06 |
| 0.3.13.post1 wheel | 320 | 1536 | 1024 (all) | 0.04 | 0.00 |
| 0.3.13.post1 + `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535` | 320 | 0 | 0 | 1.00 | 0.99 |
| main @ dd1217b (`docker/Dockerfile.mooncake-main`) | 320 | 0 | 0 | 0.99 | 0.97 |
| main @ dd1217b | 16 | 0 | 0 | 1.10 | 1.07 |

**AgentX, paired** (c=64, 900 s, least-loaded routing so conversations move between DP ranks, image `deepinfer:agentx-l3-mcmain`, L3 on GPUs 0-3 and L3 off on GPUs 4-7 at the same time):

| | L3 off | L3 on (main) | on / off |
|---|---|---|---|
| total tok/s/GPU | 67,468 | 91,917 | 1.36× |
| interactivity p50 | 93.0 | 179.5 | 1.93× |
| interactivity mean | 61.0 | 152.4 | 2.50× |
| TTFT p50 / p90 s | 0.53 / 2.04 | 0.15 / 0.56 | |
| prefix-cache token hit rate | 73.7% | 92.2% | |
| L3 hits / pages / states loaded | – | 1757 / 461,862 / 1756 | |
| L3 put objects / GiB, failures | – | 194,921 / 94, 0 | |
| L3 get time total | – | 25 s (14 ms per hit) | |
| queue-full / TRANSFER_FAIL / 60 s timeouts in the log | 0 / 0 / 0 | 0 / 0 / 0 | |

The same arm on 0.3.13 (E1, 2026-10-23 12:15) had 0 L3 hits and 63,110 tok/s/GPU.

**Caveats.**
- These runs use least-loaded routing, which is where L3 matters. With sticky routing and no L3, the headline config (`deepinfer-ix-ep4-noeplb-fastok`, 3600 s, fastokens) reaches 0.98× vLLM throughput and 0.92× interactivity at c=64. These 900 s runs don't use fastokens and are shorter, so they are not directly comparable with it.
- Each request's L3 get still runs synchronously on the engine thread. Failure handling (fail fast, circuit breaker) is still open.

<details><summary>中文</summary>

# 负载下的 Mooncake Store L3：0.3.13 vs Mooncake main（#20）

**E1 失败的根因：** Mooncake 0.3.13 的 TCP 传输对每个 peer 最多排队 1024 个传输。AgentX 一个 16.7 万 token 的请求要存约 1300 页，所以一次 `batch_put_from_multi_buffers` 调用就带约 1300 个 buffer。超出上限的传输被拒（`TCP lane queue-full rejection`）。整批任务不会立刻失败，而是等到 60 s 超时才返回 `TRANSFER_FAIL (-800)`，积压随后级联扩散。Mooncake main 把默认值提到 65535（kvcache-ai/Mooncake#3710，2026-09-01，就在 0.3.13.post1 发布后一天）。

**不用 GPU 也能复现**（`bench/store_stress_mp.py`：4 个进程各带一个 Store 客户端，segment 由独立的 `mooncake_client` 提供，进程之间互相读对方写入的数据，跑 60 s）：只要每次调用的对象数到 320，0.3.13 就会出现 queue-full，put 掉到 0.04 GB/s，get 全部失败。给 0.3.13 加上两个环境变量（`MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535`、`MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535`），或者换成 main，都是零失败、约 1 GB/s。

**AgentX 配对结果**（c=64，900 s，least-loaded 路由让同一对话在不同 DP rank 之间迁移；L3 on 跑在 GPU 0-3、L3 off 跑在 GPU 4-7，两组同时跑）：L3 让吞吐提升到 1.36×，interactivity p50 提升到 1.93×；TTFT p50 从 0.53 s 降到 0.15 s；token 命中率从 73.7% 升到 92.2%；L3 命中 1757 次、put 19.5 万个对象，零失败。同样配置在 0.3.13 上（E1）L3 命中为 0。

**注意：** 这组用的是 least-loaded 路由，正是 L3 最有用的场景。sticky 路由、不开 L3 的主结果（3600 s + fastokens）在 c=64 时已经达到 vLLM 吞吐的 0.98×，和这组 900 s、没开 fastokens 的结果不能直接比较。L3 get 目前仍在引擎线程上同步执行，失败处理（快速失败、熔断）还没做。
</details>
