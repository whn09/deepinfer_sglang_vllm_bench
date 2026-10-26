# Cross-node DeepInfer + Mooncake correctness (2×B200, ap-south-2b, cluster placement group, 8 EFA each)

All runs use real verification (no synthetic acceptance), so greedy outputs are deterministic and comparable.

## PD split across machines (#214 rebased on the latest #197)
PE on B200-1 GPUs 0-3, CE on B200-2 GPUs 0-3. `bench/pd_parity.py` uses 12 prompts of 1K–64K tokens and takes an md5 of each greedy completion.

| comparison | result |
|---|---|
| cross-node, Mooncake EFA vs cross-node, TCP | identical 12/12 |
| cross-node, TCP vs single-node PD (TCP, one box) | identical 12/12 |
| cross-node, Mooncake EFA vs single-node PD | identical 12/12 |

- All 4 PE and 4 CE ranks installed the EFA transport, with no fallback or retry lines in the logs.
- Time to the first answer of a 64K prompt: 2.07 s over EFA vs 2.26 s over TCP.

## L3 across machines (#216)
The Store (master + 100 GB segment, EFA `rdmap113s0`) runs on B200-1.
1. Engine A on B200-1 populates turn 1. Engine B on B200-2 then sends turn 2 with an empty host pool, so a hit can only come from the remote Store.
2. Engine C on B200-2 has no L3 and runs with `--max-num-batched-tokens` = B's resume point.

| | result |
|---|---|
| engine A puts | 219 objects, 0 failed / retried / dropped |
| engine B L3 hit | 89,600 tokens (700 pages + the SWA state) in 13 ms, 0 errors |
| engine B TTFT | 129 ms cached vs 1,426 ms uncached (11×) |
| B cached == C (no cache, chunk at b) | **True** (bit-exact) |
| B cached == B uncached (default chunk) | False, expected: the chunk-split rule from #144 |

## Findings along the way
- **Mooncake `setup()` reads `rdma_devices`, not `device_name`.** With the wrong key, the client silently auto-discovers all 8 EFA devices. Fixed in `serve_deepinfer_agentx.sh` and `bench/store_stress_mp.py`.
- **Mooncake client regression between dd1217b and 83f8f877.** With a 4-process EFA stress run (20 s, batch 16), dd1217b gives 0 failures; 83f8f877 gives ~30 `TRANSFER_FAIL` (-800) per ~180K puts (~0.015%), plus a few get failures. On AgentX this tripped #216's breaker on the first put of every rank.
  - **Bisected to kvcache-ai/Mooncake 487da6af** ("[TE] fix(efa): evict stale peer on reconnect + clearer fi_av_insert logs", #2063). Its parent 12a7e9f1 has 0 failures; 487da6af has 57 put + 49 get failures in 30 s.
  - **Cause:** `EfaContext::submitSlicesOnPeer` fails the slices after `kMaxStalledWaves = 1024` consecutive `FI_EAGAIN` waves, and the limit is hard-coded and counted in waves, not time. The failing log line reads `1024 consecutive FI_EAGAIN waves posted nothing ...; wr_depth=1/2048; cq_outstanding=1; last_completion=0-1ms ago; this_peer_inflight=1`. The peer is healthy (a completion arrived <1 ms ago, one op in flight); the provider is only briefly busy, and 1024 waves elapse in microseconds.
  - `MC_EFA_CQ_DRAIN_TIMEOUT_MS` is not it: 10000 still fails, and 0 makes it far worse.
  - **Suggested fix:** also require no completion progress for some wall time (e.g. `msSinceLastCompletion()` > a few ms) before declaring the peer stalled, or make the limit configurable.
  - **Fix: kvcache-ai/Mooncake#4956.** It adds a wall-clock floor (`MC_EFA_STALL_MIN_MS`). The default was raised from 50 ms to **1000 ms** after cross-host validation ([docs/upstream/mooncake_efa_v0314_release_validation.md](../../docs/upstream/mooncake_efa_v0314_release_validation.md)): at 50 ms, a cross-host 1-GPU read still failed 4/4 runs. Dead-peer cost at 1000 ms is within noise.
    - Healthy peer: 0 failures and 4.35 GB/s, vs 37 failures and 3.71 GB/s with the old check.
    - Dead peer: no slower to fail (121 vs 120 s mean over 3 runs).
  - Workaround until it lands: build Mooncake at ≤ 12a7e9f1 (we use dd1217b). #216's one-retry also absorbs it.
- **#216 now retries failed objects once before tripping the breaker** (625606d), with new counters `l3_put_retried` / `l3_get_retried`. The L3 runs above use Mooncake dd1217b plus this change.

<details><summary>中文</summary>

# DeepInfer + Mooncake 跨机正确性（2×B200，ap-south-2b，cluster 置放组，每台 8 个 EFA）

所有测试都用真实验证（不用合成接受长度），所以 greedy 输出是确定的，可以直接对比。

**跨机 PD 分离（#214，rebase 到最新的 #197）：**
- PE 在 B200-1 的 GPU 0-3，CE 在 B200-2 的 GPU 0-3。
- 跨机 Mooncake EFA、跨机 TCP、单机 PD 三者的 greedy 输出两两一致，都是 12/12。
- 两边 4 个 rank 都装上了 EFA 传输，没有回退或重试。
- 64K prompt 的首个响应：EFA 2.07 s，TCP 2.26 s。

**跨机 L3（#216）：**
- Store 在 B200-1。引擎 A 在 B200-1 写入第一轮；引擎 B 在 B200-2 上、主机缓存池为空，从远端 Store 命中 89,600 个 token（700 页加 SWA state），用时 13 ms，零错误；TTFT 是 129 ms，不走缓存要 1,426 ms（快 11 倍）。
- B 命中缓存的输出与引擎 C（不开 L3、在恢复点切分）的输出**逐字节相同**。

**过程中的发现：**
- Mooncake `setup()` 读的是 `rdma_devices`，不是 `device_name`。键名写错时客户端会静默地自动发现全部 8 块 EFA 网卡，已修正。
- **Mooncake 客户端在 dd1217b 和 83f8f877 之间出现了回归：** 4 进程 EFA 压测下，dd1217b 零失败，83f8f877 约 0.015% 的 put 返回 `TRANSFER_FAIL`（-800），还有少量 get 失败。在 AgentX 上，这会让 #216 的熔断在每个 rank 的第一批 put 时就被触发。二分定位到 kvcache-ai/Mooncake 的 487da6af（#2063，"evict stale peer on reconnect"）：`EfaContext::submitSlicesOnPeer` 在连续 `kMaxStalledWaves = 1024` 轮 `FI_EAGAIN` 后就判失败，这个上限是写死的、按轮数计，不按时间。日志显示当时完成事件 0–1 ms 前刚回来过、只有 1 个请求在途，peer 是健康的，provider 只是短暂繁忙。`MC_EFA_CQ_DRAIN_TIMEOUT_MS` 不是原因。建议的修法：判定卡死前同时要求一段时间内没有任何完成事件，或者把这个上限做成可配置。修好之前的绕法：用 ≤ 12a7e9f1 的版本（我们用 dd1217b）；#216 的重试一次也能吸收这类失败。
- #216 现在会先把失败的对象重试一次，重试仍失败才触发熔断（625606d）。上面的 L3 测试用的是 Mooncake dd1217b 加上这个改动。
</details>
