# `--prefill-any-interval` 30 vs 100 at high concurrency: the TTFT cost (#147 follow-up)

Setup: AgentX, 3600 s per point. v3 settings: extra experts 0, JIT cache, host pools pinned before ready. The two arms ran at the same time on a fresh pair of B200 boxes: interval 30 on B200-1, interval 100 on B200-2. 30 is the new #147 default (f638894).

| | interval 30 | interval 100 | SGLang (ours, DEP4 + HiCache) |
|---|---|---|---|
| **c=128** tok/s/GPU | 176,449 | 176,391 | 172,312 |
| interactivity p50 / p90 | 145.8 / 108.4 | **173.0 / 144.9** | 148 / – |
| TTFT p50 / p90 | **0.72 / 1.64 s** | 1.51 / 4.81 s | 1.40 / 2.40 s |
| **c=256** tok/s/GPU | **318,573** | 307,932 | 300,338 |
| interactivity p50 / p90 | 92.8 / 82.1 | **119.0 / 83.6** | 81.3 / – |
| TTFT p50 / p90 | **1.39** / 6.37 s | 2.53 / 15.48 s | 1.61 / 2.89 s |

- **Interval 100 trades TTFT for interactivity.** Against SGLang, TTFT p90 is 2× worse at c=128 and 5.4× worse at c=256.
- **Interval 30 gets most of the throughput without the TTFT cost.**
  - c=128: throughput 1.02×, interactivity 0.98×, and TTFT p50 and p90 both *better* than SGLang.
  - c=256: throughput 1.06×, interactivity 1.14×, TTFT p50 better (1.39 vs 1.61 s). Only TTFT p90 is worse (6.4 vs 2.9 s).
- This matches kzjeef's GB300 sweep on #147: at c=256, interval 30 has the best throughput and ~2.3× lower TTFT p90 than 100.
- Interval 100 is still the choice when per-user speed matters most: interactivity 1.17–1.46× SGLang.

<details><summary>中文</summary>

# 高并发下 `--prefill-any-interval` 取 30 和 100 的比较：TTFT 代价（#147 后续）

测试条件：AgentX，每点 3600 s，v3 配置（冗余专家 0、JIT 缓存、server 报告 ready 前就 pin 好主机内存池）。两组同时在新开的一对 B200 上跑：间隔 30 在 B200-1，间隔 100 在 B200-2。30 是 #147 新的默认值（f638894）。

- **间隔 100 是拿 TTFT 换 interactivity：** 和 SGLang 相比，TTFT p90 在 c=128 时差 2 倍，在 c=256 时差 5.4 倍。
- **间隔 30 拿到了大部分吞吐，却没有付出 TTFT 代价：**
  - c=128：吞吐 1.02 倍，interactivity 0.98 倍，TTFT 的 p50 和 p90 都*优于* SGLang。
  - c=256：吞吐 1.06 倍，interactivity 1.14 倍，TTFT p50 更好（1.39 对 1.61 s），只有 TTFT p90 更差（6.4 对 2.9 s）。
- 这和 kzjeef 在 #147 里贴的 GB300 扫描结果一致：c=256 时间隔 30 吞吐最高，TTFT p90 比间隔 100 低约 2.3 倍。
- 如果最看重每用户速度，仍然应该选间隔 100：interactivity 是 SGLang 的 1.17–1.46 倍。
</details>
