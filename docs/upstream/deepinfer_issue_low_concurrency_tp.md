**Low-concurrency decode is bounded by the EP-only MoE, which leaves a ~7 ms floor per step. Is a TP path for small batches planned?**

Setup: 8×B200; DeepSeek-V4.1-Flash-1017; the #147 server with #144; `--dp 4` (EP4, DP attention); MTP 5; InferenceX AgentX traces with the golden acceptance length 3.51.

At high concurrency, DeepInfer now matches or beats vLLM and SGLang on AgentX. At c=128 / c=256 throughput is 1.02×, and interactivity is 1.15× / 1.45× the stronger of the two. Two serving settings made the difference: 0 redundant experts with EPLB off, and `--prefill-any-interval 100`. At c ≤ 16, though, interactivity is ~0.6× of vLLM / SGLang, which serve those points with TP4:

| c | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| DeepInfer interactivity p50 (tok/s/user) | 391 | 382 | 356 | 327 |
| best of vLLM TP4 / SGLang TP4 | 665 | 639 | 599 | 505 |

Where a decode step goes, from `examples/benchmark.py`, EP4, rank 0:

| per-rank batch | 0 (idle rank) | 1 | 4 | 16 |
|---|---|---|---|---|
| MLP (`sm100_fp8_fp4_mega_moe`) | ~7.4 ms | 11.3 | 13.6 | 15.3 |
| attention block | – | 5.1 | 5.5 | 9.0 |
| whole step | – | ~15.6 | ~18.5 | ~24 |

- The mega MoE scheduler only builds M blocks for experts that received tokens. Even so, there is a fixed ~7 ms per step, about 175 µs per layer for the in-kernel dispatch / combine and the barriers. Because DP ranks step in lockstep, an idle rank pays it too.
- TP4 avoids the all-to-all and splits attention, so a whole step takes ~5.6 ms.
- What we tried:
  - The non-mega path (DeepEP dispatch + `m_grouped_fp8_fp4_gemm_nt_contiguous` + combine, forced with a local switch) is 2–2.4× slower than mega MoE.
  - CP keeps EP MoE.
  - EP2 doesn't fit: 172 GB per GPU before the KV cache.

Questions:
1. Is a tensor-parallel decode path planned, i.e. TP attention plus TP or replicated-token MoE for small batches? Or a lower-latency mode of the mega MoE kernel, e.g. fewer SMs or a lighter barrier?
2. Is the ~175 µs per layer at 1–6 tokens per rank expected for the mega kernel, or is there a configuration we are missing?

Full data: https://github.com/whn09/deepinfer_sglang_vllm_bench/blob/main/results/20261024_agentx_v3/summary.md and the decode profiles in `results/20261024_decode_profile/`.

<details><summary>中文</summary>

**低并发 decode 受限于只有 EP 方式的 MoE，每步有约 7 ms 的下限。请问有计划为小 batch 提供 TP 路径吗？**

测试环境：8×B200，DeepSeek-V4.1-Flash-1017，#147 的 server 加 #144，`--dp 4`（EP4，DP attention），MTP 5，InferenceX AgentX 的 trace，使用 golden 接受长度 3.51。

高并发下，DeepInfer 在 AgentX 上已经达到或超过 vLLM 和 SGLang：c=128 / c=256 时吞吐是两者中更强那一方的 1.02 倍，interactivity 分别是 1.15 倍和 1.45 倍。起作用的是两个 serving 配置：EPLB 关闭时把冗余专家设为 0，以及 `--prefill-any-interval 100`。但在 c ≤ 16 时，interactivity 只有 vLLM / SGLang 的约 0.6 倍，而它们在这些点用的是 TP4（数据见上表）。

- mega MoE 的调度器只为分到 token 的专家生成 M 块。即便如此，每步仍有约 7 ms 的固定开销，也就是每层约 175 µs，花在 kernel 内的 dispatch / combine 和同步上。由于 DP rank 是同步走步的，空闲的 rank 也要付这笔开销。
- TP4 没有 all-to-all，而且把 attention 也切开了，所以整个一步只要约 5.6 ms。
- 我们试过的办法：
  - 非 mega 路径（DeepEP dispatch + `m_grouped_fp8_fp4_gemm_nt_contiguous` + combine，用一个本地开关强制启用）比 mega MoE 慢 2–2.4 倍。
  - CP 下的 MoE 仍然是 EP。
  - EP2 装不下：分配 KV cache 之前每张卡就已经用了 172 GB。

问题：
1. 是否计划提供张量并行的 decode 路径，即小 batch 时使用 TP attention 加上 TP（或复制 token）方式的 MoE？或者为 mega MoE kernel 提供一个低延迟模式，比如用更少的 SM 或更轻量的同步？
2. 每 rank 只有 1–6 个 token 时，每层约 175 µs 对 mega kernel 来说是预期的吗？还是我们漏掉了某个配置？
</details>
