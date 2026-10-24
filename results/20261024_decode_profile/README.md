# DeepInfer decode step at AgentX-like shapes (#21 W2)

The high-concurrency interactivity gap to SGLang is per-step decode time. Prefill chunk size does not matter: a c=64 sweep over 4096–32768 gave the same result.

## Same-shape steady decode, DeepInfer EP4 vs SGLang DEP4

Measured with `bench/decode_probe.py`. Synthetic AL 3.51 on both, InferenceX recipe for SGLang, 6000 tokens per request with `ignore_eos`.

| shape | DeepInfer tok/s/user | SGLang tok/s/user | SGLang / DeepInfer |
|---|---|---|---|
| 16 requests = 4 per rank, 150K context | 237.7 | 239.8 | 1.01 |
| 64 requests = 16 per rank, 64K context | 162.7 | 211.3 | 1.30 |

From 4 to 16 requests per rank, the step grows from ~14.8 to ~21.6 ms in DeepInfer, but only from ~14.6 to ~16.6 ms in SGLang.

## DeepInfer step composition

Source: `examples/benchmark.py`, `PROFILE=1`, real verification, nextn 5, rank 0. Traces are in this directory; `trace_kernels.py` in the bench queue produced the table.

| per step, ms | 4 per rank (150K) | 16 per rank (64K) | delta |
|---|---|---|---|
| forward (GPU window) | 19.90 | 24.83 | +4.93 |
| `deep_gemm::sm100_fp8_fp4_mega_moe_impl` (fused MoE incl. EP exchange) | 13.05 | 14.28 | +1.23 |
| `core_attn` (`sm100::prefill::fused_norm_rope_attn_rope_cast_fwd`) | 0.97 | 2.41 | +1.44 |
| `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl` | 2.17 | 3.08 | +0.90 |
| `deep_ep::engram::engram_fetch_impl` | 1.24 | 1.70 | +0.46 |
| `deep_gemm::sm100_mega_mhc_impl` | 1.16 | 1.61 | +0.45 |
| sampling window (incl. DSpark draft) | 1.91 | 2.73 | +0.82 |

- **The decode step is nearly independent of context length** (sparse attention). With 16 per rank it is 27.3 ms at 32K and 27.8 ms at 150K.
- **The fused MoE megakernel is two thirds of the step even at 4 requests per rank.**
- **Most of the growth with batch is attention plus the dense FP4 GEMMs:** 2.3 of the 4.9 ms.
- **In AgentX itself**, a further ~25% at c=64 goes to lockstep stalls. Whenever any DP rank prefills, all ranks' decode waits: the pure step is 15–18 ms, the effective step ~21 ms.

<details><summary>中文</summary>

# AgentX 类似形态下 DeepInfer 的 decode 单步（#21 W2）

高并发下 interactivity 落后 SGLang，差在每一步 decode 的耗时上。prefill chunk 大小没有影响：c=64 下把 chunk 从 4096 扫到 32768，结果都一样。

**同形态稳态 decode：** 每 rank 4 个请求 × 15 万上下文时，两边持平（237.7 对 239.8 tok/s/用户）；每 rank 16 个 × 6.4 万时，SGLang 快 1.30 倍（211.3 对 162.7）。batch 从 4 增加到 16，DeepInfer 每步从约 14.8 ms 涨到 21.6 ms，SGLang 只从约 14.6 ms 涨到 16.6 ms。

**DeepInfer 每步的构成**（见上表）：
- decode 单步几乎和上下文长度无关，这是稀疏注意力的效果。
- 融合 MoE megakernel 即使在每 rank 只有 4 个请求时，也占了一步的三分之二。
- batch 变大后多出来的 4.9 ms 里，有 2.3 ms 来自 attention 和 dense FP4 GEMM。
- 在 AgentX 实际负载下（c=64），还有约 25% 的时间花在同步等待上：任何一个 DP rank 在 prefill，所有 rank 的 decode 都要等。纯 decode 步是 15–18 ms，有效步长约 21 ms。
</details>
