# InferenceX published AgentX points: DeepSeek-V4.1-Flash, B200, SGLang

Source: `https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash` (retrieved 2026-10-21T13:59:45Z), 11 points, image `lmsysorg/sglang:nightly-dev-cu13-20261014-684fea8a@sha256:ddae0e399d3d94f7827f16183301cf8dac4cf0eae0ef25590b7dac5881d713d8`, run https://github.com/SemiAnalysisAI/InferenceX/actions/runs/38749770474/attempts/1.

| topology | GPUs | c | KV offload | total tok/s/GPU | output tok/s/GPU | intvty p50 | TTFT p50 (s) | E2EL p90 (s) | date |
|---|---|---|---|---|---|---|---|---|---|
| TP4 | 4 | 1 | off | 7858 | 47.2 | 671.1 | 0.177 | 8.00 | 2026-10-16 |
| TP4 | 4 | 4 | off | 9319 | 64.6 | 641.0 | 0.144 | 7.28 | 2026-10-16 |
| TP4 | 4 | 8 | off | 16454 | 114.8 | 591.7 | 0.179 | 6.48 | 2026-10-16 |
| TP4 EP4 | 4 | 16 | off | 27738 | 174.5 | 502.5 | 0.624 | 8.91 | 2026-10-16 |
| TP4 EP4 | 4 | 24 | off | 43306 | 261.9 | 425.5 | 0.575 | 10.08 | 2026-10-16 |
| TP4 EP4 DP-attn | 4 | 48 | on | 84650 | 555.3 | 286.5 | 1.499 | 12.08 | 2026-10-16 |
| TP4 EP4 DP-attn | 4 | 64 | on | 115369 | 803.6 | 237.5 | 1.457 | 13.32 | 2026-10-16 |
| TP4 EP4 DP-attn | 4 | 96 | on | 147278 | 1107.0 | 182.5 | 1.373 | 16.50 | 2026-10-16 |
| TP4 EP4 DP-attn | 4 | 128 | on | 173311 | 1359.0 | 150.2 | 1.343 | 19.37 | 2026-10-16 |
| TP4 EP4 DP-attn | 4 | 192 | on | 254011 | 1823.3 | 118.1 | 1.487 | 24.91 | 2026-10-16 |
| TP4 EP4 DP-attn | 4 | 256 | on | 305126 | 2165.8 | 83.5 | 1.535 | 33.05 | 2026-10-16 |
