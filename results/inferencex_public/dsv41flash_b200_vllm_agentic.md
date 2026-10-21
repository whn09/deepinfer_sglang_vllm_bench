# InferenceX published AgentX points: DeepSeek-V4.1-Flash, B200, vLLM

Source: `https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash` (retrieved 2026-10-21T13:59:45Z), filter hardware=b200, framework=vllm, benchmark_type=agentic_traces; 12 points, run https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37070984585/attempts/1, image `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7`.

Throughput per GPU = (input + output tokens) / duration / GPUs (`infx/results/agentic/request_metrics.py`). Interactivity = 1 / ITL.

| topology | GPUs | c | total tok/s/GPU | output tok/s/GPU | intvty p50 (tok/s/user) | TTFT p50 (s) | E2EL p90 (s) | date |
|---|---|---|---|---|---|---|---|---|
| TP2 EP2 DP-attn | 2 | 8 | 31045 | 222.8 | 358.4 | 0.117 | 8.70 | 2026-10-03 |
| TP2 EP2 DP-attn | 2 | 16 | 56674 | 358.1 | 305.8 | 0.127 | 8.36 | 2026-10-03 |
| TP2 EP2 DP-attn | 2 | 32 | 117908 | 683.4 | 232.0 | 0.157 | 12.08 | 2026-10-03 |
| TP4 | 4 | 1 | 7699 | 46.0 | 606.1 | 0.181 | 8.56 | 2026-10-03 |
| TP4 | 4 | 4 | 10293 | 70.3 | 595.2 | 0.105 | 6.85 | 2026-10-03 |
| TP4 | 4 | 8 | 16549 | 115.1 | 520.8 | 0.086 | 6.15 | 2026-10-03 |
| TP4 | 4 | 16 | 29413 | 190.6 | 418.4 | 0.093 | 5.97 | 2026-10-03 |
| TP4 | 4 | 32 | 62356 | 367.6 | 314.5 | 0.114 | 8.76 | 2026-10-03 |
| TP4 | 4 | 64 | 116808 | 812.6 | 190.1 | 0.117 | 13.05 | 2026-10-03 |
| TP4 | 4 | 128 | 166193 | 1315.7 | 116.4 | 0.158 | 22.37 | 2026-10-03 |
| TP4 EP4 DP-attn | 4 | 64 | 117169 | 818.4 | 199.6 | 0.134 | 12.62 | 2026-10-03 |
| TP4 EP4 DP-attn | 4 | 128 | 173194 | 1362.2 | 131.6 | 0.163 | 19.93 | 2026-10-03 |
