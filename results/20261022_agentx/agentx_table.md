| system | conc | topo | tok/s/GPU | out tok/s/GPU | intvty p50 | TTFT p50 s | TTFT p90 s | E2EL p50 s |
|---|---|---|---|---|---|---|---|---|
| DeepInfer DEP4 (EPLB off) | 1 | dep4 | 5,519 | 37.81 | 380 | 0.77 | 1.24 | 3.31 |
| DeepInfer DEP4 (EPLB off) | 4 | dep4 | 8,701 | 58.08 | 368 | 0.36 | 1.27 | 1.83 |
| DeepInfer DEP4 (EPLB off) | 8 | dep4 | 14,977 | 108 | 342 | 0.28 | 0.84 | 1.75 |
| DeepInfer DEP4 (EPLB off) | 16 | dep4 | 28,005 | 177 | 314 | 0.32 | 0.76 | 1.64 |
| DeepInfer DEP4 (EPLB off) | 32 | dep4 | 59,623 | 345 | 267 | 0.39 | 0.93 | 2.25 |
| DeepInfer DEP4 (EPLB off) | 64 | dep4 | 112,157 | 785 | 175 | 0.34 | 1.10 | 3.34 |
| DeepInfer DEP4 (EPLB off) | 128 | dep4 | 161,557 | 1,292 | 112 | 0.44 | 1.19 | 5.17 |
| DeepInfer DEP4 (EPLB on, #147 default) | 1 | dep4 | 3,476 | 28.02 | 173 | 0.70 | 1.61 | 5.43 |
| DeepInfer DEP4 (EPLB on, #147 default) | 4 | dep4 | 5,931 | 44.27 | 160 | 0.39 | 1.24 | 2.98 |
| DeepInfer DEP4 (EPLB on, #147 default) | 8 | dep4 | 11,395 | 86.98 | 140 | 0.43 | 0.94 | 3.70 |
| DeepInfer DEP4 (EPLB on, #147 default) | 16 | dep4 | 21,418 | 151 | 115 | 0.45 | 1.09 | 3.64 |
| DeepInfer DEP4 (EPLB on, #147 default) | 32 | dep4 | 42,837 | 271 | 89.81 | 0.58 | 1.22 | 5.49 |
| DeepInfer DEP4 (EPLB on, #147 default) | 64 | dep4 | 62,740 | 494 | 47.90 | 0.61 | 1.44 | 10.26 |
| DeepInfer DEP4 (EPLB on, #147 default) | 128 | dep4 | 79,675 | 704 | 24.56 | 0.79 | 1.67 | 18.31 |
| SGLang (ours) | 1 | tp4 | 7,801 | 46.81 | 665 | 0.25 | 0.38 | 1.95 |
| SGLang (ours) | 4 | tp4 | 9,891 | 68.29 | 639 | 0.20 | 1.17 | 1.25 |
| SGLang (ours) | 8 | tp4 | 16,271 | 114 | 599 | 0.23 | 1.22 | 1.40 |
| SGLang (ours) | 16 | tp4 | 28,157 | 178 | 505 | 0.65 | 2.00 | 1.66 |
| SGLang (ours) | 24 | tp4 | 43,208 | 261 | 417 | 0.62 | 1.60 | 1.67 |
| SGLang (ours) | 48 | dep4 | 84,217 | 554 | 291 | 1.56 | 2.77 | 3.37 |
| vLLM TP4 (ours) | 1 | tp4 | 7,796 | 46.78 | 629 | 0.19 | 0.29 | 1.78 |
| vLLM TP4 (ours) | 4 | tp4 | 9,810 | 68.03 | 621 | 0.10 | 0.28 | 0.96 |
| vLLM TP4 (ours) | 8 | tp4 | 16,834 | 116 | 551 | 0.09 | 0.25 | 1.02 |
| vLLM TP4 (ours) | 16 | tp4 | 29,438 | 191 | 445 | 0.10 | 0.26 | 0.97 |
| vLLM TP4 (ours) | 32 | tp4 | 62,991 | 370 | 337 | 0.12 | 0.29 | 1.45 |
| vLLM TP4 (ours) | 64 | tp4 | 116,808 | 813 | 187 | 0.12 | 0.34 | 2.70 |
| vLLM TP4 (ours) | 128 | tp4 | 167,606 | 1,329 | 118 | 0.15 | 0.61 | 4.40 |
| vLLM (InferenceX) | 8 | dep2 | 31,045 | 223 | 358 | 0.12 | 0.29 | 1.54 |
| vLLM (InferenceX) | 16 | dep2 | 56,674 | 358 | 306 | 0.13 | 0.36 | 1.44 |
| vLLM (InferenceX) | 32 | dep2 | 117,908 | 683 | 232 | 0.16 | 0.41 | 2.29 |
| vLLM (InferenceX) | 64 | dep4 | 117,169 | 818 | 200 | 0.13 | 0.36 | 2.67 |
| vLLM (InferenceX) | 128 | dep4 | 173,194 | 1,362 | 132 | 0.16 | 0.44 | 4.08 |
| vLLM (InferenceX) | 1 | tp4 | 7,699 | 46.03 | 606 | 0.18 | 0.28 | 1.84 |
| vLLM (InferenceX) | 4 | tp4 | 10,293 | 70.32 | 595 | 0.10 | 0.27 | 1.02 |
| vLLM (InferenceX) | 8 | tp4 | 16,549 | 115 | 521 | 0.09 | 0.24 | 1.05 |
| vLLM (InferenceX) | 16 | tp4 | 29,413 | 191 | 418 | 0.09 | 0.26 | 1.01 |
| vLLM (InferenceX) | 32 | tp4 | 62,356 | 368 | 314 | 0.11 | 0.29 | 1.55 |
| vLLM (InferenceX) | 64 | tp4 | 116,808 | 813 | 190 | 0.12 | 0.34 | 2.72 |
| vLLM (InferenceX) | 128 | tp4 | 166,193 | 1,316 | 116 | 0.16 | 0.78 | 4.58 |
| SGLang (InferenceX) | 48 | dep4 | 84,650 | 555 | 287 | 1.50 | 2.57 | 3.30 |
| SGLang (InferenceX) | 64 | dep4 | 115,369 | 804 | 238 | 1.46 | 2.50 | 3.71 |
| SGLang (InferenceX) | 96 | dep4 | 147,278 | 1,107 | 182 | 1.37 | 2.33 | 4.19 |
| SGLang (InferenceX) | 128 | dep4 | 173,311 | 1,359 | 150 | 1.34 | 2.25 | 4.70 |
| SGLang (InferenceX) | 192 | dep4 | 254,011 | 1,823 | 118 | 1.49 | 2.58 | 5.31 |
| SGLang (InferenceX) | 256 | dep4 | 305,126 | 2,166 | 83.54 | 1.54 | 2.69 | 6.80 |
| SGLang (InferenceX) | 1 | tp4 | 7,858 | 47.19 | 671 | 0.18 | 0.39 | 1.81 |
| SGLang (InferenceX) | 4 | tp4 | 9,319 | 64.61 | 641 | 0.14 | 1.32 | 1.18 |
| SGLang (InferenceX) | 8 | tp4 | 16,454 | 115 | 592 | 0.18 | 1.23 | 1.38 |
| SGLang (InferenceX) | 16 | tp4 | 27,738 | 175 | 503 | 0.62 | 1.84 | 1.63 |
| SGLang (InferenceX) | 24 | tp4 | 43,306 | 262 | 426 | 0.58 | 1.65 | 1.59 |
