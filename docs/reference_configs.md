# 官方参考配置对照

## SGLang（官方参考命令）

```bash
docker run --gpus all --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=<your-hf-token>" --ipc=host \
  lmsysorg/sglang:dev-dsv41 \
  sglang serve \
    --trust-remote-code \
    --model-path deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --tp 4 --ep-size 4 \
    --mem-fraction-static 0.8 \
    --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
    --cuda-graph-max-bs-decode 64 \
    --reasoning-parser auto --tool-call-parser auto \
    --host 0.0.0.0 --port 30000
```

## vLLM（官方参考命令）

```bash
docker run --gpus all --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm/vllm-openai:latest deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --tokenizer-mode deepseek_v41 \
  --tensor-parallel-size 2 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --kernel-config '{"enable_flashinfer_autotune":true}' \
  --gpu-memory-utilization 0.97 \
  --engram-config '{"cpu_offload":true,"use_thp":true}' \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092]}' \
  --max-cudagraph-capture-size 4092 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data
```

## vLLM 的投机解码（官方说明）

V4.1 去掉了 V3/V4 一起训练的 MTP 模块，**DSpark 是这个 checkpoint 唯一的投机解码方法**，5-token block；drafter 的专家大约多 14B 参数。NVIDIA 上会同时打开 adaptive verification。官方给出的命令里 `"enable_adaptive_verification":false` 是 AMD/ROCm 上的绕过写法（ROCm 上两个检查不过），在 B200 上不需要。接受率取决于负载，要在自己的流量上实测。

本仓库 `serve_vllm.sh *-spec` 用：

```
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

`VLLM_ADAPTIVE_VERIFICATION=false` 可以强制关掉 adaptive verification（验证完整 block）。三家用的都是 DSpark 5-token block：DeepInfer 的 `DEEPINFER_USE_NEXTN_DECODE` 走 DSpark/DFlash 草稿，SGLang 是 `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`。但 adaptive verification 这类验证策略各家不一定相同，结果里要记下来。

## 三家参考配置的差异

| 维度 | DeepInfer（README 默认） | SGLang 参考 | vLLM 参考 |
|---|---|---|---|
| GPU 数 | 8（EP8，每卡一个 DP rank） | 4 | 2 |
| 并行 | attention DP + MoE EP8 | TP4 + EP4 | TP2 |
| 投机解码 | DSpark，block 5（`DEEPINFER_USE_NEXTN_DECODE=5`） | DSPARK，block 5 | 参考命令无；官方说明给的是 DSpark 5-token block |
| 最大并发 | 由 `max_num_seqs` 决定 | cuda graph decode bs ≤ 64 | max-num-seqs 8 |
| prefill chunk | 32768（benchmark 默认） | 默认 | 4096 |
| KV cache 精度 | 默认（待确认） | 默认（待确认） | fp8；indexer mxfp4 |
| 显存占比 | 0.9 | 0.8 | 0.97 |
| 其他 | — | — | engram CPU offload；FlashInfer autotune |

vLLM 参考配置（2 卡、max-num-seqs 8）明显是低延迟取向，并发一旦超过 8 就会排队；SGLang 参考的 cuda graph 只覆盖到 decode bs 64，再往上会退回 eager。所以本仓库在"官方参考原样"之外，还提供了 8 卡、大并发的 `tp8` 档（见 `scripts/serve_*.sh`）。

## 本仓库各配置名

| 配置名 | 命令 | GPU |
|---|---|---|
| `sglang-ref` | `serve_sglang.sh ref` | 0-3 |
| `sglang-ref-nospec` | `serve_sglang.sh ref-nospec` | 0-3 |
| `sglang-tp8` | `serve_sglang.sh tp8` | 0-7 |
| `sglang-tp8-nospec` | `serve_sglang.sh tp8-nospec` | 0-7 |
| `vllm-ref` | `serve_vllm.sh ref` | 0-1 |
| `vllm-ref-spec` | `serve_vllm.sh ref-spec` | 0-1 |
| `vllm-tp8` | `serve_vllm.sh tp8` | 0-7 |
| `vllm-tp8-spec` | `serve_vllm.sh tp8-spec` | 0-7 |
| `deepinfer-ep8-nospec` | `sweep_deepinfer.sh deepinfer-ep8-nospec 8 0 <场景>` | 0-7 |
| `deepinfer-ep8-dspark5` | `sweep_deepinfer.sh deepinfer-ep8-dspark5 8 5 <场景>` | 0-7 |
| `deepinfer-ep4-nospec` / `-dspark5` | `sweep_deepinfer.sh ... 4 ...` | 0-3 |

本仓库脚本与官方命令只有三处差别：模型从本地目录挂进容器（不走 HF 下载）；加 `--served-model-name` 让三家报同一个模型名；SGLang 加 `--enable-metrics` 以便抓投机解码接受率。
