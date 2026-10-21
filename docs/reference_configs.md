# Official reference configs

## SGLang (official reference command)

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

## vLLM (official reference command)

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

## vLLM speculative decoding (official notes)

V4.1 dropped the MTP module that V3 and V4 trained alongside the backbone; **DSpark is the only speculative method for this checkpoint**, with a 5-token block, and the drafter's experts add roughly 14B parameters. On NVIDIA it also turns on adaptive verification. The `"enable_adaptive_verification":false` in the published command is the AMD/ROCm workaround (two checks fail on ROCm) and is not needed on B200. Acceptance depends on the workload, so measure it on your own traffic.

`serve_vllm.sh *-spec` uses:

```
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

`VLLM_ADAPTIVE_VERIFICATION=false` forces adaptive verification off (verify the full block). All three stacks use DSpark with a 5-token block: DeepInfer's `DEEPINFER_USE_NEXTN_DECODE` drives DSpark/DFlash drafting, SGLang uses `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`. Verification policies such as adaptive verification may still differ; record them with the results.

## Images

Use the latest official release, pinned by version (not `latest`); fall back to a dedicated image only if the release cannot serve the model.

| Stack | Default in this repo | Fallback | Override |
|---|---|---|---|
| SGLang | `lmsysorg/sglang:v0.5.22-cu130` | `lmsysorg/sglang:dev-dsv41` (the official reference command's image) | `SGLANG_IMAGE=...` |
| vLLM | `vllm/vllm-openai:v0.32.0` | — | `VLLM_IMAGE=...` |

## How the three reference configs differ

| Dimension | DeepInfer (README default) | SGLang reference | vLLM reference |
|---|---|---|---|
| GPUs | 8 (EP8, one DP rank per GPU) | 4 | 2 |
| Parallelism | attention DP + MoE EP8 | TP4 + EP4 | TP2 |
| Speculation | DSpark, block 5 (`DEEPINFER_USE_NEXTN_DECODE=5`) | DSPARK, block 5 | none in the command; the official notes give DSpark with a 5-token block |
| Max concurrency | set by `max_num_seqs` | CUDA graph decode bs ≤ 64 | max-num-seqs 8 |
| Prefill chunk | 32768 (benchmark default) | default | 4096 |
| KV cache dtype | default (to be confirmed) | default (to be confirmed) | fp8; indexer mxfp4 |
| Memory fraction | 0.9 | 0.8 | 0.97 |
| Other | — | — | engram CPU offload; FlashInfer autotune |

The vLLM reference (2 GPUs, max-num-seqs 8) is clearly latency-oriented: beyond concurrency 8 requests queue. The SGLang reference captures CUDA graphs only up to decode bs 64 and falls back to eager above that. That is why this repo adds high-concurrency `tp4` profiles (4 GPUs, the main comparison, matching InferenceX) and `tp8` profiles (8 GPUs, supplement) next to the verbatim references (`scripts/serve_*.sh`).

## Config names in this repo

Server profiles take a suffix: SGLang `-nospec` turns DSpark off, vLLM `-spec` turns it on. Per-concurrency topologies follow InferenceX's B200 recipes (#14).

| Config | Command | Topology | GPUs |
|---|---|---|---|
| `sglang-ref` | `serve_sglang.sh ref` | TP4/EP4, CUDA graph bs ≤ 64 (official reference) | 0-3 |
| `sglang-tp4` | `serve_sglang.sh tp4` | TP4/EP4 | 0-3 |
| `sglang-tp4-ep1` | `serve_sglang.sh tp4-ep1` | TP4, no EP | 0-3 |
| `sglang-dpa4` | `serve_sglang.sh dpa4` | TP4 + DP4 + EP4, DP attention, MegaMoE | 0-3 |
| `sglang-dpa4-deepep` | `serve_sglang.sh dpa4-deepep` | as `dpa4`, DeepEP a2a (Tier C) | 0-3 |
| `sglang-tp8` | `serve_sglang.sh tp8` | TP8/EP8 | 0-7 |
| `vllm-ref` | `serve_vllm.sh ref` | TP2, max-num-seqs 8 (official reference) | 0-1 |
| `vllm-tp4` | `serve_vllm.sh tp4` | TP4 | 0-3 |
| `vllm-dep4` | `serve_vllm.sh dep4` | TP1 × DP4 + EP, MegaMoE | 0-3 |
| `vllm-dep4-deepep` | `serve_vllm.sh dep4-deepep` | as `dep4`, DeepEP a2a (Tier C) | 0-3 |
| `vllm-tp8` | `serve_vllm.sh tp8` | TP8 | 0-7 |
| `deepinfer-ep4-{nospec,dspark5}` | `sweep_deepinfer.sh ... 4 ...` | attention DP + MoE EP4 (DeepEP) | 0-3 |
| `deepinfer-ep8-{nospec,dspark5}` | `sweep_deepinfer.sh ... 8 ...` | attention DP + MoE EP8 | 0-7 |

`GPUS_OVERRIDE=4,5,6,7` (servers) / `--device-offset 4` (DeepInfer) move a 4-GPU config to the other half of the box.

The scripts differ from the official commands in four ways only: the image is the pinned latest release, the model is mounted from a local directory (no HF download), `--served-model-name` gives all three the same model name, and SGLang adds `--enable-metrics` so speculative counters can be scraped.

<details><summary>中文</summary>

# 官方参考配置

## SGLang / vLLM 官方参考命令

见上方英文部分的两段命令。

## vLLM 的投机解码（官方说明）

V4.1 去掉了 V3/V4 和主干一起训练的 MTP 模块，**DSpark 是这个 checkpoint 唯一的投机解码方法**，5-token block；drafter 的专家大约多 14B 参数。NVIDIA 上会同时打开 adaptive verification。官方命令里的 `"enable_adaptive_verification":false` 是 AMD/ROCm 上的绕过写法（ROCm 上有两个检查不过），B200 上不需要。接受率取决于负载，要在自己的流量上实测。

`serve_vllm.sh *-spec` 用的参数见上方英文部分。`VLLM_ADAPTIVE_VERIFICATION=false` 可以强制关掉 adaptive verification（验证完整 block）。三家用的都是 DSpark 5-token block：DeepInfer 的 `DEEPINFER_USE_NEXTN_DECODE` 走 DSpark/DFlash 草稿，SGLang 是 `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`。但 adaptive verification 这类验证策略各家不一定相同，结果里要记下来。

## 镜像

优先用最新正式发行版，并钉版本号（不用 `latest`）；正式版跑不了这个模型时，才退到专用镜像。

| 框架 | 本仓库默认 | 备选 | 覆盖方式 |
|---|---|---|---|
| SGLang | `lmsysorg/sglang:v0.5.22-cu130` | `lmsysorg/sglang:dev-dsv41`（官方参考命令用的镜像） | `SGLANG_IMAGE=...` |
| vLLM | `vllm/vllm-openai:v0.32.0` | — | `VLLM_IMAGE=...` |

## 三家参考配置的差异

| 维度 | DeepInfer（README 默认） | SGLang 参考 | vLLM 参考 |
|---|---|---|---|
| GPU 数 | 8（EP8，每卡一个 DP rank） | 4 | 2 |
| 并行 | attention DP + MoE EP8 | TP4 + EP4 | TP2 |
| 投机解码 | DSpark，block 5（`DEEPINFER_USE_NEXTN_DECODE=5`） | DSPARK，block 5 | 命令里没有；官方说明给的是 DSpark 5-token block |
| 最大并发 | 由 `max_num_seqs` 决定 | cuda graph decode bs ≤ 64 | max-num-seqs 8 |
| prefill chunk | 32768（benchmark 默认） | 默认 | 4096 |
| KV cache 精度 | 默认（待确认） | 默认（待确认） | fp8；indexer mxfp4 |
| 显存占比 | 0.9 | 0.8 | 0.97 |
| 其他 | — | — | engram CPU 卸载；FlashInfer autotune |

vLLM 参考配置（2 卡、max-num-seqs 8）明显是低延迟取向，并发超过 8 就会排队；SGLang 参考配置的 cuda graph 只覆盖到 decode bs 64，再往上会退回 eager。所以本仓库在原样照抄的参考配置之外，还提供了大并发的 `tp4` 档（4 卡，主线对比，和 InferenceX 一致）和 `tp8` 档（8 卡，补充），见 `scripts/serve_*.sh`。

## 本仓库各配置名

server 的档可以加后缀：SGLang 加 `-nospec` 关闭 DSpark，vLLM 加 `-spec` 打开 DSpark。按并发区间选拓扑的做法参照 InferenceX 的 B200 recipe（#14）。配置表见上方英文部分。

`GPUS_OVERRIDE=4,5,6,7`（server）/ `--device-offset 4`（DeepInfer）可以把 4 卡配置挪到机器的另一半卡上。

本仓库脚本和官方命令只有四处差别：镜像用钉了版本号的最新正式版；模型从本地目录挂进容器（不走 HF 下载）；加 `--served-model-name` 让三家报同一个模型名；SGLang 加 `--enable-metrics` 以便抓投机解码计数器。

</details>
