#!/bin/bash
# 起 vLLM server（docker，后台），等到 /v1/models 可用为止。
#   bash scripts/serve_vllm.sh <profile>
#
# profile（任何档加 -spec 后缀 = 开 DSpark 投机解码）：
#   ref          官方参考命令原样：TP2、max-num-seqs 8、max-num-batched-tokens 4096、fp8 KV、
#                engram CPU offload，不开投机解码，用 0-1 号卡
#   tp4          4 卡（主线，InferenceX 在 c=1-128 用）：TP4，max-num-seqs 256，max-num-batched-tokens 16384
#   dep4         4 卡（InferenceX 在 c=64-128 用）：TP1 x DP4 + EP、deep_gemm_mega_moe、engram embedding_across_dp
#   dep4-deepep  同 dep4，all2all 换成 DeepEP（C 层消融：和 DeepInfer 同样的策略）
#   tp8          8 卡：TP8（补充）
#
# 投机解码：V4.1 去掉了 V3/V4 的 MTP 模块，DSpark 是这个 checkpoint 唯一的投机方法
# （drafter 的专家约多 14B 参数）。NVIDIA 上 vLLM 默认打开 adaptive verification；
# 官方说明里 "enable_adaptive_verification":false 是 AMD/ROCm 上的绕过写法，B200 上默认不传。
# 想强制关掉（验证完整 block）：VLLM_ADAPTIVE_VERIFICATION=false。
# 整个 spec config 也可以用 VLLM_SPEC_CONFIG='{...}' 直接覆盖。
#
# 与官方命令的差别：模型从本地目录挂进容器，加了 --served-model-name。
set -euo pipefail
source "$(dirname "$0")/env.sh"

PROFILE="${1:?usage: serve_vllm.sh <ref|tp4|dep4|dep4-deepep|tp8>[-spec]}"
# 优先用最新正式发行版（钉版本号，不用 latest）
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0}"
PORT="${VLLM_PORT:-8000}"
NAME=bench-vllm${CONTAINER_SUFFIX:-}

KERNEL_CFG='{"enable_flashinfer_autotune":true}'
ENGRAM_CFG='{"cpu_offload":true,"use_thp":true}'
case "${PROFILE%-spec}" in
    ref) GPUS=0,1;             PAR=(--tensor-parallel-size 2); MAX_SEQS=8;   MAX_BATCHED=4096 ;;
    tp4) GPUS=0,1,2,3;         PAR=(--tensor-parallel-size 4); MAX_SEQS=256; MAX_BATCHED=16384 ;;
    # InferenceX c=64-128：TP1 x DP4 + EP、MegaMoE；max-num-seqs 按每个 DP rank 计（4 x 64 = 256）
    dep4) GPUS=0,1,2,3;        PAR=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel)
          MAX_SEQS=64;  MAX_BATCHED=8192
          KERNEL_CFG='{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe"}'
          ENGRAM_CFG='{"cpu_offload":true,"embedding_across_dp":true}' ;;
    # C 层消融：和 DeepInfer 同样的策略（DP attention + EP + DeepEP）。deepep_v2 要 NCCL GIN，先用 V1 节点内路径
    dep4-deepep) GPUS=0,1,2,3; PAR=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --all2all-backend deepep_high_throughput)
          MAX_SEQS=64;  MAX_BATCHED=8192
          ENGRAM_CFG='{"cpu_offload":true,"embedding_across_dp":true}' ;;
    tp8) GPUS=0,1,2,3,4,5,6,7; PAR=(--tensor-parallel-size 8); MAX_SEQS=256; MAX_BATCHED=16384 ;;
    *) echo "unknown profile $PROFILE" >&2; exit 1 ;;
esac
# GPUS_OVERRIDE=4,5,6,7 可以把同一个档挪到别的卡上（启动阶段可以和别的任务并行；测量时 GPU 上只能有一个系统）
SPEC=()
if [[ "$PROFILE" == *-spec ]]; then
    if [ -n "${VLLM_SPEC_CONFIG:-}" ]; then
        SPEC_JSON="$VLLM_SPEC_CONFIG"
    else
        SPEC_JSON='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"'
        if [ -n "${VLLM_ADAPTIVE_VERIFICATION:-}" ]; then
            SPEC_JSON+=',"enable_adaptive_verification":'"$VLLM_ADAPTIVE_VERIFICATION"
        fi
        SPEC_JSON+='}'
    fi
    SPEC=(--speculative-config "$SPEC_JSON")
fi
echo "speculative: ${SPEC[*]:-none}"

GPUS="${GPUS_OVERRIDE:-$GPUS}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
    --gpus "\"device=$GPUS\"" \
    --privileged --ipc=host --network host \
    -v "$MODEL_DIR":/model:ro \
    -v "$HF_HOME":/root/.cache/huggingface \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    "$IMAGE" /model \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tokenizer-mode deepseek_v41 \
        "${PAR[@]}" \
        --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
        --kv-cache-dtype fp8 \
        --kernel-config "$KERNEL_CFG" \
        --gpu-memory-utilization 0.97 \
        --engram-config "$ENGRAM_CFG" \
        --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED" \
        --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092]}' \
        --max-cudagraph-capture-size 4092 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v41 \
        --mm-encoder-tp-mode data \
        --host 0.0.0.0 --port "$PORT" \
        "${SPEC[@]}" \
        ${VLLM_EXTRA_ARGS:-}

echo "vllm[$PROFILE] starting on GPUs $GPUS, port $PORT; logs: docker logs -f $NAME"
bash "$BENCH_ROOT/scripts/wait_ready.sh" "http://127.0.0.1:$PORT" "$NAME"
echo "NUM_GPUS=$(( $(tr -cd ',' <<<"$GPUS" | wc -c) + 1 ))"
