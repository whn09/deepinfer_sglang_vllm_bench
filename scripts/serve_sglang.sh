#!/bin/bash
# 起 SGLang server（docker，后台），等到 /v1/models 可用为止。
#   bash scripts/serve_sglang.sh <profile>
#
# profile（任何档加 -nospec 后缀 = 关掉投机解码）：
#   ref          官方参考命令原样：TP4/EP4、DSPARK block 5、cuda graph decode bs ≤ 64，用 0-3 号卡
#   tp4          4 卡（主线）：TP4/EP4、cuda graph decode bs ≤ 256（ref 只到 64，扫到 c=128 会退回 eager）
#   tp4-ep1      4 卡：纯 TP4、不开 EP（InferenceX 在 c=1-8 用）
#   dpa4         4 卡：TP4 + DP4 + EP4、DP attention、MegaMoE（InferenceX 在 c=48-256 用）
#   dpa4-deepep  同 dpa4，MoE 通信换成 DeepEP（C 层消融：和 DeepInfer 同样的策略）
#   tp8          8 卡：TP8/EP8（补充）
#
# 与官方命令唯一的差别：模型从本地目录挂进容器（不走 HF 下载），加了 --served-model-name
# 让三家报同一个模型名，加了 --enable-metrics 用来抓投机解码的接受率。
set -euo pipefail
source "$(dirname "$0")/env.sh"

PROFILE="${1:?usage: serve_sglang.sh <ref|tp4|tp4-ep1|dpa4|dpa4-deepep|tp8>[-nospec]}"
# 优先用最新正式发行版（钉版本号，不用 latest）；正式版不支持这个模型时再退到专用镜像：
#   SGLANG_IMAGE=lmsysorg/sglang:dev-dsv41
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.22-cu130}"
PORT="${SGLANG_PORT:-30000}"
NAME=bench-sglang${CONTAINER_SUFFIX:-}

# GPUS_OVERRIDE=4,5,6,7 可以把同一个档挪到别的卡上（启动阶段可以和别的任务并行；测量时 GPU 上只能有一个系统）
SPEC=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
# 任何档加 -nospec 后缀 = 同样拓扑、关掉投机解码
BASE="${PROFILE%-nospec}"
[ "$BASE" != "$PROFILE" ] && SPEC=()
EXTRA=()
MEM=0.8
# DP attention 档：负载按请求数在 DP rank 间均衡（定长场景没有前缀复用，不需要 InferenceX 的 cache-aware router / HiCache）
DPA4=(--tp 4 --dp-size 4 --ep-size 4 --enable-dp-attention --enable-dp-lm-head --load-balance-method total_requests --max-running-requests 256)
case "$BASE" in
    ref)         GPUS=0,1,2,3;         PAR=(--tp 4 --ep-size 4); GRAPH_BS=64 ;;
    tp4)         GPUS=0,1,2,3;         PAR=(--tp 4 --ep-size 4); GRAPH_BS=256 ;;
    # InferenceX c=1-8：纯 TP、不开 EP
    # 不开 EP 时每卡权重更多，0.8 放不下 KV（报 minimum viable 0.833）；InferenceX 这里用 0.88
    tp4-ep1)     GPUS=0,1,2,3;         PAR=(--tp 4);             GRAPH_BS=256; MEM=0.88 ;;
    # InferenceX c=48-256：DP attention + EP4 + MegaMoE；cuda graph bs 按每个 DP rank 计
    dpa4)        GPUS=0,1,2,3;         PAR=("${DPA4[@]}");       GRAPH_BS=64
                 EXTRA=(--moe-a2a-backend megamoe --moe-runner-backend flashinfer_mxfp4) ;;
    # C 层消融：和 DeepInfer 同样的策略（DP attention + EP + DeepEP）。DeepEP V2 要 NCCL GIN，先用 V1 节点内路径
    dpa4-deepep) GPUS=0,1,2,3;         PAR=("${DPA4[@]}");       GRAPH_BS=64
                 EXTRA=(--moe-a2a-backend deepep) ;;
    tp8)         GPUS=0,1,2,3,4,5,6,7; PAR=(--tp 8 --ep-size 8); GRAPH_BS=256 ;;
    *) echo "unknown profile $PROFILE" >&2; exit 1 ;;
esac

GPUS="${GPUS_OVERRIDE:-$GPUS}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
    --gpus "\"device=$GPUS\"" \
    --shm-size 32g --ipc=host --network host \
    -v "$MODEL_DIR":/model:ro \
    -v "$HF_HOME":/root/.cache/huggingface \
    ${HF_TOKEN:+--env HF_TOKEN=$HF_TOKEN} \
    "$IMAGE" \
    sglang serve \
        --trust-remote-code \
        --model-path /model \
        --served-model-name "$SERVED_MODEL_NAME" \
        "${PAR[@]}" \
        --mem-fraction-static "$MEM" \
        "${SPEC[@]}" \
        "${EXTRA[@]}" \
        --cuda-graph-max-bs-decode "$GRAPH_BS" \
        --reasoning-parser auto \
        --tool-call-parser auto \
        --enable-metrics \
        --host 0.0.0.0 --port "$PORT" \
        ${SGLANG_EXTRA_ARGS:-}

echo "sglang[$PROFILE] starting on GPUs $GPUS, port $PORT; logs: docker logs -f $NAME"
bash "$BENCH_ROOT/scripts/wait_ready.sh" "http://127.0.0.1:$PORT" "$NAME"
echo "NUM_GPUS=$(( $(tr -cd ',' <<<"$GPUS" | wc -c) + 1 ))"
