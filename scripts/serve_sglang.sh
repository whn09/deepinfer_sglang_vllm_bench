#!/bin/bash
# 起 SGLang server（docker，后台），等到 /v1/models 可用为止。
#   bash scripts/serve_sglang.sh <profile>
#
# profile：
#   ref         官方参考命令原样：TP4/EP4、DSPARK block 5、cuda graph decode bs ≤ 64，用 0-3 号卡
#   ref-nospec  同上，去掉投机解码
#   tp8         8 卡：TP8/EP8、DSPARK block 5、cuda graph decode bs ≤ 256
#   tp8-nospec  同上，去掉投机解码
#
# 与官方命令唯一的差别：模型从本地目录挂进容器（不走 HF 下载），加了 --served-model-name
# 让三家报同一个模型名，加了 --enable-metrics 用来抓投机解码的接受率。
set -euo pipefail
source "$(dirname "$0")/env.sh"

PROFILE="${1:?usage: serve_sglang.sh <ref|ref-nospec|tp8|tp8-nospec>}"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-dsv41}"
PORT="${SGLANG_PORT:-30000}"
NAME=bench-sglang

SPEC=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
case "$PROFILE" in
    ref)        GPUS=0,1,2,3;       PAR=(--tp 4 --ep-size 4); GRAPH_BS=64 ;;
    ref-nospec) GPUS=0,1,2,3;       PAR=(--tp 4 --ep-size 4); GRAPH_BS=64;  SPEC=() ;;
    tp8)        GPUS=0,1,2,3,4,5,6,7; PAR=(--tp 8 --ep-size 8); GRAPH_BS=256 ;;
    tp8-nospec) GPUS=0,1,2,3,4,5,6,7; PAR=(--tp 8 --ep-size 8); GRAPH_BS=256; SPEC=() ;;
    *) echo "unknown profile $PROFILE" >&2; exit 1 ;;
esac

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
        --mem-fraction-static 0.8 \
        "${SPEC[@]}" \
        --cuda-graph-max-bs-decode "$GRAPH_BS" \
        --reasoning-parser auto \
        --tool-call-parser auto \
        --enable-metrics \
        --host 0.0.0.0 --port "$PORT" \
        ${SGLANG_EXTRA_ARGS:-}

echo "sglang[$PROFILE] starting on GPUs $GPUS, port $PORT; logs: docker logs -f $NAME"
bash "$BENCH_ROOT/scripts/wait_ready.sh" "http://127.0.0.1:$PORT" "$NAME"
echo "NUM_GPUS=$(( $(tr -cd ',' <<<"$GPUS" | wc -c) + 1 ))"
