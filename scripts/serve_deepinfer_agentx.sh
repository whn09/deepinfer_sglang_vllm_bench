#!/bin/bash
# DeepInfer OpenAI server for AgentX (#5): image deepinfer:agentx (docker/Dockerfile.deepinfer-agentx),
# EP4 = 4 DP ranks, Engram over EFA (#10), host prefix cache (#144), synthetic AL (golden 3.51, as InferenceX).
#   bash scripts/serve_deepinfer_agentx.sh            # GPUs 0-3, port 8888
#   HALF=b bash scripts/serve_deepinfer_agentx.sh     # GPUs 4-7, port 8889
# 环境变量：IX_SYNTHETIC_AL（默认 3.51，0 = 真实验证）、DI_THINKING（thinking|chat，默认 thinking，对应 golden AL 的 thinking_on）、
#          DI_HICACHE_GB（每 rank 主机前缀缓存，默认 32；0 = 关）、DI_EXTRA（追加给 openai_server.py 的参数）
set -euo pipefail
source "$(dirname "$0")/env.sh"
HALF="${HALF:-a}"
if [ "$HALF" = b ]; then OFFSET=4; PORT=8889; MPORT=27333; SUF=-b; else OFFSET=0; PORT=8888; MPORT=28333; SUF=; fi
NAME="bench-deepinfer$SUF"
AL="${IX_SYNTHETIC_AL:-3.51}"
bash "$BENCH_ROOT/scripts/wait_gpus_free.sh" "$(seq -s, $OFFSET $((OFFSET + 3)))" || exit 1
docker rm -f "$NAME" > /dev/null 2>&1 || true
# --privileged 下选卡靠 DEVICE_OFFSET（DeepInfer 自己设 CUDA_VISIBLE_DEVICES）；两个实例并行时 MASTER_PORT 要错开
docker run -d --name "$NAME" --gpus all --privileged --ipc=host --network host --shm-size 32g --ulimit memlock=-1 \
    -e DEVICE_OFFSET=$OFFSET -e MASTER_PORT=$MPORT \
    -e NCCL_GIN_TYPE=5 -e "NCCL_IB_HCA=^=ibp115s0f0,ibp116s0f0" \
    -e DEEPINFER_USE_NEXTN_DECODE=5 -e DEEPINFER_SIMULATE_ACC_LEN="$AL" \
    -e DEEPINFER_HF_MODEL_DIR=/model -v "$MODEL_DIR":/model:ro \
    deepinfer:agentx python -u serve/openai_server.py --model /model --served-model-name "$SERVED_MODEL_NAME" \
        --dp 4 --port "$PORT" --thinking-mode "${DI_THINKING:-thinking}" --hicache-size-gb "${DI_HICACHE_GB:-32}" \
        ${DI_EXTRA:-} > /dev/null
echo "deepinfer[agentx] GPUs $OFFSET-$((OFFSET + 3)) port $PORT synthetic_al=$AL; logs: docker logs -f $NAME"
bash "$BENCH_ROOT/scripts/wait_ready.sh" "http://127.0.0.1:$PORT" "$NAME" "${DI_READY_TIMEOUT:-3600}"
