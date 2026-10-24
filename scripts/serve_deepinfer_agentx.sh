#!/bin/bash
# DeepInfer OpenAI server for AgentX (#5): image deepinfer:agentx (docker/Dockerfile.deepinfer-agentx),
# EP4 = 4 DP ranks, Engram over EFA (#10), host prefix cache (#144), synthetic AL (golden 3.51, as InferenceX).
#   bash scripts/serve_deepinfer_agentx.sh            # GPUs 0-3, port 8888
#   HALF=b bash scripts/serve_deepinfer_agentx.sh     # GPUs 4-7, port 8889
# 环境变量：IX_SYNTHETIC_AL（默认 3.51，0 = 真实验证）、DI_THINKING（thinking|chat，默认 thinking，对应 golden AL 的 thinking_on）、
#          DI_HICACHE_GB（每 rank 主机前缀缓存，默认 32；0 = 关）、DI_ROUTING（sticky-balanced|affinity|least-loaded，
#          默认 sticky-balanced：哈希亲和在会话少时会让 DP rank 负载严重不均）、DI_EXTRA（追加给 openai_server.py 的参数）、
#          DI_IMAGE（默认 deepinfer:agentx；EPLB 开关要 deepinfer:agentx-eplb 或重新构建的 agentx）、
#          DI_TOKENIZE_VERIFY（前 N 条请求对比并行/整段 tokenize，要 deepinfer:agentx-tok）、DI_TRACE（主机上的 jsonl 路径：开 DEEPINFER_SERVER_TRACE 逐请求 TTFT 埋点，要 deepinfer:agentx-trace，#17）
set -euo pipefail
source "$(dirname "$0")/env.sh"
HALF="${HALF:-a}"
if [ "$HALF" = b ]; then OFFSET=4; PORT=8889; MPORT=27333; SUF=-b; else OFFSET=0; PORT=8888; MPORT=28333; SUF=; fi
NAME="bench-deepinfer$SUF"
AL="${IX_SYNTHETIC_AL:-3.51}"
bash "$BENCH_ROOT/scripts/wait_gpus_free.sh" "$(seq -s, $OFFSET $((OFFSET + 3)))" || exit 1
docker rm -f "$NAME" > /dev/null 2>&1 || true
# DI_L3=1: Mooncake Store L3 under the host prefix cache (#20; needs deepinfer:agentx-l3 and scripts/serve_mooncake_store.sh)
L3_ARGS=(); [ "${DI_L3:-0}" = 1 ] && L3_ARGS=(-e 'DEEPINFER_HICACHE_MOONCAKE_STORE={"local_hostname":"127.0.0.1","metadata_server":"http://127.0.0.1:18080/metadata","master_server_addr":"127.0.0.1:50551","protocol":"tcp","global_segment_size":0}')
# DI_JIT_CACHE: persistent TileLang / DeepGEMM / DeepEP JIT cache on the host (default $WORK/cache/deepinfer_jit; "0" = off).
# Without it every fresh container recompiles: the first requests take 88-170 s and new shapes add 12-14 s spikes later.
JIT_DIR="${DI_JIT_CACHE:-$WORK/cache/deepinfer_jit}"; JIT_ARGS=()
[ "$JIT_DIR" != 0 ] && mkdir -p "$JIT_DIR" && JIT_ARGS=(-e DEEPINFER_JIT_CACHE_DIR=/jit -v "$JIT_DIR":/jit)
TRACE_ARGS=(); [ -n "${DI_TRACE:-}" ] && TRACE_ARGS=(-e "DEEPINFER_SERVER_TRACE=/trace/$(basename "$DI_TRACE")" -v "$(dirname "$DI_TRACE")":/trace)
# --privileged 下选卡靠 DEVICE_OFFSET（DeepInfer 自己设 CUDA_VISIBLE_DEVICES）；两个实例并行时 MASTER_PORT 要错开
docker run -d --name "$NAME" --gpus all --privileged --ipc=host --network host --shm-size 32g --ulimit memlock=-1 \
    -e DEVICE_OFFSET=$OFFSET -e MASTER_PORT=$MPORT \
    -e NCCL_GIN_TYPE=5 -e "NCCL_IB_HCA=^=ibp115s0f0,ibp116s0f0" \
    -e DEEPINFER_USE_NEXTN_DECODE=5 -e DEEPINFER_SIMULATE_ACC_LEN="$AL" -e DEEPINFER_TOKENIZE_VERIFY="${DI_TOKENIZE_VERIFY:-0}" \
    -e DEEPINFER_HF_MODEL_DIR=/model -v "$MODEL_DIR":/model:ro "${TRACE_ARGS[@]}" "${L3_ARGS[@]}" "${JIT_ARGS[@]}" \
    "${DI_IMAGE:-deepinfer:agentx}" python -u serve/openai_server.py --model /model --served-model-name "$SERVED_MODEL_NAME" \
        --dp 4 --port "$PORT" --thinking-mode "${DI_THINKING:-thinking}" --hicache-size-gb "${DI_HICACHE_GB:-32}" \
        --routing "${DI_ROUTING:-sticky-balanced}" \
        ${DI_EXTRA:-} > /dev/null
echo "deepinfer[agentx] GPUs $OFFSET-$((OFFSET + 3)) port $PORT synthetic_al=$AL image=${DI_IMAGE:-deepinfer:agentx} extra=${DI_EXTRA:-}; logs: docker logs -f $NAME"
# /v1/models 在引擎加载完之前就会应答；要等 /health 报告 4 个 rank 都 ready
bash "$BENCH_ROOT/scripts/wait_ready.sh" "http://127.0.0.1:$PORT" "$NAME" "${DI_READY_TIMEOUT:-3600}"
START=$(date +%s)
until curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then docker logs --tail 60 "$NAME" >&2; exit 1; fi
    [ $(( $(date +%s) - START )) -gt "${DI_READY_TIMEOUT:-3600}" ] && { echo "engine not ready" >&2; exit 1; }
    sleep 15
done
echo "engine ready after $(( $(date +%s) - START ))s"
