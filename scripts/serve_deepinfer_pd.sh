#!/bin/bash
# DeepInfer prefill/decode split on one node (#18): PE (prefill tray) on GPUs 0-3, CE (decode tray) on GPUs 4-7,
# serve/pd_server.py from deepseek-ai/DeepInfer#197 (image deepinfer:pd, docker/Dockerfile.deepinfer-agentx with the PD patch).
# KV + SWA state go PE -> CE over TCP on loopback (#197's transport); the client talks to the CE on port 8888.
#   bash scripts/serve_deepinfer_pd.sh
# 环境变量：IX_SYNTHETIC_AL（默认 3.51）、DI_THINKING、DI_HICACHE_GB（PE 前缀缓存，默认 32）、DI_ROUTING（默认 sticky-balanced）、
#          DI_IMAGE（默认 deepinfer:pd）、DI_EXTRA（两边都追加，默认 --eplb-interval 0）、DI_PE_EXTRA / DI_CE_EXTRA、
#          DI_KV_TRANSPORT（tcp|mooncake，默认 tcp；mooncake = Transfer Engine nvlink_intra，GPU arena 直写）
set -euo pipefail
source "$(dirname "$0")/env.sh"
AL="${IX_SYNTHETIC_AL:-3.51}"; IMAGE="${DI_IMAGE:-deepinfer:pd}"
EXTRA="${DI_EXTRA:---eplb-interval 0}"
bash "$BENCH_ROOT/scripts/wait_gpus_free.sh" "0,1,2,3,4,5,6,7" || exit 1
docker rm -f bench-di-pe bench-di-ce > /dev/null 2>&1 || true
COMMON=(--gpus all --privileged --ipc=host --network host --shm-size 32g --ulimit memlock=-1
        -e NCCL_GIN_TYPE=5 -e "NCCL_IB_HCA=^=ibp115s0f0,ibp116s0f0"
        -e DEEPINFER_USE_NEXTN_DECODE=5 -e DEEPINFER_SIMULATE_ACC_LEN="$AL"
        -e DEEPINFER_HF_MODEL_DIR=/model -v "$MODEL_DIR":/model:ro)
ARGS=(--model /model --served-model-name "$SERVED_MODEL_NAME" --dp 4 --thinking-mode "${DI_THINKING:-thinking}"
      --routing "${DI_ROUTING:-sticky-balanced}" --kv-transport "${DI_KV_TRANSPORT:-tcp}")
[ "${DI_KV_TRANSPORT:-tcp}" = mooncake ] && ARGS+=(--mooncake-protocol nvlink_intra --mooncake-host 127.0.0.1)
# 两个引擎在同一台机上：DEVICE_OFFSET 选卡，MASTER_PORT 错开（gloo/rendezvous 端口由它派生）
docker run -d --name bench-di-pe "${COMMON[@]}" -e DEVICE_OFFSET=0 -e MASTER_PORT=28333 "$IMAGE" \
    python -u serve/pd_server.py --role pe --port 8889 --hicache-size-gb "${DI_HICACHE_GB:-32}" "${ARGS[@]}" $EXTRA ${DI_PE_EXTRA:-} > /dev/null
docker run -d --name bench-di-ce "${COMMON[@]}" -e DEVICE_OFFSET=4 -e MASTER_PORT=27333 "$IMAGE" \
    python -u serve/pd_server.py --role ce --port 8888 --kv-port 8900 --kv-host 127.0.0.1 --pe-url http://127.0.0.1:8889 \
    "${ARGS[@]}" $EXTRA ${DI_CE_EXTRA:-} > /dev/null
echo "deepinfer PD: PE GPUs 0-3 :8889, CE GPUs 4-7 :8888 (KV 8900+rank, transport ${DI_KV_TRANSPORT:-tcp}) image=$IMAGE extra=$EXTRA; logs: docker logs -f bench-di-pe|bench-di-ce"
START=$(date +%s)
for spec in pe:8889 ce:8888; do
    name=bench-di-${spec%%:*}; port=${spec##*:}
    until curl -s -m 5 "http://127.0.0.1:$port/health" | grep -q '"status":"ok"'; do
        if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]; then docker logs --tail 60 "$name" >&2; exit 1; fi
        [ $(( $(date +%s) - START )) -gt "${DI_READY_TIMEOUT:-3600}" ] && { echo "$name not ready" >&2; exit 1; }
        sleep 15
    done
    echo "$name ready after $(( $(date +%s) - START ))s"
done
