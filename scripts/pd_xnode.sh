#!/bin/bash
# (Run on the laptop.) Cross-node PD split: PE on PE_HOST GPUs 0-3, CE on CE_HOST GPUs 0-3, KV over the network.
# Starts both trays, waits for /health, runs bench/pd_parity.py on the CE host, writes the fingerprints to
# $WORK/results/pd_xnode/<tag>.json there, and stops both trays.
#   bash scripts/pd_xnode.sh <tcp|mooncake> <tag>
# env: PE_HOST/CE_HOST (ssh aliases, default B200-1/B200-2), PE_IP/CE_IP (private IPs), DI_IMAGE (default
#      deepinfer:pd214-mcmain), MC_DEVICE (EFA device for mooncake, default rdmap113s0), IX_SYNTHETIC_AL (default 0:
#      real verification, so outputs are deterministic and comparable)
set -euo pipefail
TRANSPORT="${1:?tcp|mooncake}"; TAG="${2:?tag}"
PE_HOST="${PE_HOST:-B200-1}"; CE_HOST="${CE_HOST:-B200-2}"
PE_IP="${PE_IP:-172.31.32.216}"; CE_IP="${CE_IP:-172.31.37.169}"
IMAGE="${DI_IMAGE:-deepinfer:pd214-mcmain}"; AL="${IX_SYNTHETIC_AL:-0}"; DEV="${MC_DEVICE:-rdmap113s0}"
W=/opt/dlami/nvme/bench_tmp
COMMON="--gpus all --privileged --ipc=host --pid=host --network host --shm-size 32g --ulimit memlock=-1 \
 -e DEVICE_OFFSET=0 -e NCCL_GIN_TYPE=5 -e NCCL_IB_HCA=^=ibp115s0f0,ibp116s0f0 -e DEEPINFER_USE_NEXTN_DECODE=5 \
 -e DEEPINFER_SIMULATE_ACC_LEN=$AL -e DEEPINFER_HF_MODEL_DIR=/model -e DEEPINFER_JIT_CACHE_DIR=/jit \
 -v /opt/dlami/nvme/models/DeepSeek-V4.1-Flash-1017:/model:ro -v $W/cache/deepinfer_jit:/jit"
ARGS="--model /model --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 --dp 4 --thinking-mode thinking --kv-transport $TRANSPORT"
mc() { [ "$TRANSPORT" = mooncake ] && echo "--mooncake-protocol efa --mooncake-device $DEV --mooncake-host $1" || true; }

ssh "$PE_HOST" "docker rm -f bench-di-pe > /dev/null 2>&1; docker run -d --name bench-di-pe $COMMON -e MASTER_PORT=28333 $IMAGE \
    python -u serve/pd_server.py --role pe --port 8889 $ARGS $(mc $PE_IP) > /dev/null" 2> /dev/null
ssh "$CE_HOST" "docker rm -f bench-di-ce > /dev/null 2>&1; docker run -d --name bench-di-ce $COMMON -e MASTER_PORT=27333 $IMAGE \
    python -u serve/pd_server.py --role ce --port 8888 --kv-port 8900 --kv-host $CE_IP --pe-url http://$PE_IP:8889 $ARGS $(mc $CE_IP) > /dev/null" 2> /dev/null
echo "$(date +%H:%M:%S) PE $PE_HOST ($PE_IP) + CE $CE_HOST ($CE_IP), transport $TRANSPORT, image $IMAGE, AL $AL"
wait_health() {  # $1 host  $2 port  $3 container
    for _ in $(seq 240); do
        ssh "$1" "curl -s -m 5 http://127.0.0.1:$2/health" 2> /dev/null | grep -q '"status":"ok"' && return 0
        ssh "$1" "docker inspect -f '{{.State.Running}}' $3" 2> /dev/null | grep -q true || { ssh "$1" "docker logs --tail 40 $3" 2>&1 | tail -40; return 1; }
        sleep 15
    done
    return 1
}
wait_health "$PE_HOST" 8889 bench-di-pe && wait_health "$CE_HOST" 8888 bench-di-ce && echo "$(date +%H:%M:%S) both trays ready" || { echo "not ready" >&2; exit 1; }
ssh "$CE_HOST" "cd $W/repo && mkdir -p $W/results/pd_xnode && python3 bench/pd_parity.py --port 8888 --out $W/results/pd_xnode/$TAG.json" 2> /dev/null | tail -3
ssh "$PE_HOST" "docker logs bench-di-pe > $W/logs/pd_xnode_${TAG}_pe.log 2>&1; docker rm -f bench-di-pe > /dev/null" 2> /dev/null
ssh "$CE_HOST" "docker logs bench-di-ce > $W/logs/pd_xnode_${TAG}_ce.log 2>&1; grep -ciE 'TRANSFER_FAIL|fallback|retry' $W/logs/pd_xnode_${TAG}_ce.log | sed 's/^/ce transfer-problem lines: /'; docker rm -f bench-di-ce > /dev/null" 2> /dev/null
echo "$(date +%H:%M:%S) done $TAG"
