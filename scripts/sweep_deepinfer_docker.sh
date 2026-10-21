#!/bin/bash
# 在 DeepInfer 容器里扫一个场景（#9 / #11）。本仓库、数据集、结果目录挂进容器，口径和裸机的 sweep_deepinfer.sh 相同。
#   IMAGE=deepinfer:efa DOCKER_EXTRA="..." bash scripts/sweep_deepinfer_docker.sh <config名> <num_gpus> <nextn> <scenario> [run_deepinfer.py 参数...]
# 例（#99 NVLink Engram，不需要 privileged）：
#   IMAGE=deepinfer:pr99 DOCKER_EXTRA='--gpus "device=0,1,2,3" -e EP_DISABLE_GIN=1' \
#     bash scripts/sweep_deepinfer_docker.sh deepinfer-docker-nvlink-ep4-dspark5 4 5 sg-1k-1k
# 例（EFA GDA，privileged 下用 --device-offset 选卡）：
#   IMAGE=deepinfer:efa DOCKER_EXTRA='--gpus all --privileged --ulimit memlock=-1 -e NCCL_GIN_TYPE=5 -e NCCL_IB_HCA=^=ibp115s0f0,ibp116s0f0' \
#     bash scripts/sweep_deepinfer_docker.sh deepinfer-docker-efa-ep4-dspark5 4 5 sg-1k-1k --device-offset 4
# 两个 DeepInfer 任务同时跑（host 网络）时必须用不同的 MASTER_PORT（rendezvous 端口 = MASTER_PORT - 50，默认 23333），
# 否则后起的那个 EADDRINUSE 失败、容器还会挂住不退出：DOCKER_EXTRA='... -e MASTER_PORT=24333'
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_ROOT/scripts/scenarios.sh"
: "${IMAGE:?set IMAGE}"
CONFIG="${1:?config}"; NGPU="${2:?num_gpus}"; NEXTN="${3:?nextn}"; SCN="${4:?scenario}"; shift 4

bash "$BENCH_ROOT/scripts/make_datasets.sh" "$SCN"
PLAN=()
for C in "${CONCURRENCIES[@]}"; do
    PLAN+=("/data/${SCN}_s${C}.jsonl@$C")
done

# DOCKER_EXTRA 按空白拆成参数；--gpus 的值写成 "device=0,1,2,3"（引号是 docker 要的字面量，不会被 shell 去掉）
read -r -a EXTRA <<< "${DOCKER_EXTRA:-}"
docker run --rm --ipc=host --network host --shm-size 32g "${EXTRA[@]}" \
    -v "$MODEL_DIR":/model:ro -v "$BENCH_ROOT":/bench:ro -v "$DATA_DIR":/data:ro -v "$RESULTS_DIR":/results \
    -e DEEPINFER_ROOT=/workspace/DeepInfer \
    "$IMAGE" python -u /bench/bench/run_deepinfer.py --model /model --num-gpus "$NGPU" --nextn "$NEXTN" \
        --config "$CONFIG" --plan "${PLAN[@]}" --out /results/results.jsonl "$@" \
    2>&1 | tee -a "$RESULTS_DIR/${CONFIG}_${SCN}.log"
