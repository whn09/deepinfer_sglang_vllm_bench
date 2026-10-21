#!/bin/bash
# 起一个 server 档 → 扫一个或多个场景 → 停掉。用来把多组对比排成队列自动跑。
#   bash scripts/run_profile.sh <sglang|vllm> <profile> <config名> <num_gpus> <scenario...>
# 例：GPUS_OVERRIDE=4,5,6,7 SGLANG_PORT=30001 bash scripts/run_profile.sh sglang dpa4 sglang-dpa4 4 sg-1k-1k
set -uo pipefail
source "$(dirname "$0")/env.sh"
BACKEND="$1"; PROFILE="$2"; CONFIG="$3"; NGPU="$4"; shift 4
LOG="$WORK/logs/run_${CONFIG}.log"
echo "=== $(date +%H:%M:%S) serve $BACKEND $PROFILE (GPUs ${GPUS_OVERRIDE:-default})" | tee -a "$LOG"
if ! bash "$BENCH_ROOT/scripts/serve_${BACKEND}.sh" "$PROFILE" >> "$LOG" 2>&1; then
    echo "=== serve failed, see $LOG" | tee -a "$LOG"
    docker logs "bench-${BACKEND}${CONTAINER_SUFFIX:-}" > "$WORK/logs/run_${CONFIG}.server_failed.log" 2>&1
    docker rm -f "bench-${BACKEND}${CONTAINER_SUFFIX:-}" > /dev/null 2>&1
    exit 1
fi
for SCN in "$@"; do
    echo "=== $(date +%H:%M:%S) sweep $CONFIG $SCN" | tee -a "$LOG"
    bash "$BENCH_ROOT/scripts/sweep_server.sh" "$BACKEND" "$CONFIG" "$NGPU" "$SCN" >> "$LOG" 2>&1
done
docker logs "bench-${BACKEND}${CONTAINER_SUFFIX:-}" > "$WORK/logs/run_${CONFIG}.server.log" 2>&1
docker rm -f "bench-${BACKEND}${CONTAINER_SUFFIX:-}" > /dev/null 2>&1
echo "=== $(date +%H:%M:%S) done $CONFIG" | tee -a "$LOG"
