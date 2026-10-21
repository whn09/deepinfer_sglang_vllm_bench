#!/bin/bash
# （本地运行）spot 实例随时可能被回收：每隔 INTERVAL 秒把远端 $WORK 下的结果和日志镜像到本地 results/b200_live/（不进 git）。
#   bash scripts/sync_from_host.sh [ssh_host=B200-1] [interval_s=600]
HOST="${1:-B200-1}"; INTERVAL="${2:-600}"
REMOTE_WORK="${REMOTE_WORK:-/opt/dlami/nvme/bench_tmp}"
DST="$(cd "$(dirname "$0")/.." && pwd)/results/b200_live"
mkdir -p "$DST"
while true; do
    rsync -aq --timeout=120 --exclude 'agentx-venv' --exclude '.tmp_*' "$HOST:$REMOTE_WORK/results/" "$DST/results/" 2>/dev/null
    rsync -aq --timeout=120 --include '*.log' --include '*.sh' --exclude '*' "$HOST:$REMOTE_WORK/logs/" "$DST/logs/" 2>/dev/null
    rsync -aq --timeout=60 --include '*.sh' --exclude '*' "$HOST:$REMOTE_WORK/" "$DST/queues/" 2>/dev/null
    echo "$(date '+%F %T') synced ($(du -sh "$DST" | cut -f1))" >> "$DST/sync.log"
    sleep "$INTERVAL"
done
