#!/bin/bash
# 在本地（不是 B200 上）运行：把远端结果拉回本仓库 results/<日期>/，然后可选清掉远端临时目录。
#   bash scripts/pull_results.sh [ssh_host] [--clean]
# B200 机器不固定，结果只在本地保留；--clean 会删除远端 $WORK（数据集、结果、agentx venv）。
set -euo pipefail
HOST="${1:-B200-1}"
REMOTE_WORK="${REMOTE_WORK:-/opt/dlami/nvme/bench_tmp}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/results/$(date +%Y%m%d_%H%M)"
mkdir -p "$LOCAL"
rsync -av --exclude 'agentx-venv' "$HOST:$REMOTE_WORK/results/" "$LOCAL/"
echo "pulled -> $LOCAL"
if [ "${2:-}" = "--clean" ]; then
    ssh "$HOST" "rm -rf '$REMOTE_WORK'"
    echo "removed $HOST:$REMOTE_WORK"
fi
