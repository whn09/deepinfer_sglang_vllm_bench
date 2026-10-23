#!/bin/bash
# （本地运行）spot 实例随时可能被回收：每隔 INTERVAL 秒
#   1. 在远端把 $WORK/results 和 logs 原样同步到 S3（$S3_DST，完整备份，含 server_metrics_export.json / timeslices 等大文件）；
#   2. 把结果镜像到本地 results/b200_live/（不进 git）。本地只留小文件：
#      server_metrics_export.json、profile_export_aiperf_timeslices.{json,csv} 不拉（报告用不到，要看去 S3 取）；
#      profile_export.jsonl 在远端压成 .jsonl.gz 再拉（bench/agentx_metrics.py 直接读 .gz）；超过 20 MB 的日志只在 S3。
#   bash scripts/sync_from_host.sh [ssh_host=B200-1] [interval_s=600]
HOST="${1:-B200-1}"; INTERVAL="${2:-600}"
REMOTE_WORK="${REMOTE_WORK:-/opt/dlami/nvme/bench_tmp}"
S3_DST="${S3_DST:-s3://datalab/deepinfer_bench/b200_live}"
DST="$(cd "$(dirname "$0")/.." && pwd)/results/b200_live"
mkdir -p "$DST"
BIG=(--exclude 'server_metrics_export.json' --exclude 'profile_export_aiperf_timeslices.*' --exclude 'profile_export.jsonl')
while true; do
    ssh "$HOST" "cd $REMOTE_WORK && \
        aws s3 sync results $S3_DST/results/ --region us-east-1 --only-show-errors --exclude '*agentx-venv/*' --exclude '*.tmp_*' --exclude '*.jsonl.gz'; \
        aws s3 sync logs $S3_DST/logs/ --region us-east-1 --only-show-errors; \
        find results -name profile_export.jsonl | while read -r f; do [ \"\$f.gz\" -nt \"\$f\" ] || gzip -kf \"\$f\"; done" 2>/dev/null
    rsync -aq --timeout=120 --exclude 'agentx-venv' --exclude '.tmp_*' "${BIG[@]}" "$HOST:$REMOTE_WORK/results/" "$DST/results/" 2>/dev/null
    rsync -aq --timeout=120 --max-size=20M --include '*.log' --include '*.sh' --exclude '*' "$HOST:$REMOTE_WORK/logs/" "$DST/logs/" 2>/dev/null
    rsync -aq --timeout=60 --include '*.sh' --exclude '*' "$HOST:$REMOTE_WORK/" "$DST/queues/" 2>/dev/null
    echo "$(date '+%F %T') synced ($(du -sh "$DST" | cut -f1))" >> "$DST/sync.log"
    sleep "$INTERVAL"
done
