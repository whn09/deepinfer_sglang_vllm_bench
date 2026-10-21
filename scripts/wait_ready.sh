#!/bin/bash
# 轮询 server 的 /v1/models 直到可用；容器提前退出就打印日志尾巴并报错。
#   bash scripts/wait_ready.sh <base_url> <container_name> [timeout_s]
set -uo pipefail
URL="$1"; NAME="$2"; TIMEOUT="${3:-3600}"
START=$(date +%s)
while true; do
    if curl -sf "$URL/v1/models" >/dev/null 2>&1; then
        echo "ready after $(( $(date +%s) - START ))s: $URL"
        exit 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
        echo "container $NAME exited:" >&2
        docker logs --tail 80 "$NAME" >&2
        exit 1
    fi
    if [ $(( $(date +%s) - START )) -gt "$TIMEOUT" ]; then
        echo "timeout after ${TIMEOUT}s" >&2
        docker logs --tail 80 "$NAME" >&2
        exit 1
    fi
    sleep 15
done
