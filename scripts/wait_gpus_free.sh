#!/bin/bash
# 等指定 GPU 的显存真正释放（上一个任务退出后，进程收尾和显存回收有延迟）再启动下一个任务。
#   bash scripts/wait_gpus_free.sh 4,5,6,7 [max_used_mib=2000] [timeout_s=900]
GPUS="$1"; MAX="${2:-2000}"; TIMEOUT="${3:-900}"
START=$(date +%s)
while true; do
    BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | awk -F', ' -v m="$MAX" '$2 > m {print $1":"$2"MiB"}' | tr '\n' ' ')
    [ -z "$BUSY" ] && exit 0
    if [ $(( $(date +%s) - START )) -gt "$TIMEOUT" ]; then
        echo "GPUs still busy after ${TIMEOUT}s: $BUSY" >&2
        exit 1
    fi
    sleep 10
done
