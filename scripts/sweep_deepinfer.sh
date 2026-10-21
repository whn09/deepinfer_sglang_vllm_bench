#!/bin/bash
# DeepInfer 扫一个场景的所有并发点（一个进程跑完全部点，只加载一次权重）。
#   bash scripts/sweep_deepinfer.sh <config名> <num_gpus> <nextn> <scenario> [run_deepinfer.py 额外参数...]
# 例：
#   bash scripts/sweep_deepinfer.sh deepinfer-ep8-nospec 8 0 sg-1k-1k
#   bash scripts/sweep_deepinfer.sh deepinfer-ep4-dspark5 4 5 sg-1k-1k
# 和另一个 DeepInfer 任务并行时设不同的 MASTER_PORT（rendezvous 端口 = MASTER_PORT - 50，默认 23333）
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_ROOT/scripts/scenarios.sh"

CONFIG="${1:?config}"; NGPU="${2:?num_gpus}"; NEXTN="${3:?nextn}"; SCN="${4:?scenario}"
shift 4

bash "$BENCH_ROOT/scripts/make_datasets.sh" "$SCN"
PLAN=()
for C in "${CONCURRENCIES[@]}"; do
    PLAN+=("$DATA_DIR/${SCN}_s${C}.jsonl@$C")
done

# 参数里的 --device-offset 决定用哪几张卡；等它们的显存释放
OFFSET=0; PREV=""
for a in "$@"; do [ "$PREV" = "--device-offset" ] && OFFSET="$a"; PREV="$a"; done
bash "$BENCH_ROOT/scripts/wait_gpus_free.sh" "$(seq -s, "$OFFSET" $((OFFSET + NGPU - 1)))" || exit 1

cd "$DEEPINFER_ROOT"   # DeepInfer 是源码树内构建，要在仓库根目录跑
python -u "$BENCH_ROOT/bench/run_deepinfer.py" --num-gpus "$NGPU" --nextn "$NEXTN" \
    --config "$CONFIG" --plan "${PLAN[@]}" --out "$RESULTS_DIR/results.jsonl" "$@" \
    2>&1 | tee -a "$RESULTS_DIR/${CONFIG}_${SCN}.log"
