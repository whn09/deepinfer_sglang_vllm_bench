#!/bin/bash
# 为一个场景的每个并发点生成独立数据集（seed = 并发值），三家共用同一批文件。
#   bash scripts/make_datasets.sh sg-1k-1k
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_ROOT/scripts/scenarios.sh"

SCN="${1:?usage: make_datasets.sh <scenario>}"
read -r SRC IN OUT <<<"${SCENARIOS[$SCN]:?unknown scenario $SCN}"

for C in "${CONCURRENCIES[@]}"; do
    N=$(num_requests_for "$C")
    F="$DATA_DIR/${SCN}_s${C}.jsonl"
    if [ -s "$F" ]; then
        echo "exists: $F"
        continue
    fi
    python "$BENCH_ROOT/bench/make_dataset.py" --tokenizer "$MODEL_DIR" --source "$SRC" \
        --input-len "$IN" --output-len "$OUT" --num-requests "$N" --seed "$C" --out "$F"
done
