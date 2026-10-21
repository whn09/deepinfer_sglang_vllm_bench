#!/bin/bash
# 对一个已经起好的 sglang / vllm server 扫一个场景的所有并发点。
#   bash scripts/sweep_server.sh <sglang|vllm> <config名> <num_gpus> <scenario>
# 例：
#   bash scripts/serve_sglang.sh ref
#   bash scripts/sweep_server.sh sglang sglang-ref 4 sg-1k-1k
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_ROOT/scripts/scenarios.sh"

BACKEND="${1:?backend}"; CONFIG="${2:?config}"; NGPU="${3:?num_gpus}"; SCN="${4:?scenario}"
case "$BACKEND" in
    sglang) URL="http://127.0.0.1:${SGLANG_PORT:-30000}" ;;
    vllm)   URL="http://127.0.0.1:${VLLM_PORT:-8000}" ;;
    *) echo "unknown backend" >&2; exit 1 ;;
esac

bash "$BENCH_ROOT/scripts/make_datasets.sh" "$SCN"
for C in "${CONCURRENCIES[@]}"; do
    python "$BENCH_ROOT/bench/client_openai.py" --backend "$BACKEND" --base-url "$URL" \
        --dataset "$DATA_DIR/${SCN}_s${C}.jsonl" --concurrency "$C" --num-gpus "$NGPU" \
        --config "$CONFIG" --out "$RESULTS_DIR/results.jsonl" ${CLIENT_EXTRA_ARGS:-} \
        2>&1 | tee -a "$RESULTS_DIR/${CONFIG}_${SCN}.log"
done
