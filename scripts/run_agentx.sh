#!/bin/bash
# 用 InferenceX AgentX 压一个已经起好的 server，参数照 InferenceX 的 argv builder
# （inferencex-e2e/infx/bench/agentic/replay.py:138-167）；跑完用 InferenceX 自己的代码算指标（bench/agentx_metrics.py）。
#   bash scripts/run_agentx.sh <sglang|vllm> <config名> <num_gpus> <topology> <concurrency>
# 例：IX_CONC=8 bash scripts/serve_vllm.sh ix-tp4 && bash scripts/run_agentx.sh vllm vllm-ix-tp4 4 tp4 8
#
# InferenceX 每个并发点都单独起一次 server、只跑一遍 3600s；这里一次调用也只跑一个并发点。
# AGENTX_DURATION（默认 3600）；低于 900s 时 harness 要求 --unsafe-override，结果不能和公开数据比。
set -euo pipefail
source "$(dirname "$0")/env.sh"

BACKEND="${1:?backend}"; CONFIG="${2:?config}"; NGPU="${3:?num_gpus}"; TOPO="${4:?topology}"; C="${5:?concurrency}"
case "$BACKEND" in
    sglang) URL="http://127.0.0.1:${SGLANG_PORT:-30000}" ;;
    vllm)   URL="http://127.0.0.1:${VLLM_PORT:-8000}" ;;
    *) echo "unknown backend" >&2; exit 1 ;;
esac
METRICS_URL="$URL/metrics"
# serve_ix.sh 起的配置：AgentX 连 recipe 要求的 frontend（router），server 的 /metrics 仍然直接抓 server；
# recipe 的 benchmark.env 里客户端相关的设置照 InferenceX replay.py 的方式应用
IX_FLAGS=()
if [ -n "${IX_JSON:-}" ]; then
    URL="$(cat "$WORK/logs/ix_endpoint${CONTAINER_SUFFIX:-}")"
    while IFS='=' read -r k v; do
        case "$k" in
            AGENTIC_WARMUP_GRACE_PERIOD) IX_FLAGS+=(--warmup-grace-period "$v") ;;
            AIPERF_MAX_CONTEXT_LENGTH)   IX_FLAGS+=(--max-context-length "$v") ;;
            AIPERF_BENCHMARK_GRACE_PERIOD) IX_FLAGS+=(--benchmark-grace-period "$v") ;;
            AIPERF_*) export "$k=$v" ;;
        esac
    done < <(python3 -c "import json;[print(f'{k}={v}') for k,v in json.load(open('$IX_JSON'))['benchmark_env'].items()]")
fi

DATASET="${AGENTX_DATASET:-semianalysis_cc_traces_weka_062126}"
VERSION="${AGENTX_VERSION:-1.0.12}"           # InferenceX: infx/bench/agentic/requirements.txt
DURATION="${AGENTX_DURATION:-3600}"            # InferenceX: DEFAULT_AGENTIC_DURATION_SECONDS
INFERENCEX_E2E="${INFERENCEX_E2E:-$WORK/inferencex-e2e}"

# agentx-harness 带的是 fork 版 aiperf，必须装在独立 venv 里；InferenceX 用 Python 3.11–3.13 皆可
VENV="$WORK/agentx-venv-$VERSION"
if [ ! -x "$VENV/bin/agentx" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q "agentx-harness==$VERSION" "datasets>=4.7.0"
fi

EXTRA=()
[ "$DURATION" -lt 900 ] && EXTRA+=(--unsafe-override)
OUT="$RESULTS_DIR/agentx/$CONFIG/c$C"
mkdir -p "$OUT"
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10     # InferenceX workflow env
"$VENV/bin/agentx" profile \
    --url "$URL" \
    --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_DIR" \
    --concurrency "$C" --benchmark-duration "$DURATION" \
    --tokenizer-trust-remote-code \
    --server-metrics "$METRICS_URL" \
    --output-artifact-dir "$OUT/aiperf_artifacts" \
    --public-dataset "$DATASET" \
    --ui simple \
    "${EXTRA[@]}" "${IX_FLAGS[@]}" ${AGENTX_EXTRA_ARGS:-} \
    2>&1 | tee "$OUT/agentx.log"

INFERENCEX_E2E="$INFERENCEX_E2E" python3 "$BENCH_ROOT/bench/agentx_metrics.py" "$OUT/aiperf_artifacts" \
    --num-gpus "$NGPU" --conc "$C" --topology "$TOPO" \
    --published "$BENCH_ROOT/results/inferencex_public/dsv41flash_b200_${BACKEND}_agentic.json" \
    --out "$OUT/metrics.json" 2>&1 | tee "$OUT/compare.txt"
