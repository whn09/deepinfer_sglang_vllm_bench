#!/bin/bash
# 用 InferenceX 的 AgentX（agentx-harness）压一个已经起好的 sglang / vllm server。
#   bash scripts/run_agentx.sh <sglang|vllm> <config名> <concurrency...>
# 例：
#   bash scripts/serve_sglang.sh ref
#   bash scripts/run_agentx.sh sglang sglang-ref 8 32
#
# 只适用于 sglang / vllm：AgentX 需要 OpenAI chat 服务端 + 跨请求 KV 前缀复用，
# DeepInfer 开源版两样都没有（见 docs/agentx.md）。
#
# 每个并发点默认 profiling 3600s + 最长 1800s 预热，单点就要 1~1.5 小时。
# AGENTX_EXTRA_ARGS 可以覆盖默认值（如 --benchmark-duration 600 做快速冒烟，但那样结果不是有效提交）。
set -euo pipefail
source "$(dirname "$0")/env.sh"

BACKEND="${1:?backend}"; CONFIG="${2:?config}"; shift 2
CONCS=("$@")
[ "${#CONCS[@]}" -gt 0 ] || { echo "need at least one concurrency" >&2; exit 1; }
case "$BACKEND" in
    sglang) URL="http://127.0.0.1:${SGLANG_PORT:-30000}" ;;
    vllm)   URL="http://127.0.0.1:${VLLM_PORT:-8000}" ;;
    *) echo "unknown backend" >&2; exit 1 ;;
esac

# 钉日期版本的语料，保证可复现、能和 InferenceX 公布的数字对上
DATASET="${AGENTX_DATASET:-semianalysis_cc_traces_weka_062126}"
VERSION="${AGENTX_VERSION:-1.0.7}"

# agentx-harness 带的是 fork 版 aiperf，必须装在独立 venv 里，别和别的环境混
VENV="$WORK/agentx-venv"
if [ ! -x "$VENV/bin/agentx" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q "agentx-harness==$VERSION"
fi

for C in "${CONCS[@]}"; do
    OUT="$RESULTS_DIR/agentx/$CONFIG/c$C"
    mkdir -p "$OUT"
    "$VENV/bin/agentx" profile \
        --url "$URL" --endpoint /v1/chat/completions \
        --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_DIR" \
        --tokenizer-trust-remote-code \
        --concurrency "$C" \
        --public-dataset "$DATASET" \
        --output-artifact-dir "$OUT" \
        --ui simple \
        ${AGENTX_EXTRA_ARGS:-} \
        2>&1 | tee "$OUT/agentx.log"
    python3 - "$OUT" <<'EOF'
import json, sys, pathlib
p = next(pathlib.Path(sys.argv[1]).rglob("profile_export_aiperf.json"), None)
if p is None:
    print("no profile_export_aiperf.json found"); sys.exit(0)
d = json.loads(p.read_text())
print(f"{p}: submission_valid={d.get('submission_valid')}")
EOF
done
