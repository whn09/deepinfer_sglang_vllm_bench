#!/bin/bash
# 按 InferenceX 的 B200 AgentX recipe 起一个并发点的 server（+ recipe 里要求的 router）（#4）。
# 参数不手抄：bench/ix_recipe.py 直接展开 InferenceX 仓库里的 agentic.yaml。
#   bash scripts/serve_ix.sh <vllm|sglang> <conc> [topology]
# 例：bash scripts/serve_ix.sh sglang 128          # override_dep4_c128：DP attention + HiCache + sglang-router
#     bash scripts/serve_ix.sh vllm 128 dep4       # zip_override_dep4（同一并发有 tp4/dep4 两个变体时要指定）
#
# 环境变量：
#   IX_RECIPES       InferenceX recipe 目录（默认 $WORK/inferencex-e2e/recipes/dsv41flash）
#   IX_SYNTHETIC_AL  golden 接受长度（默认 3.51 = DSpark 5 thinking on；设 0 用真实验证）
#   GPUS_OVERRIDE / SGLANG_PORT / VLLM_PORT / CONTAINER_SUFFIX  同 serve_*.sh
#   IX_EXTRA_ARGS    追加到 recipe 的 server 参数之后（在公开配置上做 A/B）
# 成功后把 AgentX 应该连的地址写到 $WORK/logs/ix_endpoint<suffix>（router 存在时是 router 的地址）。
set -euo pipefail
source "$(dirname "$0")/env.sh"

FW="${1:?usage: serve_ix.sh <vllm|sglang> <conc> [topology]}"; C="${2:?conc}"; TOPO="${3:-}"
RECIPES="${IX_RECIPES:-$WORK/inferencex-e2e/recipes/dsv41flash}"
RECIPE="$RECIPES/$FW/b200-fp4-mtp/agentic.yaml"
AL="${IX_SYNTHETIC_AL:-3.51}"
SUF="${CONTAINER_SUFFIX:-}"
NAME="bench-$FW$SUF"
case "$FW" in
    vllm)   PORT="${VLLM_PORT:-8000}";   IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0}" ;;
    sglang) PORT="${SGLANG_PORT:-30000}"; IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.22-cu130}" ;;
    *) echo "unknown framework $FW" >&2; exit 1 ;;
esac

J="$WORK/logs/ix_${FW}_c${C}${TOPO:+_$TOPO}$SUF.json"
SYN=(); [ "$AL" != "0" ] && SYN=(--synthetic-al "$AL")
python "$BENCH_ROOT/bench/ix_recipe.py" "$RECIPE" --conc "$C" ${TOPO:+--topology "$TOPO"} --engine "$FW" "${SYN[@]}" --format json > "$J"
mapfile -t ARGS < <(python "$BENCH_ROOT/bench/ix_recipe.py" "$RECIPE" --conc "$C" ${TOPO:+--topology "$TOPO"} --engine "$FW" "${SYN[@]}" --format cli)
mapfile -t ENVS < <(python "$BENCH_ROOT/bench/ix_recipe.py" "$RECIPE" --conc "$C" ${TOPO:+--topology "$TOPO"} --engine "$FW" "${SYN[@]}" --format env)
# IX_EXTRA_ARGS: appended to the recipe's server args (A/B on top of the published config, e.g. "--tokenizer-backend fastokens")
[ -n "${IX_EXTRA_ARGS:-}" ] && read -ra _EXTRA <<< "$IX_EXTRA_ARGS" && ARGS+=("${_EXTRA[@]}")
read -r VARIANT NGPU FRONT < <(python -c "import json;d=json.load(open('$J'));print(d['variant'],d['gpus'],d['frontend'].get('type','none'))")

GPUS="${GPUS_OVERRIDE:-$(seq -s, 0 $((NGPU - 1)))}"
ENV_ARGS=(); for e in "${ENVS[@]}"; do ENV_ARGS+=(-e "$e"); done
echo "ix $FW c=$C variant=$VARIANT gpus=$GPUS ($NGPU) frontend=$FRONT synthetic_al=$AL image=$IMAGE"
if [ "${DRY_RUN:-0}" = 1 ]; then printf '  %s\n' "${ARGS[@]}"; printf '  env %s\n' "${ENVS[@]}"; exit 0; fi

bash "$BENCH_ROOT/scripts/wait_gpus_free.sh" "$GPUS" || exit 1
mkdir -p "$WORK/cache/$FW"
docker rm -f "$NAME" > /dev/null 2>&1 || true
COMMON=(--ipc=host --network host --shm-size 32g
        -v "$MODEL_DIR":/model:ro -v "$WORK/cache/$FW":/root/.cache -v "$HF_HOME":/root/.cache/huggingface
        "${ENV_ARGS[@]}")
if [ "$FW" = vllm ]; then
    # 官方 vLLM 命令带 --privileged：选卡只能靠 CUDA_VISIBLE_DEVICES（物理编号）
    DP_ARGS=(); [[ " ${ARGS[*]} " == *" --data-parallel-size "* ]] && DP_ARGS=(--data-parallel-rpc-port $((29550 + (PORT - 8000) * 100)))
    docker run -d --name "$NAME" --gpus all --privileged -e CUDA_VISIBLE_DEVICES="$GPUS" "${COMMON[@]}" \
        "$IMAGE" /model --served-model-name "$SERVED_MODEL_NAME" "${ARGS[@]}" "${DP_ARGS[@]}" \
        --host 0.0.0.0 --port "$PORT" > /dev/null
else
    docker run -d --name "$NAME" --gpus "\"device=$GPUS\"" "${COMMON[@]}" \
        "$IMAGE" sglang serve --model-path /model --served-model-name "$SERVED_MODEL_NAME" "${ARGS[@]}" \
        --host 0.0.0.0 --port "$PORT" --dist-init-addr "127.0.0.1:$((PORT + 234))" > /dev/null
fi
bash "$BENCH_ROOT/scripts/wait_ready.sh" "http://127.0.0.1:$PORT" "$NAME" "${IX_READY_TIMEOUT:-7200}"
ENDPOINT="http://127.0.0.1:$PORT"

# router（recipe 的 frontend 段）。router 端口 = server 端口 + 1000
RPORT=$((PORT + 1000))
case "$FRONT" in
    sglang-router)
        mapfile -t RARGS < <(python -c "
import json
d=json.load(open('$J'))['frontend'].get('args') or {}
for k,v in d.items():
    if v is False: continue
    print('--'+k)
    if v is not True: print(v)")
        # 两个 router 同机时默认 Prometheus 端口会冲突（Address already in use）；第二个实例另给一个
        [ -n "$SUF" ] && RARGS+=(--prometheus-port $((RPORT + 29)))
        docker rm -f "bench-router$SUF" > /dev/null 2>&1 || true
        docker run -d --name "bench-router$SUF" --network host "$IMAGE" \
            python3 -m sglang_router.launch_router --worker-urls "http://127.0.0.1:$PORT" \
            --host 0.0.0.0 --port "$RPORT" "${RARGS[@]}" > /dev/null
        ENDPOINT="http://127.0.0.1:$RPORT"
        bash "$BENCH_ROOT/scripts/wait_ready.sh" "$ENDPOINT" "bench-router$SUF" 600
        ;;
    vllm-router)
        echo "vllm-router frontends (DEP arms) are not wired up yet" >&2; exit 1 ;;
esac
echo "$ENDPOINT" > "$WORK/logs/ix_endpoint$SUF"
echo "ENDPOINT=$ENDPOINT"
