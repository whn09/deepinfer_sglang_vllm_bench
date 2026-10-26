#!/bin/bash
# (Run on the laptop.) Cross-node L3 correctness: a prefix stored into the Mooncake Store by an engine on host A is
# reused by a fresh engine on host B over EFA, and the result is bit-exact with an uncached prefill chunked at the
# resume point (the chunk-split rule from #144). Real verification (AL 0) so outputs are deterministic.
#   1. Store (master + 100 GB segment, EFA) on HOST_A
#   2. engine A on HOST_A GPUs 0-3, L3 on: populate turn 1
#   3. engine B on HOST_B GPUs 0-3, L3 on (master on HOST_A), empty host pool: turn 2 with the cache -> hits from L3 at b
#   4. engine C on HOST_B GPUs 0-3, no L3, --max-num-batched-tokens b: turn 2 -> must equal B's cached output
#   bash scripts/l3_xnode.sh
set -euo pipefail
A="${HOST_A:-B200-1}"; B="${HOST_B:-B200-2}"; AIP="${A_IP:-172.31.32.216}"; BIP="${B_IP:-172.31.37.169}"
IMG="${DI_IMAGE:-deepinfer:agentx-l3-mcmain}"; W=/opt/dlami/nvme/bench_tmp; S=$W/results/l3_xnode
DOC="${DOC_TOKENS:-40001}"
run() { ssh "$1" "cd $W/repo && source scripts/env.sh > /dev/null 2>&1 && $2" 2> /dev/null; }
L3ENV="IX_SYNTHETIC_AL=0 DI_IMAGE=$IMG DI_JIT_CACHE=$W/cache/deepinfer_jit DI_EXTRA_EXPERTS=0 DI_L3=1 DI_L3_PROTOCOL=efa DI_L3_DEVICES=rdmap132s0 DI_L3_MASTER=$AIP"

echo "$(date +%H:%M:%S) store on $A ($AIP)"
run "$A" "DI_IMAGE=$IMG MC_HOST=$AIP MC_SEGMENT='100 GB' MC_PROTOCOL=efa MC_DEVICES=rdmap113s0 bash scripts/serve_mooncake_store.sh start"
echo "$(date +%H:%M:%S) engine A on $A: populate"
run "$A" "$L3ENV DI_L3_HOST=$AIP HALF=a bash scripts/serve_deepinfer_agentx.sh > $W/logs/l3x_A.log 2>&1 && mkdir -p $S && \
    python3 bench/l3_restart_check.py populate --port 8888 --state $S/state.json --doc-tokens $DOC --warmup 8; docker rm -f bench-deepinfer > /dev/null"
scp -q "$A:$S/state.json" /tmp/l3x_state.json 2> /dev/null && ssh "$B" "mkdir -p $S" 2> /dev/null && scp -q /tmp/l3x_state.json "$B:$S/state.json" 2> /dev/null
echo "$(date +%H:%M:%S) engine B on $B: hit from the remote Store"
OUT_B=$(run "$B" "$L3ENV DI_L3_HOST=$BIP HALF=a bash scripts/serve_deepinfer_agentx.sh > $W/logs/l3x_B.log 2>&1 && \
    python3 bench/l3_restart_check.py check --port 8888 --state $S/state.json --doc-tokens $DOC --warmup 8 --dump $S/B.json; \
    docker logs bench-deepinfer 2>&1 | grep -cE 'TRANSFER_FAIL|Store disabled' | sed 's/^/l3 errors: /'; docker rm -f bench-deepinfer > /dev/null")
echo "$OUT_B"
HIT=$(echo "$OUT_B" | grep '"phase": "check"' | python3 -c "import json,sys; print(int(json.loads(sys.stdin.read())['delta_counters_for_cached_request']['hit_tokens']))")
echo "$(date +%H:%M:%S) resume point b=$HIT; engine C on $B: no L3, chunk b"
run "$B" "IX_SYNTHETIC_AL=0 DI_IMAGE=$IMG DI_JIT_CACHE=$W/cache/deepinfer_jit DI_EXTRA_EXPERTS=0 DI_EXTRA='--max-num-batched-tokens $HIT' HALF=a \
    bash scripts/serve_deepinfer_agentx.sh > $W/logs/l3x_C.log 2>&1 && \
    python3 bench/l3_restart_check.py check --port 8888 --state $S/state.json --doc-tokens $DOC --warmup 8 --dump $S/C.json; docker rm -f bench-deepinfer > /dev/null"
run "$A" "bash scripts/serve_mooncake_store.sh stop"
run "$B" "python3 -c \"
import json; b=json.load(open('$S/B.json')); c=json.load(open('$S/C.json'))
print('B cached (L3 hit from the other host, at b) == C (no cache, chunk b):', b['cached'] == c['cached'], '| == C uncached:', b['cached'] == c['uncached'])
print('B uncached (default chunk) == B cached:', b['uncached'] == b['cached'])\""
