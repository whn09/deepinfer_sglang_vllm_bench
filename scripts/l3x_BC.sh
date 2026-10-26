#!/bin/bash
# Cross-node L3, step BC (on the other host): engine B reuses the remote Store (hit at b); engine C has no L3 and
# --max-num-batched-tokens b. B's cached output must be bit-exact with C's (the #144 chunk-split rule).
MY_IP=${1:?this host private ip}; MASTER_IP=${2:?store host private ip}
W=/opt/dlami/nvme/bench_tmp; cd $W/repo; source scripts/env.sh > /dev/null 2>&1
until grep -q BUILD_L3R_DONE /opt/dlami/nvme/bench_tmp/logs/build_l3r.log 2>/dev/null; do sleep 20; done
L=$W/logs/l3x_BC.log; S=$W/results/l3_xnode; IMG=deepinfer:agentx-l3r-mc
until [ -f $S/state.json ] && bash scripts/wait_gpus_free.sh 0,1,2,3 > /dev/null 2>&1; do sleep 30; done
echo "$(date +%H:%M:%S) engine B (L3 master $MASTER_IP)" > $L
IX_SYNTHETIC_AL=0 DI_IMAGE=$IMG DI_JIT_CACHE=$W/cache/deepinfer_jit DI_EXTRA_EXPERTS=0 DI_L3=1 DI_L3_PROTOCOL=efa DI_L3_DEVICES=rdmap132s0 \
    DI_L3_HOST=$MY_IP DI_L3_MASTER=$MASTER_IP HALF=a bash scripts/serve_deepinfer_agentx.sh >> $L 2>&1 || { echo serve-failed >> $L; exit 1; }
timeout 1200 python3 bench/l3_restart_check.py check --port 8888 --state $S/state.json --doc-tokens 40001 --warmup 8 --dump $S/B.json > $S/B.out 2>&1
cat $S/B.out >> $L
echo "l3 errors: $(docker logs bench-deepinfer 2>&1 | grep -cE 'TRANSFER_FAIL|Store disabled')" >> $L
docker logs bench-deepinfer > $W/logs/l3x_B_engine.log 2>&1; docker rm -f bench-deepinfer > /dev/null 2>&1
B=$(python3 -c "import json; print(int(json.loads(open('$S/B.out').read().strip().splitlines()[-1])['delta_counters_for_cached_request']['hit_tokens']))")
echo "$(date +%H:%M:%S) resume point b=$B; engine C (no L3, chunk b)" >> $L
[ "$B" -gt 0 ] || { echo "no L3 hit" >> $L; exit 1; }
IX_SYNTHETIC_AL=0 DI_IMAGE=$IMG DI_JIT_CACHE=$W/cache/deepinfer_jit DI_EXTRA_EXPERTS=0 DI_EXTRA="--max-num-batched-tokens $B" HALF=a \
    bash scripts/serve_deepinfer_agentx.sh >> $L 2>&1 || { echo serve-failed >> $L; exit 1; }
timeout 1200 python3 bench/l3_restart_check.py check --port 8888 --state $S/state.json --doc-tokens 40001 --warmup 8 --dump $S/C.json >> $L 2>&1
docker rm -f bench-deepinfer > /dev/null 2>&1
python3 -c "
import json; b=json.load(open('$S/B.json')); c=json.load(open('$S/C.json'))
print('B cached (L3 hit from the other host, at b) == C (no cache, chunk b):', b['cached'] == c['cached'], '| == C uncached:', b['cached'] == c['uncached'])
print('B uncached (default chunk) == B cached:', b['uncached'] == b['cached'])" >> $L 2>&1
echo "$(date +%H:%M:%S) L3X_BC_DONE" >> $L
