#!/bin/bash
# Cross-node L3, step A (on the Store host): Store (master + 100 GB EFA segment) + engine A populates turn 1.
# Leaves the Store running for step BC on the other host.
MY_IP=${1:?this host private ip}
W=/opt/dlami/nvme/bench_tmp; cd $W/repo; source scripts/env.sh > /dev/null 2>&1
until grep -q BUILD_L3R_DONE /opt/dlami/nvme/bench_tmp/logs/build_l3r.log 2>/dev/null; do sleep 20; done
L=$W/logs/l3x_A.log; S=$W/results/l3_xnode; mkdir -p $S; IMG=deepinfer:agentx-l3r-mc
echo "$(date +%H:%M:%S) store" > $L
DI_IMAGE=$IMG MC_HOST=$MY_IP MC_SEGMENT="100 GB" MC_PROTOCOL=efa MC_DEVICES=rdmap113s0 bash scripts/serve_mooncake_store.sh start >> $L 2>&1 || { echo store-failed >> $L; exit 1; }
echo "$(date +%H:%M:%S) engine A" >> $L
IX_SYNTHETIC_AL=0 DI_IMAGE=$IMG DI_JIT_CACHE=$W/cache/deepinfer_jit DI_EXTRA_EXPERTS=0 DI_L3=1 DI_L3_PROTOCOL=efa DI_L3_DEVICES=rdmap132s0 \
    DI_L3_HOST=$MY_IP DI_L3_MASTER=$MY_IP HALF=a bash scripts/serve_deepinfer_agentx.sh >> $L 2>&1 || { echo serve-failed >> $L; exit 1; }
timeout 1200 python3 bench/l3_restart_check.py populate --port 8888 --state $S/state.json --doc-tokens 40001 --warmup 8 >> $L 2>&1
sleep 10   # let the write-behind puts finish
curl -s http://127.0.0.1:8888/deepinfer/metrics | grep -E "hicache:l3_put" | awk '{split($1,a,"{"); s[a[1]]+=$2} END {for (k in s) printf "%s %.0f\n", k, s[k]}' >> $L
docker logs bench-deepinfer > $W/logs/l3x_A_engine.log 2>&1; docker rm -f bench-deepinfer > /dev/null 2>&1
echo "$(date +%H:%M:%S) L3X_A_DONE" >> $L
