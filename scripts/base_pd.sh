#!/bin/bash
# Single-node PD baseline (#214 rebased, TCP, affinity, AL 0) on this host's 8 GPUs; logs kept for debugging.
W=/opt/dlami/nvme/bench_tmp; cd $W/repo; source scripts/env.sh > /dev/null 2>&1
L=$W/logs/base_pd.log; mkdir -p $W/results/pd_xnode
echo "$(date +%H:%M:%S) serve" > $L
IX_SYNTHETIC_AL=0 DI_ROUTING=affinity DI_IMAGE=deepinfer:pd214-mcmain bash scripts/serve_deepinfer_pd.sh >> $L 2>&1 || { echo serve-failed >> $L; exit 1; }
echo "$(date +%H:%M:%S) parity" >> $L
timeout 1200 python3 bench/pd_parity.py --port 8888 --out $W/results/pd_xnode/base_tcp_1node.json >> $L 2>&1; echo "parity rc=$?" >> $L
docker logs bench-di-pe > $W/logs/base_pd_pe.log 2>&1; docker logs bench-di-ce > $W/logs/base_pd_ce.log 2>&1
docker rm -f bench-di-pe bench-di-ce > /dev/null 2>&1
echo "$(date +%H:%M:%S) BASE_PD_DONE" >> $L
