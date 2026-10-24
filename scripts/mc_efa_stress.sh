#!/bin/bash
# W6 (#21): Mooncake Store over EFA (main build), CPU-only multi-process stress, no GPUs used.
WORK=/opt/dlami/nvme/bench_tmp; IMG=deepinfer:agentx-l3-mcmain; R=$WORK/repo; OUT=$WORK/logs/mc_efa_test.log
docker rm -f bench-mc-store-efa > /dev/null 2>&1
docker run -d --name bench-mc-store-efa --privileged --gpus all --network host --ipc=host --pid=host --shm-size 32g --ulimit memlock=-1 \
    --entrypoint bash $IMG -c "
    /opt/venv/bin/mooncake_master -rpc_port 50561 -enable_http_metadata_server -http_metadata_server_port 18090 -metrics_port 19013 &
    sleep 3
    exec /opt/venv/bin/mooncake_client -master_server_address 127.0.0.1:50561 -metadata_server http://127.0.0.1:18090/metadata \
        -global_segment_size '100 GB' -protocol efa -device_names rdmap113s0 -host 127.0.0.1 -port 50562" > /dev/null
for i in $(seq 60); do docker logs bench-mc-store-efa 2>&1 | grep -q "Starting real client service" && break; sleep 2; done
docker logs bench-mc-store-efa 2>&1 | grep -iE "error|efa|ready|Starting real" | head -8 > $OUT
sed -e 's/127.0.0.1:18080/127.0.0.1:18090/; s/127.0.0.1:50551/127.0.0.1:50561/' $R/bench/store_stress_mp.py > $WORK/store_stress_mp_efa.py
for b in 16 320; do
    timeout 600 docker run --rm --privileged --gpus all --network host --ipc=host --ulimit memlock=-1 -v $WORK:/w --entrypoint /opt/venv/bin/python $IMG \
        /w/store_stress_mp_efa.py --procs 4 --seconds 60 --batch $b --protocol efa --device rdmap113s0 2>&1 | grep -E "^\{|TRANSFER_FAIL|queue-full|Failed" | sort | uniq -c | tail -4 >> $OUT
done
docker rm -f bench-mc-store-efa > /dev/null 2>&1; rm -f $WORK/store_stress_mp_efa.py
echo MC_EFA_DONE >> $OUT
