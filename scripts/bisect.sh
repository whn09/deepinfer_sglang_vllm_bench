#!/bin/bash
# git bisect run step: build the Mooncake client layer at HEAD and fail if the 4-proc EFA probe sees any put/get failure.
W=/opt/dlami/nvme/bench_tmp; SRC=$W/Mooncake-bisect; IMG=deepinfer:agentx-l3-mc-bisect
cd $W/repo && docker build -q -f docker/Dockerfile.mooncake-main --build-arg BASE=deepinfer:agentx-l3 --build-context mooncake=$SRC -t $IMG . > /dev/null 2>> $W/logs/bisect_build.log || exit 125
sed -e 's/"local_hostname": "127.0.0.1"/"local_hostname": "172.31.37.169"/' $W/repo/bench/store_stress_mp.py > $W/bisect_stress.py
out=$(timeout 300 docker run --rm --privileged --gpus all --network host --ipc=host --ulimit memlock=-1 -v $W:/w --entrypoint /opt/venv/bin/python $IMG \
    /w/bisect_stress.py --procs 4 --seconds 30 --batch 16 --protocol efa --device rdmap132s0 2>/dev/null | grep '^{')
echo "$(git -C $SRC log -1 --format=%h) $out" >> $W/logs/bisect.log
echo "$out" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); sys.exit(0 if d['put_fail'] + d['get_fail'] == 0 else 1)"
