#!/bin/bash
# Mooncake Store for the HostPrefixCache L3 tier (#20): master (RPC 50551, HTTP metadata 18080) + one standalone client
# that contributes MC_SEGMENT of host DRAM to the pool. --gpus all only provides libcuda (the cuda13 wheel links it); no GPU memory is used. Engines connect with DI_L3=1 (scripts/serve_deepinfer_agentx.sh).
#   bash scripts/serve_mooncake_store.sh [start|stop]     env: MC_SEGMENT (default "400 GB"), MC_PROTOCOL (tcp|efa, default tcp),
#   MC_DEVICES (efa: comma-separated EFA devices, default rdmap113s0), MC_HOST (address remote clients reach this segment at; cross-node: the private IP). efa needs the Mooncake main build
#   (DI_IMAGE=deepinfer:agentx-l3-mcmain); 0.3.13 returns TRANSFER_FAIL. --privileged gives the container /dev/infiniband.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=bench-mc-store
case "${1:-start}" in
    stop) docker rm -f "$NAME" > /dev/null 2>&1 || true; exit 0 ;;
esac
docker rm -f "$NAME" > /dev/null 2>&1 || true
docker run -d --name "$NAME" --privileged --ulimit memlock=-1 --gpus all --network host --ipc=host --pid=host --shm-size 32g --entrypoint bash "${DI_IMAGE:-deepinfer:agentx-l3}" -c "
    /opt/venv/bin/mooncake_master -rpc_port 50551 -enable_http_metadata_server -http_metadata_server_port 18080 -metrics_port 19003 &
    sleep 3
    exec /opt/venv/bin/mooncake_client -master_server_address 127.0.0.1:50551 -metadata_server http://127.0.0.1:18080/metadata \
        -global_segment_size '${MC_SEGMENT:-400 GB}' -protocol ${MC_PROTOCOL:-tcp} $( [ "${MC_PROTOCOL:-tcp}" = efa ] && echo "-device_names ${MC_DEVICES:-rdmap113s0}" ) -host ${MC_HOST:-127.0.0.1} -port 50552" > /dev/null
for i in $(seq 60); do
    docker logs "$NAME" 2>&1 | grep -q "Starting real client service" && { echo "mooncake store ready (segment ${MC_SEGMENT:-400 GB})"; exit 0; }
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ] && { docker logs --tail 30 "$NAME" >&2; exit 1; }
    sleep 2
done
echo "mooncake store not ready" >&2; exit 1
