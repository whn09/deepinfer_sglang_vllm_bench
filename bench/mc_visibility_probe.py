"""Does nvlink_intra work when each process only sees its own GPU (+ the peer's)? (#18)
   server: CUDA_VISIBLE_DEVICES=<B>   client: CUDA_VISIBLE_DEVICES=<A>[,<B>]"""
import os, sys, time, subprocess, json

N = 64 << 20
role = sys.argv[1]
os.environ["MC_INTRANODE_NVLINK"] = "true"
import torch
from mooncake.engine import TransferEngine
te = TransferEngine()
assert te.initialize("127.0.0.1", "P2PHANDSHAKE", "nvlink_intra", "") == 0
if role == "server":
    buf = torch.zeros(N, dtype=torch.uint8, device="cuda:0")
    assert te.register_memory(buf.data_ptr(), N) == 0
    print(json.dumps({"seg": f"127.0.0.1:{te.get_rpc_port()}", "addr": buf.data_ptr()}), flush=True)
    sys.stdin.readline()
    torch.cuda.synchronize()
    print("SUM", int(buf.sum().item()), flush=True)
else:
    info = json.loads(sys.argv[2])
    src = torch.ones(N, dtype=torch.uint8, device="cuda:0")
    assert te.register_memory(src.data_ptr(), N) == 0
    rc = te.transfer_sync_write(info["seg"], src.data_ptr(), info["addr"], N)
    t = time.perf_counter(); rc2 = te.transfer_sync_write(info["seg"], src.data_ptr(), info["addr"], N); dt = time.perf_counter() - t
    print(f"RC {rc} {rc2} {N/dt/1e9:.0f} GB/s", flush=True)
