"""Mooncake nvlink_intra feasibility probe (#18): process on GPU A writes into a buffer registered by a process on GPU B."""
import os, sys, time, json, multiprocessing as mp

N = 256 << 20

def server(dev, q, done):
    os.environ["MC_INTRANODE_NVLINK"] = "true"
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(dev)
    buf = torch.zeros(N, dtype=torch.uint8, device=f"cuda:{dev}")
    te = TransferEngine()
    assert te.initialize("127.0.0.1", "P2PHANDSHAKE", "nvlink_intra", "") == 0
    assert te.register_memory(buf.data_ptr(), N) == 0
    q.put((f"127.0.0.1:{te.get_rpc_port()}", buf.data_ptr()))
    done.wait()
    torch.cuda.synchronize()
    q.put(int(buf[:1024].sum().item()) + int(buf[-1024:].sum().item()))

def main():
    ctx = mp.get_context("spawn")
    q, done = ctx.Queue(), ctx.Event()
    p = ctx.Process(target=server, args=(4, q, done)); p.start()
    seg, remote = q.get(timeout=120)
    os.environ["MC_INTRANODE_NVLINK"] = "true"
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(0)
    src = torch.ones(N, dtype=torch.uint8, device="cuda:0")
    te = TransferEngine()
    assert te.initialize("127.0.0.1", "P2PHANDSHAKE", "nvlink_intra", "") == 0
    assert te.register_memory(src.data_ptr(), N) == 0
    res = {}
    for size in (114048, 1 << 20, 16 << 20, 107 << 20, N):
        te.transfer_sync_write(seg, src.data_ptr(), remote, size)  # warm
        t = time.perf_counter(); reps = 5
        for _ in range(reps):
            rc = te.transfer_sync_write(seg, src.data_ptr(), remote, size)
            assert rc == 0, rc
        dt = (time.perf_counter() - t) / reps
        res[size] = (dt * 1e3, size / dt / 1e9)
    # 900 blocks x 114 KB as a batch (100K-token prompt)
    nb = 900; bs = 114048
    t = time.perf_counter()
    rc = te.batch_transfer_sync_write(seg, [src.data_ptr() + i * bs for i in range(nb)], [remote + i * bs for i in range(nb)], [bs] * nb)
    dtb = time.perf_counter() - t
    done.set(); chk = q.get(timeout=60); p.join()
    for s, (ms, gbs) in res.items():
        print(f"single write {s/1e6:8.2f} MB: {ms:7.3f} ms  {gbs:6.1f} GB/s")
    print(f"batch 900x114KB (102.6 MB): rc={rc} {dtb*1e3:.2f} ms  {nb*bs/dtb/1e9:.1f} GB/s")
    print("data check (expect 2048):", chk)

if __name__ == "__main__":
    main()
