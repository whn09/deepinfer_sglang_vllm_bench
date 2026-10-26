"""Multi-process version of store_stress.py (#20): N processes, one Store client each (global_segment_size 0, the segment
comes from the standalone mooncake_client), as with the 4 DP engines of the AgentX L3 arm. Each process runs a put
thread (G-page multi-buffer objects, like _l3_submit) and, from its main thread, reads keys the NEXT process wrote
(cross-client hits, like an L3 hit after the router moved a conversation).

    python bench/store_stress_mp.py --procs 4 --seconds 60 --protocol tcp --batch 96
"""
import argparse, json, multiprocessing as mp, threading, time
import numpy as np

PAGE, G, SLOTS = 114048, 4, 4096


def worker(rank: int, a, out) -> None:
    BATCH = a.batch
    from mooncake.store import MooncakeDistributedStore
    s = MooncakeDistributedStore()
    rc = s.setup({"local_hostname": "127.0.0.1", "metadata_server": "http://127.0.0.1:18080/metadata",
                  "global_segment_size": 0, "local_buffer_size": 16 << 20, "protocol": a.protocol,
                  "rdma_devices": a.device, "master_server_addr": "127.0.0.1:50551"})
    assert rc == 0, f"setup rc={rc}"
    pool = np.random.randint(0, 255, size=SLOTS * PAGE, dtype=np.uint8); s.register_buffer(pool.ctypes.data, pool.nbytes)
    dst = np.zeros(BATCH * G * PAGE, dtype=np.uint8); s.register_buffer(dst.ctypes.data, dst.nbytes)
    st = {"put_ok": 0, "put_fail": 0, "get_ok": 0, "get_fail": 0, "get_bad_bytes": 0, "rcs": {}}
    stop = time.time() + a.seconds
    tag = f"{a.tag}/{rank}"

    def putter():
        i = 0
        while time.time() < stop:
            keys = [f"{tag}/{i + k}" for k in range(BATCH)]
            ptrs = [[pool.ctypes.data + ((i + k) * G + j) % SLOTS * PAGE for j in range(G)] for k in range(BATCH)]
            for r in s.batch_put_from_multi_buffers(keys, ptrs, [[PAGE] * G] * BATCH):
                if r in (0, -705): st["put_ok"] += 1
                else: st["put_fail"] += 1; st["rcs"][str(r)] = st["rcs"].get(str(r), 0) + 1
            i += BATCH

    threading.Thread(target=putter, daemon=True).start()
    time.sleep(3)
    peer, j = f"{a.tag}/{(rank + 1) % a.procs}", 0
    while time.time() < stop:
        keys = [f"{peer}/{j + k}" for k in range(BATCH)]
        have = [(j + n, k) for n, (k, e) in enumerate(zip(keys, s.batch_is_exist(keys))) if e == 1]
        if not have:
            time.sleep(0.01); continue
        rcs = s.batch_get_into_multi_buffers([k for _, k in have],
                                             [[dst.ctypes.data + (n * G + q) * PAGE for q in range(G)] for n in range(len(have))],
                                             [[PAGE] * G] * len(have))
        for n, ((idx, _), r) in enumerate(zip(have, rcs)):
            if r != G * PAGE:
                st["get_fail"] += 1; st["rcs"][str(r)] = st["rcs"].get(str(r), 0) + 1; continue
            st["get_ok"] += 1
            # the peer wrote pages (idx*G + q) % SLOTS of its own random pool: check against a byte we can't know, so
            # only check that the G pages are not all zero (a silently skipped copy leaves dst untouched = zeros)
            if not dst[n * G * PAGE:(n + 1) * G * PAGE:4096].any(): st["get_bad_bytes"] += 1
        dst[:] = 0
        j += BATCH
    time.sleep(1)
    out.put((rank, st))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--protocol", default="tcp")
    p.add_argument("--device", default="")
    p.add_argument("--batch", type=int, default=16, help="objects per put/get call (one AgentX request stores ~90)")
    p.add_argument("--tag", default=f"mp{int(time.time())}")
    a = p.parse_args()
    q = mp.get_context("spawn").Queue()
    ps = [mp.get_context("spawn").Process(target=worker, args=(r, a, q)) for r in range(a.procs)]
    [x.start() for x in ps]
    res = dict(q.get(timeout=a.seconds + 300) for _ in ps)
    [x.join() for x in ps]
    tot = {k: sum(r[k] for r in res.values()) for k in ("put_ok", "put_fail", "get_ok", "get_fail", "get_bad_bytes")}
    rcs = {}
    for r in res.values():
        for k, v in r["rcs"].items(): rcs[k] = rcs.get(k, 0) + v
    print(json.dumps({"protocol": a.protocol, "procs": a.procs, "batch": a.batch, "seconds": a.seconds, **tot, "rcs": rcs,
                      "put_GBps": round(tot["put_ok"] * G * PAGE / a.seconds / 1e9, 2),
                      "get_GBps": round(tot["get_ok"] * G * PAGE / a.seconds / 1e9, 2)}))


if __name__ == "__main__":
    main()
