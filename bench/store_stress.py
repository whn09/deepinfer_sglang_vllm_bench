"""Reproduce L3 TRANSFER_FAIL under load (#20): one Store client, a put thread (4-page multi-buffer objects) and
concurrent gets from the main thread, with and without a lock around Store calls."""
import sys, threading, time, numpy as np
from mooncake.store import MooncakeDistributedStore
mode = sys.argv[1]  # nolock | lock
s = MooncakeDistributedStore()
s.setup({"local_hostname": "127.0.0.1", "metadata_server": "http://127.0.0.1:18080/metadata", "global_segment_size": 0,
         "local_buffer_size": 16 << 20, "protocol": "tcp", "master_server_addr": "127.0.0.1:50551"})
page, G, slots = 114048, 4, 4096
pool = np.random.randint(0, 255, size=slots * page, dtype=np.uint8); s.register_buffer(pool.ctypes.data, pool.nbytes)
dst = np.zeros(64 * G * page, dtype=np.uint8); s.register_buffer(dst.ctypes.data, dst.nbytes)
lock = threading.Lock() if mode == "lock" else None
def call(f, *a):
    if lock:
        with lock: return f(*a)
    return f(*a)
stats = {"put_ok": 0, "put_fail": 0, "get_ok": 0, "get_fail": 0}
stop = time.time() + 40
def putter(tid):
    i = 0
    while time.time() < stop:
        keys = [f"stress/{mode}/{tid}/{i + k}" for k in range(16)]
        ptrs = [[pool.ctypes.data + ((i + k) * G + j) % slots * page for j in range(G)] for k in range(16)]
        rcs = call(s.batch_put_from_multi_buffers, keys, ptrs, [[page] * G] * 16)
        stats["put_ok"] += sum(1 for r in rcs if r == 0); stats["put_fail"] += sum(1 for r in rcs if r not in (0, -705)); i += 16
for t in range(1):
    threading.Thread(target=putter, args=(t,), daemon=True).start()
time.sleep(2); j = 0
while time.time() < stop:
    keys = [f"stress/{mode}/0/{j + k}" for k in range(8)]
    ex = call(s.batch_is_exist, keys)
    have = [k for k, e in zip(keys, ex) if e == 1]
    if have:
        rcs = call(s.batch_get_into_multi_buffers, have, [[dst.ctypes.data + (n * G + q) * page for q in range(G)] for n in range(len(have))], [[page] * G] * len(have))
        stats["get_ok"] += sum(1 for r in rcs if r == G * page); stats["get_fail"] += sum(1 for r in rcs if r != G * page)
    j += 8
print(mode, stats, flush=True)
s.remove_by_regex(f"stress/{mode}/.*", True)
