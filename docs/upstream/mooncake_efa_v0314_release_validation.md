# Mooncake v0.3.14-pre over EFA: release validation and PR #4956 review

Date: 2026-10-26/27. Hosts: 2× p6-b200.48xlarge in ap-south-2b, `B200-1` = 172.31.32.216 and `B200-2` = 172.31.37.169 (ssh aliases through `Jump`). Each has 8× B200 and 8 EFA NICs. libfabric 2.6.0amzn1.0, driver 595.91. The two hosts are not co-located (they share only the top 2 of 4 network-node layers), so bandwidth numbers are a lower bound.

## 1. What was tested

- Release wheel: `mooncake-transfer-engine` 0.3.14rc0, cp312, cu12 (artifact `mooncake-wheel-pre-release-py312` of run [40298130741](https://github.com/kvcache-ai/Mooncake/actions/runs/40298130741), tag `v0.3.14-pre`, commit b46254de).
- Installed in `/opt/dlami/nvme/mcrel/venv312`: a uv venv on system python3.12 with torch 2.11.0+cu128.
- The wheel needs `libcudart.so.12`. Either `import torch` (cu12) first, or put `venv312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib` on `LD_LIBRARY_PATH`. The wheel does not declare a CUDA runtime dependency, so `import mooncake` alone fails on a CUDA 13 host.

## 2. Release wheel results

**OK**
- Static checks:
  - `engine.so`, `_store`, `_ep` and `libmooncake_pg` contain both the EFA and the RDMA transport.
  - No `libfabric` in NEEDED; it is dlopened at runtime from `/opt/amazon/efa/lib/libfabric.so.1.32.0`, as seen in `/proc/self/maps`.
  - The wheel bundles no libfabric or libibverbs.
- Functional: `bench/mc_efa_release_check.py` (Python API, `transfer_sync_*` and `batch_transfer_sync_*`), GPU (`cudaMalloc`) and CPU memory, 4 KiB to 1 GiB plus two batch shapes: every byte checked on both sides, 0 failures.

**FAILS**
- `transfer_engine_bench` cross-host fails in every configuration tried: 8 GPUs or 1 GPU, t16 b128 or t4 b32, read or write. The initiator logs the following within about 20 ms of the first traffic, then `FAILED`:

  ```
  EFA submitSlicesOnPeer: 1024 consecutive FI_EAGAIN waves posted nothing on ... (fi_addr=1);
  wr_depth=16/2048; ... last_completion=-1ms ago; ... totals posted=16 completed=0 ...; failing 8 slice(s)
  ```

  One run also printed `free(): corrupted unsorted chunks` on the failure path. Not investigated.
- Baseline on the same hosts with the same flags: PyPI `mooncake-transfer-engine-efa==0.3.13.post1`.
  - 1 GPU read: 57.0 GB/s.
  - 8 GPU write (block 8M, buffer 16G, t16 b128): 308.4 GB/s.
  - 8 GPU read (block 1M, buffer 4G): 265.7 GB/s.
  - 0 failures.

**Cause.** `kMaxStalledWaves = 1024` in `EfaContext::submitSlicesOnPeer` came in with #2063 (487da6af, merged 2026-10-25). It fails the slices after 1024 consecutive FI_EAGAIN waves that posted nothing. On a fresh peer the first operations have not completed yet, and 1024 waves pass in milliseconds. PR #4956 addresses exactly this.

## 3. PR #4956 (`whn09:efa-stall-wall-clock`, 70957771)

Built from source on both hosts (`/opt/dlami/nvme/mcrel/src`) with `-DUSE_CUDA=ON -DUSE_EFA=ON`. The bench had one test-only change, described in section 4. Cross-host `transfer_engine_bench`; `MC_EFA_STALL_MIN_MS` was set on the initiator.

| `MC_EFA_STALL_MIN_MS` | 1 GPU, t4 b32, read 1M | 8 GPU, t16 b128 |
|---|---|---|
| 0 (= current main) | FAIL | — |
| **50 (PR default)** | **FAIL, 4/4 runs** | PASS: write (8M) 320.5 GB/s; read (1M) 329.9 / 333.8 / 346.1 / 333.8 GB/s |
| 100 | PASS, 4/4 runs, 54.7–57.1 GB/s | — |
| 200 / 500 | PASS, 57.0 GB/s | — |
| 1000 | PASS, 57.0 GB/s | write 326.3, read 333.7 GB/s |

- At 50 ms, the warning still shows the peer's first operations incomplete: `13879 consecutive FI_EAGAIN waves over 50 ms ... totals posted=16 completed=0`. On a fresh cross-host connection the first completions take 50–100 ms, so 50 ms is too short and 100 ms leaves little margin.
- **Recommendation:** raise the default to 500 ms or more, for example 1000 ms like `MC_EFA_CQ_DRAIN_TIMEOUT_MS`. The PR's own dead-peer measurements show that the total run time hardly depends on this floor.
- No bandwidth regression: 8 GPU is above v0.3.13.post1 on the same pair (write 308 → 320–326, read 266 → 330–346 GB/s).

## 4. Second release issue: `transfer_engine_bench` aborts on deferred cleanup

`mooncake-transfer-engine/example/transfer_engine_bench.cpp`, at the end of the worker loop (line 846 at 70957771):

```cpp
s = engine->freeBatchID(batch_id);
LOG_ASSERT(s.ok());
```

Since #4569, `freeBatchID` may return `BatchCleanupDeferred`. `transfer_engine.h` says that "an OK or BatchCleanupDeferred return invalidates batch_id". On EFA this happens routinely: the CQ thread's completion callback is still unwinding (`active_completion_callbacks != 0`) when the worker sees all tasks COMPLETED. The bench then aborts with `Assert failed: s.ok()` after `Batch ... is still busy; cleanup has been deferred`.

The bench ships in the wheel, so v0.3.14 is affected even once #4956 lands. Fix used for the measurements above:

```cpp
LOG_ASSERT(s.ok() || s.IsBatchCleanupDeferred());
```

No existing PR for this was found.

## 5. Draft review comment for #4956

> Validated on 2× p6-b200.48xlarge (cross-host, libfabric 2.6.0amzn1.0) with `transfer_engine_bench`, both sides built from 70957771. The bench needed a test-only change to accept `BatchCleanupDeferred` from `freeBatchID`; see below.
>
> | `MC_EFA_STALL_MIN_MS` | 1 GPU, t4 b32, read | 8 GPU, t16 b128 |
> |---|---|---|
> | 0 (current main) | FAIL | — |
> | 50 (default) | **FAIL 4/4** | PASS: write 320.5, read 329.9–346.1 GB/s |
> | 100 | PASS 4/4, ~57 GB/s | — |
> | 1000 | PASS, 57.0 GB/s | write 326.3, read 333.7 GB/s |
>
> At 50 ms the warning still shows `completed=0` after 13879 waves: on a fresh cross-host connection the first completions take 50–100 ms. Could the default be raised to 500 ms or more, for example the same 1000 ms as `MC_EFA_CQ_DRAIN_TIMEOUT_MS`? Your dead-peer numbers suggest the cost is small. For reference, v0.3.13.post1 gives 308 / 266 GB/s on the same pair, so this PR shows no regression.
>
> Separately, `transfer_engine_bench` asserts `s.ok()` on `freeBatchID`, which aborts on EFA whenever cleanup is deferred (#4569). That bench ships in the wheel, so it also affects v0.3.14.

## 6. How to reproduce

The scripts are local copies in `/tmp/rel/` on the laptop (`bench_src.sh`, `target_src.sh`, `stop_target.sh`); the target scripts are also on B200-1 under `/opt/dlami/nvme/mcrel/`. The core commands are below. Run the target on B200-1 first, then pass the segment it prints to the initiator on B200-2.

```bash
# both hosts
source /opt/dlami/nvme/mcrel/env_src.sh   # CUDA 13 from /opt/pytorch
B=/opt/dlami/nvme/mcrel/src/build/mooncake-transfer-engine/example/transfer_engine_bench

# B200-1 (target); prints "listening on 172.31.32.216:<port>"
$B --mode=target --protocol=efa --metadata_server=P2PHANDSHAKE \
   --local_server_name=172.31.32.216 --use_vram=true --gpu_id=0 --buffer_size=4294967296

# B200-2 (initiator): fails at the default 50 ms, passes with MC_EFA_STALL_MIN_MS=100 or more
MC_EFA_STALL_MIN_MS=50 $B --mode=initiator --protocol=efa --metadata_server=P2PHANDSHAKE \
   --local_server_name=172.31.37.169 --segment_id=172.31.32.216:<port> --operation=read \
   --use_vram=true --gpu_id=0 --buffer_size=4294967296 --block_size=1048576 \
   --threads=4 --batch_size=32 --duration=10
```

For 8 GPUs, use `--gpu_id=-1 --threads=16 --batch_size=128`; for writes, use `--operation=write --block_size=8388608 --buffer_size=17179869184`.

Harness pitfalls:
- `pkill -f <pattern>` inside an ssh command line matches and kills that ssh's own shell. Use script files on the remote side.
- zsh does not word-split `$var` (use `${=var}`).

## 7. Follow-up (2026-10-27)

- **#4956:** the default is now 1000 ms (ab812fb6). The PR description carries the cross-host table, and the review comment was posted as https://github.com/kvcache-ai/Mooncake/pull/4956#issuecomment-6424013913.
- **Dead-peer cost re-measured at 1000 ms** (same-host Store, segment client killed 15 s into a 60 s run): 121 / 93 s, vs 117 / 105 s at 50 ms and 118 / 93 s count-only. Healthy 60 s run at 1000 ms: 0 failures.
- **`transfer_engine_bench` fix (`s.ok() || s.IsBatchCleanupDeferred()`):** committed on the local branch `bench-free-batch-deferred` (47ff63fc) in `Mooncake-pr`. Not pushed or opened yet.
