# Mooncake `nvlink_intra`: support VMM (`cuMemCreate`) buffers — design & task brief

Task brief for an implementing agent. Target repo: [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake), Transfer Engine (`mooncake-transfer-engine`). Related:
- kvcache-ai/Mooncake#2511 ("When enable expandable_segments, mooncake register memory failed", closed without a fix).
- Diagnostics PR from branch `whn09:nvlink-intra-container-hints` (commit `caeecde`, error hints + docs; push pending).

## 1. Goal

`IntraNodeNvlinkTransport` (protocol `nvlink_intra`, enabled with `MC_INTRANODE_NVLINK=true`) can register and transfer only **legacy device allocations** (`cudaMalloc` / `cuMemAlloc`), because it shares them with `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`. Memory from the CUDA virtual memory management API (`cuMemCreate` + `cuMemMap`) has no legacy IPC handle. The main case is PyTorch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is common in serving stacks: DeepInfer sets it unconditionally (`deepinfer/worker/base.py:177`), and vLLM/SGLang users enable it. Today those buffers cannot be registered at all.

**Goal:** register VMM buffers with `nvlink_intra` and transfer into and out of them across processes on one node, with no change for legacy buffers.

**Non-goals:**
- Inter-node: that is the MNNVL `nvlink` transport, which already handles `cuMemCreate` memory with FABRIC handles.
- Making PyTorch allocations shareable when PyTorch itself did not request a shareable handle type (see §3.4).

## 2. Current code (upstream `main` @ 38f50da)

`mooncake-transfer-engine/src/transport/intranode_nvlink_transport/intranode_nvlink_transport.cpp`:
- `registerLocalMemory` (~L626):
  - `cudaPointerGetAttributes` checks the memory type.
  - `cuMemGetAddressRange(addr)` gives the allocation base and size ("cudaMalloc segment granularity").
  - `cudaIpcGetMemHandle(base)` produces the handle, serialized into `BufferDesc.shm_name`.
  - `metadata_->addLocalMemoryBuffer(desc)` publishes it.
  - Bases are de-duplicated in `registered_base_addrs_`.
- `relocateSharedMemoryAddress` (~L726):
  - Called per request on the initiator.
  - Finds the peer `BufferDesc` that contains `dest_addr`.
  - On first use, `cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)` maps it.
  - The mapping is cached in `remap_entries_[(target_id, entry.addr)]` and translated to a local pointer.
- `unregisterLocalMemory` (~L702) and the destructor (`cudaIpcCloseMemHandle` for all `remap_entries_`).
- `submitTransfer` / `submitTransferTask` call `relocateSharedMemoryAddress` and then `cudaMemcpyBatchAsync` (CUDA ≥ 13) or per-slice `cudaMemcpyAsync` on a per-device stream.

**Reference implementation of the VMM import path:** the MNNVL transport `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`.
- `registerLocalMemory` (~L980): for `cuMemCreate` memory, `cuMemRetainAllocationHandle` + `cuMemExportToShareableHandle(..., CU_MEM_HANDLE_TYPE_FABRIC)` (~L1058).
- `relocateSharedMemoryAddress` (~L1319): `cuMemImportFromShareableHandle(FABRIC)` → `cuMemAddressReserve` → `cuMemMap` → `cuMemSetAccess` (~L1370–1408).
- `allocateManagedBuffer` (~L1467): allocates `cuMemCreate` memory with `requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC`.

**Reference for POSIX-FD sharing of PyTorch expandable segments:** `pytorch/pytorch` `c10/cuda/CUDACachingAllocator.cpp`, `struct ExpandableSegment`.
- `map()` (~L585) requests `CU_MEM_HANDLE_TYPE_FABRIC` if fabric access is available, else `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`, **only when IPC is enabled** (`TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=1`).
- `share()` (~L684) writes `{pid, segment_size, num_handles, handle_type}` plus one exported fd (or fabric handle) per `segment_size_` chunk.
- `fromShared()` (~L743) imports via `pidfd_open(pid)` + `pidfd_getfd(fd)` and maps each chunk.

## 3. Facts verified on the target machine

HGX B200 ×8 (`ssh B200-1`, temporary host), driver 595.91, CUDA 13.2, kernel 7.0.0-1014-aws, test container `deepinfer:nvidia-dev` (nvidia/cuda 13.2.1-devel). Probe sources in this repo: `bench/cuda_probes/` (errcheck.cu, fabric_check.cu, vmm_layout.cu), `bench/mc_nvlink_probe.py`, `bench/mc_visibility_probe.py`.

1. `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1` and `..._HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED=1`.
2. **FABRIC handles do not work here without IMEX:** `..._HANDLE_TYPE_FABRIC_SUPPORTED=1`, but `cuMemCreate` with `requestedHandleTypes=FABRIC` returns **800 (`CUDA_ERROR_NOT_PERMITTED`)**, and there is no `/dev/nvidia-caps-imex-channels`. So for single-node HGX the portable path is **POSIX file descriptors**; FABRIC is optional and needs an IMEX channel.
3. `cudaIpcGetMemHandle` on a `cuMemCreate` mapping returns `cudaErrorInvalidValue` ("invalid argument"). `cuMemRetainAllocationHandle(ptr)` returns `CUDA_SUCCESS` only for VMM memory (`CUDA_ERROR_INVALID_VALUE`=1 for `cudaMalloc` memory), so it is a reliable VMM detector. It returns the same handle value that `cuMemCreate` produced, and each call must be balanced with `cuMemRelease`.
4. **For VMM memory, `cuMemGetAddressRange(p)` returns the extent of the single physical allocation mapped at `p`, not the whole reservation.** Test: two 4 MiB allocations mapped back to back in a 12 MiB reservation, the third chunk unmapped.
   - Queries in chunk 0 → `(base=+0, size=4 MiB)`; queries in chunk 1 → `(base=+4 MiB, size=4 MiB)`; in the unmapped tail → `CUDA_ERROR_NOT_FOUND` (500).
   - So chunks can be enumerated by walking `p = base + size`.
   - Consequence for today's code: `registerLocalMemory` on a VMM range would register only the first chunk, even if the IPC handle existed.
5. `cuMemGetAllocationPropertiesFromHandle` reports `requestedHandleTypes` (1 = POSIX FD), so an allocation created without a shareable type can be detected and rejected with a clear message.
6. `/proc/sys/kernel/yama/ptrace_scope = 1` (Ubuntu default). `pidfd_getfd` needs `PTRACE_MODE_ATTACH_REALCREDS`, which under scope 1 is limited to descendants unless the caller has `CAP_SYS_PTRACE`. PyTorch's approach works between parent and child; Mooncake peers are unrelated processes, so **do not rely on `pidfd_getfd`** by default.
7. Cross-container CUDA IPC needs a shared PID namespace (`--pid=host`): same-PID peers get `cudaErrorDeviceUninitialized`. This was diagnosed in the related PR. POSIX-FD import has no PID check, but the fd transport below needs the peers to reach each other (shared network namespace for abstract sockets, or a shared path).

## 4. Design

### 4.1 Registration (exporter)
In `IntraNodeNvlinkTransport::registerLocalMemory(addr, length, ...)`:
1. If `cuMemRetainAllocationHandle(addr)` fails → legacy path, unchanged.
2. Otherwise **enumerate chunks** covering `[addr, addr + length)`: start at `cuMemGetAddressRange(addr)`, then walk to `base + size` until the end. Every byte must be mapped; if any part returns `CUDA_ERROR_NOT_FOUND`, fail with a clear error (e.g. PyTorch expandable segments that have not grown that far yet).
3. For each chunk:
   - `cuMemRetainAllocationHandle` → handle.
   - `cuMemGetAllocationPropertiesFromHandle` → `requestedHandleTypes`. If it lacks `POSIX_FILE_DESCRIPTOR` (and FABRIC, when supported), fail. For PyTorch, the message should say "set `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=1`".
   - Keep the retained handle (released on unregister).
4. Publish **one** `BufferDesc` for the whole registered range, so `relocateSharedMemoryAddress` keeps its containment logic, plus a VMM chunk table:
   - new metadata fields, e.g. `vmm_chunks: [{offset, size, handle_type}]`;
   - an fd-server endpoint (§4.2) or, when FABRIC works, the fabric handles inline.
   - Keep `shm_name` empty for VMM entries.
   - Make old readers fail cleanly: today they would hit `"Mismatched NVLink data transfer method"`. Check how `BufferDesc` is serialized in `transfer_metadata.cpp` and add fields there, JSON and any binary path.
5. De-duplicate by registered range (not by allocation base) for VMM. Several tensors inside one expandable segment can be registered separately; decide whether to merge overlapping ranges or refcount chunks.

### 4.2 Passing POSIX fds between unrelated processes (recommended: `SCM_RIGHTS`)
An exported fd is a per-process kernel object. It cannot be put in metadata as bytes.
- **Exporter:** one fd server per `TransferEngine` (per process). It listens on a Unix domain socket and holds the exported fds (`cuMemExportToShareableHandle(&fd, h, POSIX_FILE_DESCRIPTOR)`) for registered chunks. Bind an abstract socket (`@mooncake-nvlink-<segment-name>-<pid>`, works with `--network host`); a filesystem path under a configurable dir (`MC_NVLINK_FD_DIR`, default `/dev/shm` or `/tmp`) is an alternative for setups that share a volume but not the network namespace. Publish the endpoint in the segment metadata.
- **Importer:** on first relocation into a VMM entry, connect, request `(buffer_id, chunk indices)`, and receive the fds with `recvmsg` `SCM_RIGHTS`, batched (many chunks per message; PyTorch expandable segments use 20 MiB or 2 MiB chunks, so a 4 GiB tensor is up to ~200 fds).
- **Authentication:** at least check `SO_PEERCRED` uid. Any process that can open the socket can map the memory, which matches `cudaIpc` semantics but should be documented.
- **Optional fallback:** `pidfd_open` + `pidfd_getfd`, as PyTorch does, when the caller has `CAP_SYS_PTRACE` or ptrace scope 0. Off by default.
- **FABRIC:** if `cuMemCreate(FABRIC)` succeeds on the platform (IMEX configured), fabric handles can be published inline, like the MNNVL transport does, and no fd server is needed. Prefer FABRIC when both are available.

### 4.3 Import (initiator side, `relocateSharedMemoryAddress`)
For a VMM `BufferDesc` that is not yet in `remap_entries_`:
1. Get the fds (or fabric handles) for all chunks of the entry. Lazily importing per chunk is possible, but import-all on first touch is simpler and matches the current behaviour.
2. `cuMemImportFromShareableHandle(&h_i, (void*)(uintptr_t)fd_i, POSIX_FILE_DESCRIPTOR)`, then `close(fd_i)`.
3. `cuMemAddressReserve(&va, entry.length, granularity, 0, 0)`, then `cuMemMap(va + offset_i, size_i, 0, h_i, 0)` for every chunk.
4. `cuMemSetAccess(va, entry.length, {location = this process's current device, READWRITE})`. The copy engine of the local device reads/writes the peer's physical memory over NVLink; check whether access must also be granted for the owning device. The MNNVL code sets access for all devices, `~L1405`; mirror that if needed.
5. Store `{va, length, handles[]}` in `remap_entries_` with a type tag. Teardown (unregister of the peer segment, destructor): `cuMemUnmap` → `cuMemAddressFree` → `cuMemRelease(h_i)`, instead of `cudaIpcCloseMemHandle`.
6. Keep the existing `remap_lock_` double-checked pattern.

### 4.4 Unregister (exporter)
Release the retained handles. Close exported fds held by the fd server for that buffer. Remove the metadata entry. Peers that already imported keep valid mappings: imported handles hold a reference to the physical memory, which matches `cudaIpc` semantics. Document that the exporter must not reuse the physical memory for other data while peers may still write.

### 4.5 Configuration & compatibility
- No change for legacy buffers.
- New env:
  - `MC_NVLINK_VMM` (default on): set to `0` to restore today's behaviour.
  - `MC_NVLINK_FD_DIR` (optional socket dir).
  - `MC_NVLINK_PIDFD` (default off): enable the pidfd fallback.
- Mixed versions: a new initiator must handle old exporters (no VMM fields). An old initiator talking to a new VMM exporter must fail with a clear message, not a crash.

## 5. Tests (all on the B200 box; use `/opt/dlami/nvme/...` for temporary files and remove them afterwards)
1. **Unit/integration, raw CUDA:**
   - The exporter process creates a VMM buffer as 3 chunks of different sizes mapped into one reservation, `requestedHandleTypes=POSIX_FD`, fills it with a pattern, and registers it.
   - The importer process (different GPU, one visible device: `CUDA_VISIBLE_DEVICES=<n>`) writes via `transfer_sync_write`. Do one write that spans a chunk boundary, then a read back.
   - The exporter verifies the pattern.
   - Repeat with source and destination roles swapped, and with the legacy buffer on one side.
2. **PyTorch expandable segments:**
   - With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=1`: allocate a 4 GiB tensor, register it, transfer, verify.
   - Without `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=1`: registration fails with the new, explicit message.
3. **Containers:**
   - PE/CE-style: two containers with `--ipc=host --network host` and both `--pid=host` and separate PID namespaces; separate PID namespaces must now work for VMM buffers.
   - A container without a shared network namespace, using `MC_NVLINK_FD_DIR` on a shared volume.
4. **Teardown and lifetime:** exporter unregisters while the importer still has the mapping; importer process exit; repeated register/unregister cycles with no fd or VA leaks (`ls /proc/<pid>/fd | wc -l`, `cuMemGetInfo`).
5. **Performance:** bandwidth of one 112 MiB and one 268 MiB write versus the legacy path. Baseline on this box: 664–723 GB/s for legacy `nvlink_intra`, GPU0→GPU4, `bench/mc_nvlink_probe.py`. Also measure a 900 × 114 KiB batch (legacy: 3.0 ms).
6. **CI:** add a unit test runnable on a 2-GPU NVLink runner if the repo's CI has one; otherwise mark it manual, following the repo's conventions.

## 6. Deliverables
- PR to `kvcache-ai/Mooncake`: `[TE] nvlink_intra: support VMM (cuMemCreate) buffers via POSIX-fd / fabric handles`. Include the module/type/testing sections of `.github/pull_request_template.md` and the AI-assistance disclosure. Reference #2511 and the diagnostics PR. If it exceeds 500 LOC, the template asks for an RFC issue first.
- Docs: update `docs/source/getting_started/supported-protocols.md` (nvlink_intra requirements). The diagnostics PR says "registered buffers must be legacy allocations"; change it to "legacy, or VMM with a shareable handle type (PyTorch: `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=1`)".
- Format with clang-format **20** (`./scripts/code_format.sh`; clang-format 22 reformats unrelated lines).
- Pushing to public GitHub from a corporate laptop may be blocked by Code Defender: ask the user how to push. Do not work around it.

## 7. Open questions for the implementer
- Should the chunk table be published per registered range, or per physical allocation shared by ranges? This affects de-duplication when one expandable segment backs many registered tensors.
- PyTorch expandable segments grow and shrink. Mooncake only knows what was mapped at registration, so registering a range that is later unmapped and remapped by the allocator is unsafe. Document "register stable buffers" rather than trying to track the allocator.
- Is `cuMemSetAccess` for the importer's device sufficient for copy-engine access to a peer GPU's VMM memory on all supported topologies (NVSwitch vs. direct NVLink, PCIe-only fallback)?

<details><summary>中文</summary>

# Mooncake `nvlink_intra`：支持 VMM（`cuMemCreate`）显存 —— 设计与任务说明

这是交给实现 agent 的任务说明。目标仓库是 kvcache-ai/Mooncake 的 Transfer Engine。相关：#2511（开启 expandable_segments 后注册失败，issue 已关闭但没有修复），以及我们的诊断 PR（分支 `whn09:nvlink-intra-container-hints`，提交 `caeecde`，增加报错提示和文档，尚待推送）。

**目标：** 让 `nvlink_intra` 能注册 VMM 显存，并支持同一台机器上跨进程的读写传输；传统显存的行为不变。现在 nvlink_intra 只支持 `cudaMalloc`/`cuMemAlloc` 分配的显存，因为它靠 `cudaIpcGetMemHandle` 共享，而 `cuMemCreate` 分配的显存没有这种句柄。最典型的场景是 PyTorch 开了 `expandable_segments`：DeepInfer 无条件开启它，vLLM/SGLang 的用户也常开。跨机属于 MNNVL `nvlink` 传输，它已经支持 FABRIC 句柄，不在本任务范围内。

**现有代码（§2）：** nvlink_intra 的注册、地址重定位、反注册三个函数的位置，见英文部分。可以参考的实现有两处：一是 MNNVL `nvlink_transport.cpp` 里的 FABRIC 导出/导入路径；二是 PyTorch `ExpandableSegment` 的做法：只有设置 `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=1` 才会申请 POSIX FD 或 FABRIC 句柄，再通过 `pidfd_open`/`pidfd_getfd` 在进程间共享。

**在 B200 上验证过的事实（§3）：**
1. 机器支持 VMM 和 POSIX FD 句柄。
2. FABRIC 属性虽然是 1，但用 FABRIC 调 `cuMemCreate` 返回 800（NOT_PERMITTED），因为没有 IMEX channel。所以单机 HGX 上通用的方案是 POSIX FD。
3. VMM 显存调 `cudaIpcGetMemHandle` 会失败。`cuMemRetainAllocationHandle` 只对 VMM 显存成功，可以用来判断，每次调用都要配一次 `cuMemRelease`。
4. 对 VMM 显存，`cuMemGetAddressRange` 返回的是单个物理块的映射范围，不是整段预留，可以靠它逐块枚举；碰到没映射的地方返回 500。这也说明现在的代码对 VMM 只会注册第一个物理块。
5. `requestedHandleTypes` 可以查到，没有申请可导出句柄类型的显存能识别出来，并给出明确提示。
6. `ptrace_scope=1`：`pidfd_getfd` 只能用于子孙进程，不适合 Mooncake 这种互不相关的对端进程，默认不要依赖它。
7. 跨容器使用传统 IPC 需要 `--pid=host`。

**设计（§4）：**
- **注册：** 先判断是不是 VMM；是的话逐块枚举，检查每块的句柄类型，然后为整段范围发布一条 `BufferDesc`，附上物理块表和 fd 服务地址。
- **传递文件描述符：** 推荐每个进程起一个 Unix socket 服务，用 `SCM_RIGHTS` 传 fd，并用 `SO_PEERCRED` 校验 uid；`pidfd_getfd` 作为可选退路，默认关闭；平台支持 FABRIC 时优先用 FABRIC，直接把句柄写进元数据。
- **导入：** 按 MNNVL 的流程做：import → reserve → map → setAccess，结果缓存在 `remap_entries_`；释放时依次 unmap、free、release。
- **兼容：** 新增环境变量 `MC_NVLINK_VMM`、`MC_NVLINK_FD_DIR`、`MC_NVLINK_PIDFD`；新旧版本混用时要给出清晰的报错，不能崩溃。

**测试（§5）：** 原始 CUDA 写的多物理块显存（写入要跨过块边界、读写方向都测）；PyTorch expandable segments（开和不开 IPC 两种情况）；容器场景（`--pid=host` 与独立 PID 命名空间、不共享网络时用共享目录）；释放与泄漏检查；带宽对比（这台机器上传统路径是 664–723 GB/s）。

**交付物（§6）：**
- 向上游提 PR，按模板填写并注明使用了 AI。
- 更新文档里 nvlink_intra 的要求说明。
- 用 clang-format 20 格式化代码。
- 如果推送被 Code Defender 拦截，先问用户怎么处理，不要自行绕过。

**待定问题（§7）：** 物理块表按"注册范围"发布还是按"物理块"发布；PyTorch expandable segments 会扩缩，文档里应建议只注册稳定不变的缓冲区；`cuMemSetAccess` 在不同拓扑下是否都够用。
</details>
