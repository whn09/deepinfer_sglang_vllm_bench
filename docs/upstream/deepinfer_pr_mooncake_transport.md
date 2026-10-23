<!-- Opened as Draft deepseek-ai/DeepInfer#214 (whn09:pd-mooncake-transport @ d005d1b) -->
**Title:** serve: Mooncake Transfer Engine KV transport for the prefill/decode split (`--kv-transport mooncake`)

**Stacked on #197** (and therefore #144 / #147 / #195). Only the last commit is new. It adds a Mooncake [Transfer Engine](https://github.com/kvcache-ai/Mooncake) data plane to #197's prefill/decode split. #197's flow, frontend and control socket are unchanged; the default `--kv-transport tcp` behaves exactly as before.

**What it does**
- `deepinfer/hicache/pd_mooncake.py`:
  - `GpuArena`: one registered device buffer per rank, first-fit regions. It is allocated with `cuMemAlloc` like the KV pools, because the worker's `expandable_segments` memory has no cudaIpc handle.
  - `MooncakeLink`: a `TransferEngine` with `P2PHANDSHAKE` metadata (no etcd).
- `PDTransfer.export_into` / `admit` accept device buffers. The PE exports a sequence (KV pages + full `TO_CE` state) straight into an arena region, and the CE admits from its arena with the existing page-copy kernels. The state needs no extra staging copy.
- `pd_server.py`: control on #197's KV socket. The PE sends a header; the CE reserves an arena region and replies `{addr, segment}`, or `busy` and the PE retries; the PE does one contiguous `transfer_sync_write`, then `go` / `ok`. Failed writes are retried, then sent over the TCP path, so a failing peer cannot strand a request; the PE also uses TCP when its arena is full.
- `--mooncake-protocol nvlink_intra` (default, PE and CE on one host) or `rdma` / `efa` across hosts.
- `serve/README.md`: usage, plus the container requirement (`--ipc=host --pid=host`, see below).

**Tested (8×B200, one host: PE GPUs 0-3, CE GPUs 4-7, two containers, DeepSeek-V4.1-Flash-1017, `--dp 4` each)**
- Greedy output over Mooncake is byte-identical to the TCP path.
- `nvlink_intra` raw bandwidth between the processes: 664–723 GB/s for one 112–268 MB contiguous write. For comparison, 900 separate 114 KB block writes take 3.0 ms (34 GB/s), which is why the arena is contiguous.
- AgentX (SemiAnalysis traces) end-to-end Mooncake vs TCP at c = 16 / 64 / 128: *running; numbers to follow.*

**Two pitfalls found on the way** (also reported to Mooncake):
1. `torch.empty` arenas fail `register_memory`: `cudaIpcGetMemHandle failed`, because of `expandable_segments`. Use `cuMemAlloc`.
2. With PE and CE in separate containers, only the same-index rank pairs failed with `cudaIpcOpenMemHandle failed: invalid device context`. A CUDA IPC handle carries the exporter's PID. Per-container PID namespaces gave the PE's and CE's rank-r workers the same PID, so CUDA treated the peer handle as its own. The fix is `--pid=host`.

<details><summary>中文</summary>

**标题：** serve：为 PD 分离增加 Mooncake Transfer Engine KV 传输（`--kv-transport mooncake`）

**叠在 #197 之上**（因此也依赖 #144 / #147 / #195），只有最后一个提交是新的。它给 #197 的 PD 分离加了一条 Mooncake Transfer Engine 数据通道。#197 的流程、前端和控制 socket 都不变；默认的 `--kv-transport tcp` 行为和原来完全一样。

**改了什么**
- 新增 `deepinfer/hicache/pd_mooncake.py`：
  - `GpuArena`：每个 rank 一块注册好的显存，按 first-fit 分配区域。用 `cuMemAlloc` 分配，和 KV 池一样，因为 worker 开了 `expandable_segments`，这种显存拿不到 cudaIpc 句柄。
  - `MooncakeLink`：使用 `P2PHANDSHAKE` 元数据的 `TransferEngine`，不需要 etcd。
- `PDTransfer.export_into` / `admit` 支持显存缓冲区：PE 把序列（KV 页 + 完整的 `TO_CE` state）直接导出到 arena 的一块区域，CE 用现有的 page-copy kernel 从自己的 arena 接收。state 不需要额外的暂存拷贝。
- `pd_server.py`：控制消息走 #197 原有的 KV socket。PE 发头部；CE 预留 arena 区域并回复 `{addr, segment}`，满了就回 `busy`，PE 重试；PE 做一次连续的 `transfer_sync_write`，然后 `go` / `ok`。写入失败会先重试，再退回 TCP 发送，所以出问题的对端不会让请求卡住；PE 的 arena 满了也走 TCP。
- `--mooncake-protocol`：默认 `nvlink_intra`（PE 和 CE 在同一台机器上），跨机可用 `rdma` / `efa`。
- `serve/README.md`：用法说明，以及容器部署要求（`--ipc=host --pid=host`）。

**测试**（8×B200 单机：PE 用 GPU 0-3、CE 用 GPU 4-7，两个容器，DeepSeek-V4.1-Flash-1017，两边各 `--dp 4`）
- Mooncake 和 TCP 的 greedy 输出逐字节相同。
- 两个进程之间 `nvlink_intra` 的裸带宽：112–268 MB 一次连续写 664–723 GB/s；拆成 900 个 114 KB 小块分别写要 3.0 ms（34 GB/s），所以 arena 设计成连续的。
- AgentX 端到端 Mooncake 对比 TCP（c = 16 / 64 / 128）：*正在跑，数据稍后补上。*

**过程中踩到的两个坑**（已反馈给 Mooncake）：
1. 用 `torch.empty` 分配的 arena 注册失败（`cudaIpcGetMemHandle failed`），原因是 expandable_segments。改用 `cuMemAlloc`。
2. PE 和 CE 跑在两个容器里时，只有同序号的 rank 组合报 `cudaIpcOpenMemHandle failed: invalid device context`。CUDA IPC 句柄里带着导出方的 PID，而每个容器有自己的 PID 命名空间，PE 和 CE 的第 r 个 worker 拿到了同一个 PID，CUDA 就把对端的句柄当成了自己的。修法是 `--pid=host`。
</details>
