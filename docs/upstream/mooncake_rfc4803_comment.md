<!-- Posted to kvcache-ai/Mooncake#4803: https://github.com/kvcache-ai/Mooncake/issues/4803#issuecomment-6360317607 -->
An engine-side consumer for this contract now exists: **deepseek-ai/DeepInfer#216** (draft) adds Mooncake Store as an L3 tier under DeepInfer's host prefix cache (#144 there), for DeepSeek-V4.1-Flash. Notes from it that may help the adapter and the contract tests:

**How it maps to Sections A/B today** (plain `MooncakeDistributedStore`, until the adapter lands):
- **State objects:** `state/<fp>/<prefix chain hash at the 128-aligned boundary>`. One object per boundary, holding exactly the persisted SWA region DeepInfer stores at that position (`state_size_to_disk`, taken with `select_state_slice_to_store`). Boundaries are the last few 128-aligned positions of each stored prompt.
- **Block KV:** a separate namespace, `dsv4kv/<fp>/g<G>/<chain hash of the last block>`. Each object holds G consecutive 128-token blocks, all layers per block.
- **`<fp>`:** `v1-<sha256[:16]>` over the checkpoint `config.json` plus page bytes, state bytes, block size and the env knobs that change the state ring layout. The version is in the fingerprint itself, as suggested, so a layout change is a clean miss.
- **Hit resolution follows Section B:** `batch_is_exist` on block keys gives the block run; `batch_is_exist` on state keys at candidate boundaries gives state availability. The engine then picks the deepest boundary that it can also replay from (it must recompute ≥ 140 trailing tokens) and fetches with `batch_get_into_multi_buffers`. The Store never infers policy.
- **Expiry:** an object that is gone between `exists` and `get` (we treat `-704` / size mismatch as a miss) falls back to the host-only lookup. There is a unit test for exactly that case.

**Observations from B200, TCP, one host, wheel 0.3.13.post1** that may matter for defaults:
- **Per-object overhead dominates for single 114 KB blocks.** `batch_put_from` reaches 0.17 GB/s with one block per object, but 1.7 GB/s with 4 blocks per object (the loopback limit). Hence G=4 (512 tokens) for block objects. State objects are larger and single.
- The Store rejects `protocol=shm` (`unsupported_protocol`). `protocol=efa` on the same host gave `TRANSFER_FAIL (-800)` for every put; I haven't dug into that yet.

Once `python/mooncake/store/`'s adapter lands, DeepInfer can switch its state path to it directly. Happy to run the contract tests against real V4.1 states, and to share layout numbers, if that is useful.

<details><summary>中文</summary>

这个约定现在已经有了引擎侧的使用方：**deepseek-ai/DeepInfer#216**（Draft）在 DeepInfer 的主机前缀缓存（那边的 #144）下面加了一层 Mooncake Store 作为 L3，面向 DeepSeek-V4.1-Flash。下面几点或许对 adapter 和契约测试有帮助。

**目前和 Section A/B 的对应关系**（adapter 落地之前，直接用 `MooncakeDistributedStore`）：
- **state 对象：** `state/<fp>/<128 对齐边界处的前缀链哈希>`。每个边界一个对象，内容正好是 DeepInfer 在该位置保存的 SWA 区域（`state_size_to_disk`，用 `select_state_slice_to_store` 取出）。边界取每个已保存 prompt 的最后几个 128 对齐位置。
- **Block KV：** 用单独的命名空间，`dsv4kv/<fp>/g<G>/<最后一个 block 的链哈希>`。每个对象包含 G 个连续的 128-token block，每个 block 含所有层。
- **`<fp>`：** `v1-<sha256 前 16 位>`，由 checkpoint 的 `config.json` 加上页字节数、state 字节数、block 大小，以及会改变 state 环形布局的环境变量算出。按建议把版本号放进了指纹本身，布局一变就自然查不到。
- **命中判定遵循 Section B：** 对 block key 做 `batch_is_exist` 得到连续 block 的长度，对候选边界的 state key 做 `batch_is_exist` 得到 state 是否可用；然后由引擎挑出最深的、自己也能重放的边界（末尾必须重算至少 140 个 token），再用 `batch_get_into_multi_buffers` 读取。Store 本身不做任何策略推断。
- **过期：** `exists` 和 `get` 之间对象消失时（`-704` 或大小不符都按未命中处理），退回只用主机池的查找。这种情况有专门的单元测试。

**在 B200 上的观察**（TCP，单机，wheel 0.3.13.post1），可能和默认参数有关：
- **对单个 114 KB 的 block 来说，每个对象的固定开销占了大头。** 每对象 1 个 block 时 `batch_put_from` 只有 0.17 GB/s，每对象 4 个 block 能到 1.7 GB/s（回环带宽上限），所以 block 对象取 G=4（512 token）。state 对象本身较大，单独存放。
- Store 不接受 `protocol=shm`（`unsupported_protocol`）；同机用 `protocol=efa` 时每次 put 都返回 `TRANSFER_FAIL (-800)`，我还没来得及深入查。

等 `python/mooncake/store/` 的 adapter 落地后，DeepInfer 可以直接把 state 路径切过去。如果有用，我很乐意用真实的 V4.1 state 跑契约测试，也可以提供布局相关的数据。
</details>
