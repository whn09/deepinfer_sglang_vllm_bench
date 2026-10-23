<!-- Posted to deepseek-ai/DeepInfer#144: https://github.com/deepseek-ai/DeepInfer/pull/144#issuecomment-6365218399 -->
**Greedy outputs differ between a prefix-cache hit and the uncached path, while the engine itself is deterministic. Is that expected with compact YoCo replay?**

Setup:
- 4×B200, DeepSeek-V4.1-Flash-1017, `serve/openai_server.py` (#147 @ 269a61f) with this PR @ e1f71ec.
- `--dp 4`, MTP 5 with real verification, temperature 0, `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, keep 140.

Repro on one fresh server (no restart, host pool only):
1. Turn 1: a ~45K-token user message (a deterministic pseudo-document plus a question), cache on.
2. Turn 2 = turn 1 + the assistant answer + a short follow-up. Sent first with the cache on: hit 44,928 tokens. Then the identical request with `"prefix_cache": false`.

The two completions agree for ~70 characters, then diverge:

```
cache on : We need answer. "it" refers gamma2? Which word comes right before gamma2. At fir…
cache off: We need answer. "it" refers gamma2? Which word comes right before gamma2. In ini…
```

What I could rule out:
- **Run-to-run noise:** the same request sent twice gives byte-identical output.
- **Batch composition:** c=1 vs c=8 give identical outputs for 32 requests, even without `DEEPINFER_BATCH_INVARIANT`.
- **The cache transport:** the restored pages and state are byte-exact (unit-tested in deepseek-ai/DeepInfer#216).

What changes the output is the resume point. The same turn-2 prompt resumed at 44,544 tokens (from a Mooncake Store L3 tier with 4-page objects) and at 44,928 tokens (from the host pool) gives **two different** continuations, and both differ from the uncached one. The uncached outputs of the two runs are byte-identical. So the difference is between "resume at boundary b and recompute the tail" and "prefill everything", and it depends on b.

A related observation: even without the prefix cache, `compact` keep=140 versus truncation off gives different greedy outputs on 29 of 32 requests (sg-8k prompts). With `DEEPINFER_BATCH_INVARIANT=1` it is still 28 of 32.

So both the compact truncation and the resume path change the computation order: the consumer-layer prefill covers a different query span or chunking, and the SWA state is rebuilt from the recomputed tail. Is the cache-hit path meant to be bit-exact with an uncached prefill, or only equivalent up to floating-point reordering? If it is the latter, a note in the PR description would help users who A/B with greedy outputs; `serve/check_prefix_cache.py` would report these runs as `outputs DIFFER`. If it is meant to be exact, I'm happy to provide more repro details.

<details><summary>中文</summary>

**前缀缓存命中时的 greedy 输出与不走缓存时不同，而引擎本身是确定性的。在 compact YoCo 重放下，这是预期行为吗？**

环境：4×B200，DeepSeek-V4.1-Flash-1017，`serve/openai_server.py`（#147 @ 269a61f）加本 PR @ e1f71ec；`--dp 4`，MTP 5 + 真实验证，temperature 0，`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`，keep 140。

复现方法（同一个新启动的 server，不重启，只用主机池）：
1. 第一轮：一条约 4.5 万 token 的用户消息（确定性生成的伪文档加一个问题），开缓存。
2. 第二轮 = 第一轮 + 助手回答 + 一个简短追问。先开缓存发一次，命中 44,928 token；再发完全相同、但带 `"prefix_cache": false` 的请求。

两次输出前约 70 个字符相同，之后开始分开（见上面的输出片段）。

已经排除的因素：
- **运行间噪声：** 同一个请求发两次，输出逐字节相同。
- **batch 组成：** 即使不开 `DEEPINFER_BATCH_INVARIANT`，c=1 和 c=8 下 32 个请求的输出也完全相同。
- **缓存传输：** 恢复出的页和 state 逐字节精确（deepseek-ai/DeepInfer#216 里有单元测试）。

真正改变输出的是恢复点。同一个第二轮 prompt，在 44,544 token 处恢复（从 4 页一组的 Mooncake Store L3）和在 44,928 token 处恢复（从主机池），得到的是**两个不同**的续写，而且都和不走缓存的输出不同；两次运行里不走缓存的输出则逐字节相同。所以差异出在"在边界 b 恢复后重算末尾一段"与"整段 prefill"之间，并且随 b 变化。

一个相关的观察：即使不用前缀缓存，`compact` keep=140 与关闭截断相比，32 个请求里也有 29 个 greedy 输出不同（sg-8k 的 prompt）；开了 `DEEPINFER_BATCH_INVARIANT=1` 仍有 28 个不同。

所以 compact 截断和恢复路径都会改变计算顺序：consumer 层 prefill 的 query 范围或分块不同，SWA state 也是从重算的末尾重建的。请问缓存命中路径的设计目标，是和不走缓存的 prefill 逐位一致，还是只在浮点重排的意义上等价？如果是后者，建议在 PR 描述里注明一下，方便用 greedy 输出做 A/B 的用户理解；`serve/check_prefix_cache.py` 跑这类测试时会报 `outputs DIFFER`。如果设计上应该逐位一致，我很乐意提供更多复现细节。
</details>
