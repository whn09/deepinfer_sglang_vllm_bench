Thanks, that explains it. I reproduced your result with my original prompt on 4×B200 (#147 @ 269a61f + this PR @ e1f71ec, compact keep 140, MTP 5 with real verification, temperature 0). The turn-2 prompt is ~45K tokens, and the host-pool hit resumes at b = 44,928:

| comparison | greedy output |
|---|---|
| cache hit at b vs uncached on a fresh server with `--max-num-batched-tokens 44928` | **byte-identical** |
| cache hit at b vs uncached with the default 32,768 chunk | differs |
| uncached with `--max-num-batched-tokens 45056` (b + 128) vs either of the above | differs from both |

So the cache path is exact, and the difference is entirely the split point. I'll use `--max-num-batched-tokens` = resume point as the control from now on. You're also right about my c=1 vs c=8 check: the split point never moved, so it doesn't show batch invariance. I've corrected that in our notes.

<details><summary>中文</summary>

谢谢，这就解释清楚了。我在 4×B200 上用我原来的 prompt 复现了你的结论（#147 @ 269a61f + 本 PR @ e1f71ec，compact keep 140，MTP 5 + 真实验证，temperature 0）。第二轮 prompt 约 4.5 万 token，主机池命中在 b = 44,928 处恢复：

- 在 b 处命中 vs 新启动、不走缓存、`--max-num-batched-tokens 44928` 的 server：**逐字节相同**。
- 在 b 处命中 vs 不走缓存、默认按 32,768 切分：不同。
- 不走缓存、`--max-num-batched-tokens 45056`（b + 128）：与上面两者都不同。

所以缓存路径是精确的，差异完全来自切分点。之后我会用"`--max-num-batched-tokens` 等于恢复点"的 server 做对照组。你说的 c=1 vs c=8 检查的问题也对：切分点从来没动过，所以它证明不了 batch 无关。我们的记录里已经更正。
</details>
