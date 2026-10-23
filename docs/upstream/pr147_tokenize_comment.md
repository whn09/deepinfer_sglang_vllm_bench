<!-- Posted to deepseek-ai/DeepInfer#147: https://github.com/deepseek-ai/DeepInfer/pull/147#issuecomment-6355311603 -->
**Prompt tokenization dominates TTFT p50 for long agentic prompts; `fastokens` removes it (same ids)**

`Frontend.render_prompt` re-tokenizes the whole conversation on every turn with `tokenizer.encode`. For agentic traffic the prompt is 100K+ tokens at nearly every turn, so this is the largest single item in the server's TTFT once the prefix cache hits.

**Microbenchmark** (one 167K-token DeepSeek-V4.1 conversation, B200 host CPU):

| step | time |
|---|---|
| `encode_messages` (chat template) | 2.7 ms |
| `tokenizer.encode` (HF `tokenizers`; the Rust backend called directly is the same, 241 ms) | 238 ms |
| [`fastokens`](https://github.com/crusoecloud/fastokens) 0.3.4, `Tokenizer.from_file(tokenizer.json).encode(text).ids` | **2.1 ms**, identical ids |

`fastokens` is also what vLLM's Rust frontend loads first (`rust/src/tokenizer/src/hf.rs`), and SGLang exposes it as `--tokenizer-backend fastokens`.

**End to end** (8×B200, 4 GPUs, `--dp 4`, SemiAnalysis AgentX traces, c=64, 900 s each, server-side per-request tracing):

| | `tokenizer.encode` | fastokens |
|---|---|---|
| render p50 / p90 | 179 / 778 ms | **6.3 / 36 ms** |
| client TTFT p50 / p90 | 0.33 / 1.01 s | **0.14 / 0.48 s** |

Throughput and decode speed are unchanged. The first 200 real prompts were also encoded with `tokenizer.encode` as a check, with no mismatch.

**Suggestion:** use `fastokens` as an optional fast path with a fallback and a startup parity check. A minimal version:

```python
# Frontend.__init__, after AutoTokenizer.from_pretrained(...)
self._fast = None
try:
    from fastokens._native import Tokenizer as FastTokenizer  # optional: pip install fastokens
    self._fast = FastTokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    probe = "<｜begin▁of▁sentence｜>hi <｜User｜>x  y\n<｜Assistant｜></think>ok 你好"
    if list(self._fast.encode(probe).ids) != self.tokenizer.encode(probe, add_special_tokens=False):
        self._fast = None
except Exception:
    self._fast = None

# render_prompt
return list(self._fast.encode(prompt).ids) if self._fast else self.tokenizer.encode(prompt, add_special_tokens=False)
```

If a new dependency is not wanted, there is an exact dependency-free improvement. HF tokenizers split at added/special tokens before BPE, so splitting the prompt there and encoding the pieces with one `backend_tokenizer.encode_batch` call gives identical ids in 77 ms, multi-threaded. It only helps when the prompt has many segments. #197's `pd_server.py` reuses `render_prompt`, so the CE frontend would get this for free.

<details><summary>中文</summary>

**长 agentic prompt 下，tokenize 是 TTFT p50 的大头；换成 `fastokens` 就能消掉（token id 完全相同）**

`Frontend.render_prompt` 每一轮都用 `tokenizer.encode` 把整段对话重新编码。agentic 场景下几乎每一轮的 prompt 都在 10 万 token 以上，所以一旦前缀缓存命中，这一步就成了 server 端 TTFT 里最大的单项。

**微基准**（一段 16.7 万 token 的 DeepSeek-V4.1 对话，B200 主机 CPU）：chat 模板 2.7 ms；`tokenizer.encode` 238 ms（直接调 Rust 后端也是 241 ms）；`fastokens` 0.3.4 只要 **2.1 ms**，token id 完全相同。vLLM 的 Rust 前端默认就先用 `fastokens` 加载 tokenizer（`rust/src/tokenizer/src/hf.rs`），SGLang 也提供了 `--tokenizer-backend fastokens` 选项。

**端到端**（8×B200 中用 4 卡，`--dp 4`，SemiAnalysis AgentX 轨迹，c=64，各跑 900 s，server 端逐请求计时）：render p50/p90 从 179/778 ms 降到 **6.3/36 ms**；客户端 TTFT p50/p90 从 0.33/1.01 s 降到 **0.14/0.48 s**。吞吐和 decode 速度不变。前 200 个真实 prompt 另外用 `tokenizer.encode` 做了比对，没有任何不一致。

**建议：** 把 `fastokens` 作为可选的快速路径，保留回退，并在启动时做一次一致性自检（最小实现见上面的代码）。如果不想引入新依赖，还有一个不依赖任何新库、同样逐 id 精确的改进：HF tokenizers 在做 BPE 之前本来就会按特殊 token 切分，所以在这些位置把 prompt 切开、用一次 `backend_tokenizer.encode_batch` 多线程编码，结果完全相同，耗时 77 ms（只在 prompt 分段较多时有效）。#197 的 `pd_server.py` 复用了 `render_prompt`，CE 前端也会自动受益。
</details>
