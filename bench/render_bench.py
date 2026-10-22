"""#17: where does render (chat template + tokenize) spend its time for a ~150K-token agentic conversation?"""
import sys, time, json
sys.path.insert(0, "/model/encoding")
from encoding import encode_messages
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/model")
para = open("/workspace/DeepInfer/README.md").read()
msgs = [{"role": "system", "content": "You are a coding agent."}]
i = 0
while sum(len(m["content"]) for m in msgs) < 600_000:
    msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": para[(i * 997) % 2000:] + f"\nturn {i}"})
    i += 1
t = time.perf_counter(); prompt = encode_messages(msgs, thinking_mode="thinking"); t_tpl = time.perf_counter() - t
t = time.perf_counter(); ids = tok.encode(prompt, add_special_tokens=False); t_tok = time.perf_counter() - t
t = time.perf_counter(); ids2 = tok(prompt, add_special_tokens=False)["input_ids"]; t_call = time.perf_counter() - t
bt = tok.backend_tokenizer
t = time.perf_counter(); ids3 = bt.encode(prompt, add_special_tokens=False).ids; t_rust = time.perf_counter() - t
# incremental: tokenize only the last turn's text on top of a cached prefix (what a per-conversation cache would do)
prefix = encode_messages(msgs[:-1], thinking_mode="thinking")
assert prompt.startswith(prefix) or True
t = time.perf_counter(); tail = tok.encode(prompt[len(prefix):] if prompt.startswith(prefix) else prompt[-20000:], add_special_tokens=False); t_tail = time.perf_counter() - t
print(json.dumps({"messages": len(msgs), "chars": len(prompt), "tokens": len(ids), "template_ms": round(t_tpl * 1e3, 1),
                  "tok.encode_ms": round(t_tok * 1e3, 1), "tok()_ms": round(t_call * 1e3, 1), "rust_encode_ms": round(t_rust * 1e3, 1),
                  "same_ids": ids == ids2 == ids3, "prefix_is_prefix": prompt.startswith(prefix), "tail_tokens": len(tail), "tail_ms": round(t_tail * 1e3, 2)}))

# --- parallel exact tokenization: split at added/special tokens (HF tokenizers splits there first anyway)
import re, os
added = sorted(tok.get_added_vocab().keys(), key=len, reverse=True)
pat = re.compile("(" + "|".join(re.escape(a) for a in added) + ")")
def par_encode(text):
    parts = [p for p in pat.split(text) if p]
    enc = bt.encode_batch(parts, add_special_tokens=False)
    return [i for e in enc for i in e.ids]
par_encode(prompt)  # warm the thread pool
t = time.perf_counter(); ids4 = par_encode(prompt); t_par = time.perf_counter() - t
print(json.dumps({"cpus": os.cpu_count(), "segments": len([p for p in pat.split(prompt) if p]), "parallel_ms": round(t_par * 1e3, 1), "parallel_same_ids": ids4 == ids}))
