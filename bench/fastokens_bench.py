import sys, time, json, re
sys.path.insert(0, "/model/encoding")
from encoding import encode_messages
from transformers import AutoTokenizer
ref_tok = AutoTokenizer.from_pretrained("/model")
para = open(sys.argv[1]).read()
msgs = [{"role": "system", "content": "You are a coding agent."}]
i = 0
while sum(len(m["content"]) for m in msgs) < 600_000:
    msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": para[(i * 997) % 2000:] + f"\nturn {i}"}); i += 1
prompt = encode_messages(msgs, thinking_mode="thinking")
def timeit(f, n=5):
    f(); ts = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return r, min(ts) * 1e3
ref, t_ref = timeit(lambda: ref_tok.encode(prompt, add_special_tokens=False))
import fastokens
from fastokens._native import Tokenizer
ft = Tokenizer.from_file("/model/tokenizer.json") if hasattr(Tokenizer, "from_file") else Tokenizer.from_model("/model")
_e = ft.encode(prompt); _get = (lambda e: list(e.ids)) if hasattr(_e, "ids") else (lambda e: list(e))
ids_ft, t_ft = timeit(lambda: _get(ft.encode(prompt)))
fastokens.patch_transformers()
pt = AutoTokenizer.from_pretrained("/model")
ids_pt, t_pt = timeit(lambda: pt.encode(prompt, add_special_tokens=False))
print(json.dumps({"fastokens": getattr(fastokens, "__version__", "?"), "tokens": len(ref), "hf_tokenizers_ms": round(t_ref, 1),
                  "fastokens_native_ms": round(t_ft, 1), "native_same": ids_ft == ref,
                  "patched_transformers_ms": round(t_pt, 1), "patched_same": ids_pt == ref}))
