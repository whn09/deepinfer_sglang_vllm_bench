"""Greedy output fingerprint of a running server, for comparing KV transports of the PD split (#18).

Sends a fixed set of prompts (deterministic pseudo-documents of ~1K..64K tokens, so transfers range from a few
to several hundred KV pages) with temperature 0 and writes {name: md5 of the completion text}. Run it against the
Mooncake and the TCP server and diff the two files.

    python bench/pd_parity.py --port 8888 --out mooncake.json
    python bench/pd_parity.py --compare mooncake.json tcp.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]
SIZES = [1000, 4000, 16000, 64000]
QUESTIONS = [
    "Which word comes right after 'gamma2'? Answer briefly.",
    "Summarize the document in one sentence.",
    "List three words that appear in the document.",
]


def prompts() -> list[tuple[str, str]]:
    out = []
    for n in SIZES:
        doc = " ".join(f"{WORDS[(i * 7 + n) % len(WORDS)]}{i % 97}" for i in range(n))
        for qi, q in enumerate(QUESTIONS):
            out.append((f"{n}tok-q{qi}", f"{doc}\n\n{q}"))
    return out


def ask(port: int, model: str, text: str, max_tokens: int) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2)
    a = p.parse_args()
    if a.compare:
        x, y = (json.load(open(f)) for f in a.compare)
        same = [k for k in x if x[k]["md5"] == y.get(k, {}).get("md5")]
        print(f"identical {len(same)}/{len(x)}")
        for k in x:
            if k not in same:
                print(f"  DIFF {k}: {x[k]['text'][:80]!r} vs {y.get(k, {}).get('text', '')[:80]!r}")
        sys.exit(0 if len(same) == len(x) else 1)
    res = {}
    for name, text in prompts():
        t0 = time.perf_counter()
        out = ask(a.port, a.model, text, a.max_tokens)
        res[name] = {"md5": hashlib.md5(out.encode()).hexdigest(), "text": out, "seconds": round(time.perf_counter() - t0, 2)}
        print(f"{name}: {res[name]['md5'][:12]} {res[name]['seconds']}s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
