"""Steady-state decode speed at a fixed shape against any OpenAI-compatible server (#21 W2): N concurrent requests,
each a distinct prompt of ~L tokens, generating M tokens with ignore_eos. Per request, the decode-phase speed is
(tokens after the first) / (time after the first token); with N = 4 x per-rank batch on DEP4 this matches the
per-rank shape of an AgentX point without its arrival noise.

    python bench/decode_probe.py --port 30000 --n 16 --prompt-tokens 150000 --max-tokens 6000 --out sglang.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]


def prompt(n_tokens: int, seed: int) -> str:
    r = random.Random(seed)
    return " ".join(f"{r.choice(WORDS)}{r.randrange(997)}" for _ in range(int(n_tokens / 2.6))) + "\n\nWrite a long story."


def one(port: int, model: str, text: str, max_tokens: int, extra: dict, out: list, i: int) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "ignore_eos": True, "min_tokens": max_tokens, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, first, last, chunks = time.perf_counter(), None, None, 0
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = (json.loads(line[6:]).get("choices") or [{}])[0]
            if (ch.get("delta") or {}).get("content"):
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    out[i] = {"ttft_s": (first or t0) - t0, "decode_s": (last - first) if first else 0.0, "chunks": chunks, "end": last}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--prompt-tokens", type=int, default=150000)
    p.add_argument("--max-tokens", type=int, default=6000, help="long enough (~30 s) that the staggered prefills are a small part of the window")
    p.add_argument("--no-cache-field", action="store_true", help="omit DeepInfer's prefix_cache=false")
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    extra = {} if a.no_cache_field else {"prefix_cache": False}
    res: list = [None] * a.n
    ts = [threading.Thread(target=one, args=(a.port, a.model, prompt(a.prompt_tokens, 7000 + i), a.max_tokens, extra, res, i))
          for i in range(a.n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    # tokens per streamed chunk are not fixed (speculative decoding emits several), so use the requested length
    speeds = [(a.max_tokens - 1) / r["decode_s"] for r in res if r and r["decode_s"] > 0]
    summary = {"label": a.label, "n": a.n, "prompt_tokens": a.prompt_tokens, "max_tokens": a.max_tokens,
               "decode_tok_s_per_user_p50": round(statistics.median(speeds), 1),
               "decode_tok_s_per_user_mean": round(statistics.mean(speeds), 1),
               "ttft_s_max": round(max(r["ttft_s"] for r in res), 2), "requests": res}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}))


if __name__ == "__main__":
    main()
