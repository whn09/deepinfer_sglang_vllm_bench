"""Cold long-prefill TTFT at c=1 against any OpenAI-compatible server (DeepInfer / vLLM / SGLang). Each prompt is a
distinct pseudo-document (seeded per request), so nothing is in any prefix cache; max_tokens=1, so TTFT ~ prefill.

    python bench/long_prefill.py --port 8888 --lengths 25000,50000,100000,200000,300000 --reps 2 --out di.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]


def prompt(n_words: int, seed: int) -> str:
    r = random.Random(seed)
    return " ".join(f"{r.choice(WORDS)}{r.randrange(997)}" for _ in range(n_words)) + "\n\nSummarize in one word."


def run(port: int, model: str, text: str, extra: dict) -> tuple[float, int]:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, ptok = time.perf_counter(), None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and ttft is None:
                ttft = time.perf_counter() - t0
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens", 0)
    return ttft or (time.perf_counter() - t0), ptok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--lengths", default="25000,50000,100000,200000,300000", help="target prompt tokens")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--no-cache-field", action="store_true", help="omit DeepInfer's prefix_cache=false (vLLM/SGLang)")
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    extra = {} if a.no_cache_field else {"prefix_cache": False}
    run(a.port, a.model, prompt(2000, 0), extra)  # warmup
    rows = []
    for n in map(int, a.lengths.split(",")):
        for rep in range(a.reps):
            ttft, ptok = run(a.port, a.model, prompt(int(n / 2.6), n * 100 + rep), extra)
            rows.append({"target": n, "rep": rep, "prompt_tokens": ptok, "ttft_s": round(ttft, 3),
                         "tok_per_s": round(ptok / ttft) if ptok else None})
            print(json.dumps({"label": a.label, **rows[-1]}), flush=True)
    json.dump({"label": a.label, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
