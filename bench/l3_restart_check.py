"""Prefix-cache restart check against a running DeepInfer OpenAI server (#20). Needs real verification
(IX_SYNTHETIC_AL=0): synthetic acceptance draws random numbers per request, so two different requests never
produce comparable outputs.

  populate  send turn 1 with the cache on (stores to the host pool and, with L3, to the Mooncake Store)
  check     send turn 2 with the cache on FIRST (after a restart the host pool is empty, so a hit can only come
            from L3), then the same prompt with "prefix_cache": false; compare outputs, report TTFT and the
            server's hicache counters

    python bench/l3_restart_check.py populate --port 8888 --state /tmp/l3.json
    (restart the server)
    python bench/l3_restart_check.py check --port 8888 --state /tmp/l3.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]


def chat(port: int, model: str, messages: list[dict], max_tokens: int, prefix_cache: bool) -> tuple[str, float]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True, "prefix_cache": prefix_cache}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0, ttft, out = time.perf_counter(), None, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"]
            if delta and delta[0]["delta"].get("content"):
                ttft = ttft or time.perf_counter() - t0
                out.append(delta[0]["delta"]["content"])
    return "".join(out), (ttft or 0.0) * 1e3


def counters(port: int) -> dict[str, float]:
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/deepinfer/metrics", timeout=30).read().decode()
    tot: dict[str, float] = {}
    for m in re.finditer(r"^deepinfer_hicache:(\w+)\{[^}]*\}\s+([0-9.eE+-]+)$", text, re.M):
        tot[m.group(1)] = tot.get(m.group(1), 0.0) + float(m.group(2))
    return {k: tot[k] for k in ("lookups", "hits", "hit_tokens", "l3_lookups", "l3_hits", "l3_pages_loaded", "l3_states_loaded", "l3_get_seconds", "l3_put_objects") if k in tot}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["populate", "check"])
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    p.add_argument("--state", required=True)
    p.add_argument("--doc-tokens", type=int, default=20000)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--dump", help="check: write the full cached / uncached outputs here (JSON)")
    a = p.parse_args()
    doc = " ".join(f"{WORDS[i % len(WORDS)]}{i % 97}" for i in range(a.doc_tokens))
    turn1 = [{"role": "user", "content": f"{doc}\n\nWhich word comes right after 'gamma2'? Answer briefly."}]
    if a.phase == "populate":
        out, ttft = chat(a.port, a.model, turn1, a.max_tokens, True)
        json.dump({"answer1": out}, open(a.state, "w"))
        print(json.dumps({"phase": "populate", "ttft_ms": round(ttft), "counters": counters(a.port)}))
        return
    st = json.load(open(a.state))
    turn2 = turn1 + [{"role": "assistant", "content": st["answer1"]}, {"role": "user", "content": "And which word comes right before it?"}]
    before = counters(a.port)
    cached, t_cached = chat(a.port, a.model, turn2, a.max_tokens, True)
    after = counters(a.port)
    uncached, t_uncached = chat(a.port, a.model, turn2, a.max_tokens, False)
    if a.dump:
        json.dump({"cached": cached, "uncached": uncached}, open(a.dump, "w"))
    print(json.dumps({
        "phase": "check",
        "identical": cached == uncached,
        "ttft_ms_cached": round(t_cached),
        "ttft_ms_uncached": round(t_uncached),
        "delta_counters_for_cached_request": {k: round(after.get(k, 0) - before.get(k, 0), 3) for k in after},
        "cached_head": cached[:80],
        "uncached_head": uncached[:80],
    }))


if __name__ == "__main__":
    main()
