"""DeepInfer server TTFT 按阶段拆开（#17）：读 DEEPINFER_SERVER_TRACE 写的 jsonl，输出各阶段 p50/p90/mean。

阶段（server 内部，从收到 HTTP 请求到发出第一个有内容的 SSE chunk）：
  json        解析请求体                                  f_json - f_recv
  render      chat 模板 + tokenize                        f_rendered - f_json
  ipc_wait    提交后等 worker 取走（worker 只在两个 step 之间收请求）  w_get0 - f_submitted
  unpickle    mp.Queue 取出（含反序列化 prompt 列表）       w_dequeued - w_get0
  add         engine.add_request                          w_added - w_dequeued
  cache       前缀缓存决策 + 主机→GPU 恢复 + 等可分配显存    w_enqueued - w_added
  engine      进调度器 → 首个 token（prefill/decode step 分开记） w_first_token - w_enqueued
  return      worker → 前端事件循环                        f_first_token - w_first_token
  first_sse   detokenize 到第一个非空文本                   f_first_sent - f_first_token

  python bench/ttft_breakdown.py trace.jsonl [--min-plen N] [--by-hit]
"""

from __future__ import annotations

import argparse
import json
import statistics as st

STAGES = [
    ("json", "f_recv", "f_json"),
    ("render", "f_json", "f_rendered"),
    ("ipc_wait", "f_submitted", "w_get0"),
    ("unpickle", "w_get0", "w_dequeued"),
    ("add", "w_dequeued", "w_added"),
    ("cache", "w_added", "w_enqueued"),
    ("engine", "w_enqueued", "w_first_token"),
    ("return", "w_first_token", "f_first_token"),
    ("first_sse", "f_first_token", "f_first_sent"),
]


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def table(rows: list[dict], title: str) -> None:
    if not rows:
        return
    total = [r["f_first_sent"] - r["f_recv"] for r in rows]
    print(f"\n### {title}  (n={len(rows)}, prompt p50 {q([r['plen'] for r in rows], .5):,} tok, "
          f"cache hit p50 {q([r['hit'] / max(r['plen'], 1) for r in rows], .5):.1%})")
    print("| stage | p50 ms | p90 ms | mean ms | share of mean |")
    print("|---|---|---|---|---|")
    mean_total = st.mean(total)
    for name, a, b in STAGES:
        xs = [(r[b] - r[a]) * 1e3 for r in rows]
        print(f"| {name} | {q(xs, .5):.1f} | {q(xs, .9):.1f} | {st.mean(xs):.1f} | {st.mean(xs) / 1e3 / mean_total:.0%} |")
    for k in ("prefill_step_s", "decode_step_s"):
        xs = [r[k] * 1e3 for r in rows]
        print(f"| engine.{k[:-7]} steps | {q(xs, .5):.1f} | {q(xs, .9):.1f} | {st.mean(xs):.1f} | {st.mean(xs) / 1e3 / mean_total:.0%} |")
    for k in ("prefill_steps", "decode_steps"):
        xs = [r[k] for r in rows]
        print(f"| #{k} | {q(xs, .5)} | {q(xs, .9)} | {st.mean(xs):.1f} | |")
    print(f"| **server TTFT** | {q(total, .5) * 1e3:.1f} | {q(total, .9) * 1e3:.1f} | {mean_total * 1e3:.1f} | 100% |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--min-plen", type=int, default=0)
    p.add_argument("--by-hit", action="store_true", help="分开统计前缀缓存命中 / 未命中（冷请求）")
    a = p.parse_args()
    need = {k for _, x, y in STAGES for k in (x, y)}
    rows = [r for r in map(json.loads, open(a.trace)) if need <= r.keys() and r["plen"] >= a.min_plen]
    table(rows, a.trace)
    if a.by_hit:
        table([r for r in rows if r["hit"] > 0], "prefix-cache hit")
        table([r for r in rows if r["hit"] == 0], "cold (no hit)")


if __name__ == "__main__":
    main()
