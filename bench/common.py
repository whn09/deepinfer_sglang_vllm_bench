"""三家共用的数据集格式、单请求记录格式和指标计算。

只依赖标准库，DeepInfer venv 和任何 client 环境里都能 import。

数据集（jsonl，每行一个请求）：
    {"id": 0, "prompt_token_ids": [...], "output_len": 1024}
同目录下的 <dataset>.meta.json 记录生成参数（来源、长度、seed、tokenizer）。

单请求记录（各后端都写成这个格式，统一交给 compute_metrics）：
    {
      "id": 0,
      "prompt_len": 1024,          # 实际输入 token 数
      "output_len": 1024,          # 实际输出 token 数
      "expected_output_len": 1024, # 数据集要求的输出长度（ignore_eos 下两者应相等）
      "t_submit": 1729.0,          # 请求发出 / 加入引擎的时刻（time.time()，秒）
      "t_first": 1729.2,           # 收到第一个输出 token 的时刻
      "t_end": 1740.0,             # 收到最后一个 token 的时刻
      "chunks": [[t, n], ...],     # 每次收到输出的时刻和 token 数（n 未知时为 null）
      "error": null,
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any



# ---------------------------------------------------------------- dataset


def load_dataset(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if limit is not None and len(items) >= limit:
                    break
    return items


def load_dataset_meta(path: str | Path) -> dict[str, Any]:
    meta_path = Path(str(path) + ".meta.json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


# ---------------------------------------------------------------- metrics


def _stats(values: list[float], scale: float = 1.0) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p99": None}
    arr = sorted(v * scale for v in values)
    return {
        "mean": sum(arr) / len(arr),
        "p50": _percentile(arr, 50),
        "p90": _percentile(arr, 90),
        "p99": _percentile(arr, 99),
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数，与 numpy.percentile 默认（linear）一致。"""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _steady_output_throughput(records: list[dict[str, Any]], trim: float) -> float | None:
    """去掉测量窗口首尾各 trim 比例的时间后，窗口内收到的输出 token 数 / 窗口时长。

    闭环并发下，开头所有请求同时 prefill、结尾并发逐渐降到 0，这两段都不是稳态；
    稳态吞吐更接近"持续以并发 C 服务"时的真实能力。chunk 的 token 数未知（n 为 null）时
    按该请求的平均每 chunk token 数估算。
    """
    t0 = min(r["t_submit"] for r in records)
    t1 = max(r["t_end"] for r in records)
    lo, hi = t0 + (t1 - t0) * trim, t1 - (t1 - t0) * trim
    if hi <= lo:
        return None
    tokens = 0.0
    for r in records:
        chunks = r.get("chunks") or []
        if not chunks:
            continue
        known = [n for _, n in chunks if n is not None]
        avg = (r["output_len"] / len(chunks)) if len(known) < len(chunks) else None
        for t, n in chunks:
            if lo <= t <= hi:
                tokens += n if n is not None else avg
    return tokens / (hi - lo)


def compute_metrics(records: list[dict[str, Any]], num_gpus: int, steady_trim: float = 0.1) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    failed = len(records) - len(ok)
    if not ok:
        return {"num_requests": len(records), "num_failed": failed}

    t0 = min(r["t_submit"] for r in ok)
    t1 = max(r["t_end"] for r in ok)
    duration = t1 - t0
    total_in = sum(r["prompt_len"] for r in ok)
    total_out = sum(r["output_len"] for r in ok)

    ttft = [r["t_first"] - r["t_submit"] for r in ok]
    e2e = [r["t_end"] - r["t_submit"] for r in ok]
    # TPOT：首 token 之后每个 token 的平均间隔；只有 1 个输出 token 的请求没有定义
    tpot = [(r["t_end"] - r["t_first"]) / (r["output_len"] - 1) for r in ok if r["output_len"] > 1]
    # 单用户解码速度 = 1 / TPOT（tokens/s/user），画吞吐-延迟曲线的横轴
    user_speed = [1.0 / x for x in tpot if x > 0]

    # 平均每次收到输出时带的 token 数。DeepInfer 一个 chunk = 一个引擎 step；server 流式下一个 chunk
    # 基本也对应一个 decode step。不开投机解码时应 ≈1，开了之后 ≈ 平均每步接受长度。
    n_chunks = sum(len(r.get("chunks") or []) for r in ok)
    tokens_per_chunk = (total_out - len(ok)) / (n_chunks - len(ok)) if n_chunks > len(ok) else None

    len_mismatch = sum(1 for r in ok if r.get("expected_output_len") is not None and r["output_len"] != r["expected_output_len"])
    steady = _steady_output_throughput(ok, steady_trim)

    return {
        "num_requests": len(records),
        "num_failed": failed,
        "num_output_len_mismatch": len_mismatch,
        "num_gpus": num_gpus,
        "duration_s": duration,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "request_throughput": len(ok) / duration,
        "input_throughput": total_in / duration,
        "output_throughput": total_out / duration,
        "output_throughput_per_gpu": total_out / duration / num_gpus,
        "total_throughput_per_gpu": (total_in + total_out) / duration / num_gpus,
        "steady_output_throughput": steady,
        "steady_output_throughput_per_gpu": steady / num_gpus if steady is not None else None,
        "ttft_ms": _stats(ttft, 1000.0),
        "tpot_ms": _stats(tpot, 1000.0),
        "e2e_latency_s": _stats(e2e),
        "user_tokens_per_s": _stats(user_speed),
        "tokens_per_chunk": tokens_per_chunk,
    }


def append_result(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def summary_line(config: str, concurrency: int, m: dict[str, Any]) -> str:
    if "ttft_ms" not in m:
        return f"[{config}] c={concurrency} ALL {m.get('num_failed')} REQUESTS FAILED"
    return (
        f"[{config}] c={concurrency} out_tok/s/gpu={fmt(m['output_throughput_per_gpu'])} "
        f"steady={fmt(m['steady_output_throughput_per_gpu'])} "
        f"TTFT p50={fmt(m['ttft_ms']['p50'], 0)}ms TPOT p50={fmt(m['tpot_ms']['p50'])}ms "
        f"user_tok/s p50={fmt(m['user_tokens_per_s']['p50'])} tok/chunk={fmt(m['tokens_per_chunk'], 2)} "
        f"failed={m['num_failed']} len_mismatch={m['num_output_len_mismatch']}"
    )


def fmt(x: float | None, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"
