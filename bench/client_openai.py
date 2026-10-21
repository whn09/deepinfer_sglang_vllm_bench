"""SGLang / vLLM 的压测 client：OpenAI /v1/completions 流式接口，闭环固定并发。

闭环并发 C：始终保持 C 个请求在飞，一个结束立刻补下一个，和 run_deepinfer.py 的
调度方式完全一致。只依赖标准库 + requests，在 DeepInfer venv 里就能跑，
不需要进 sglang/vllm 的容器。

请求体：prompt 直接给 token ids，max_tokens=output_len，ignore_eos=True，
采样参数三家统一（默认 temperature=0）。用 stream_options.include_usage 拿服务端
统计的真实 prompt/completion token 数。

用法：
  python bench/client_openai.py --backend sglang --base-url http://127.0.0.1:30000 \
      --dataset data/sg_1k_1k_s1.jsonl --concurrency 32 --num-gpus 4 \
      --config sglang-ref --out results/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

from common import append_result, compute_metrics, load_dataset, load_dataset_meta, summary_line, write_records


def get_model_id(base_url: str) -> str:
    r = requests.get(f"{base_url}/v1/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def scrape_metrics(base_url: str) -> dict[str, float]:
    """抓 /metrics 里跟投机解码有关的计数器（名字带 spec / accept / draft 的都留下）。

    vLLM 的 spec decode 计数器默认就有；SGLang 要加 --enable-metrics 才暴露 /metrics。
    各版本的指标名会变，这里不写死名字，原样存下来，交给 summarize 或人工看。
    """
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        r.raise_for_status()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#") or not re.search(r"spec|accept|draft", line, re.I):
            continue
        try:
            name, value = line.rsplit(" ", 1)
            out[name] = float(value)
        except ValueError:
            continue
    return out


def one_request(session: requests.Session, url: str, model: str, item: dict[str, Any], sp: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": item["prompt_token_ids"],
        "max_tokens": item["output_len"],
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sp,
    }
    rec: dict[str, Any] = {
        "id": item["id"],
        "prompt_len": len(item["prompt_token_ids"]),
        "expected_output_len": item["output_len"],
        "chunks": [],
        "error": None,
    }
    rec["t_submit"] = time.time()
    usage = None
    try:
        with session.post(url, json=body, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                rec["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
                return rec
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = time.time()
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    # 一个 chunk 可能含多个 token（投机解码一次接受多个），token 数在流里拿不到，记 null
                    rec["chunks"].append([now, None])
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        return rec
    if not rec["chunks"]:
        rec["error"] = "no output chunks"
        return rec
    rec["t_first"] = rec["chunks"][0][0]
    rec["t_end"] = rec["chunks"][-1][0]
    if usage:
        rec["output_len"] = int(usage.get("completion_tokens", 0))
        rec["prompt_len"] = int(usage.get("prompt_tokens", rec["prompt_len"]))
    else:
        rec["output_len"] = len(rec["chunks"])
        rec["error"] = "no usage in stream (completion_tokens unknown)"
    return rec


def run_closed_loop(base_url: str, model: str, items: list[dict[str, Any]], concurrency: int, sp: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    url = f"{base_url}/v1/completions"
    q: queue.Queue[dict[str, Any]] = queue.Queue()
    for it in items:
        q.put(it)
    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker() -> None:
        session = requests.Session()
        while True:
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            rec = one_request(session, url, model, it, sp, timeout)
            with lock:
                records.append(rec)
                done = len(records)
            if done % max(1, len(items) // 10) == 0:
                print(f"  {done}/{len(items)} done", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(records, key=lambda r: r["id"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["sglang", "vllm"], required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--num-gpus", type=int, required=True, help="这个 server 实际占用的 GPU 数，用于归一化到每卡吞吐")
    p.add_argument("--config", required=True, help="配置名，如 sglang-ref / sglang-tp8-nospec，写进结果用于分组")
    p.add_argument("--out", required=True, help="结果 jsonl（追加）")
    p.add_argument("--records-dir", default=None, help="逐请求记录目录；默认在 --out 同级 records/")
    p.add_argument("--num-requests", type=int, default=None, help="只用数据集前 N 条")
    p.add_argument("--warmup-requests", type=int, default=None, help="正式测量前的预热请求数，默认 = concurrency")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--note", default="")
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")
    model = get_model_id(base_url)
    items = load_dataset(args.dataset, args.num_requests)
    meta = load_dataset_meta(args.dataset)
    sp = {"temperature": args.temperature, "top_p": args.top_p}

    # 预热：用数据集里的请求、短输出跑一轮，触发 cuda graph / autotune 等一次性开销。
    # 预热请求换掉前缀（逆序 token），避免和正式请求共享前缀命中缓存。
    n_warm = args.warmup_requests if args.warmup_requests is not None else args.concurrency
    if n_warm > 0:
        warm = [
            {"id": -1 - i, "prompt_token_ids": list(reversed(it["prompt_token_ids"])), "output_len": 16}
            for i, it in enumerate(items[:n_warm])
        ]
        print(f"[{args.config}] warmup {len(warm)} requests ...", flush=True)
        run_closed_loop(base_url, model, warm, min(args.concurrency, len(warm)), sp, args.timeout)

    m0 = scrape_metrics(base_url)
    print(f"[{args.config}] run {len(items)} requests @ concurrency {args.concurrency} ...", flush=True)
    records = run_closed_loop(base_url, model, items, args.concurrency, sp, args.timeout)
    m1 = scrape_metrics(base_url)
    # 计数器（*_total）看差值；SGLang 的 spec_accept_length / spec_accept_rate 是 gauge，差值没意义，看 after
    spec_delta = {k: m1[k] - m0.get(k, 0.0) for k in m1 if k.split("{")[0].endswith("_total") and m1[k] != m0.get(k, 0.0)}

    metrics = compute_metrics(records, args.num_gpus)
    result = {
        "system": args.backend,
        "config": args.config,
        "dataset": str(args.dataset),
        "dataset_meta": meta,
        "concurrency": args.concurrency,
        "sampling": sp,
        "server_metrics_delta": spec_delta,
        "server_metrics_after": m1,
        "note": args.note,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics,
    }
    append_result(args.out, result)
    rec_dir = Path(args.records_dir) if args.records_dir else Path(args.out).parent / "records"
    write_records(rec_dir / f"{args.config}_{Path(args.dataset).stem}_c{args.concurrency}.jsonl", records)

    print(summary_line(args.config, args.concurrency, metrics), flush=True)


if __name__ == "__main__":
    main()
