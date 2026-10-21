"""AgentX 结果按 InferenceX 的口径算指标，并和 InferenceX 公布的数据点对比（#4）。

指标直接调用 InferenceX 自己的代码（infx.results.agentic：load_records_with_accounting +
compute_request_metrics），保证和 dashboard 同一套算法：
  每卡吞吐 = (输入 + 输出 token) / (最晚 request_end - 最早 request_start) / GPU 数
  interactivity = 1 / ITL（各分位）

用法（INFERENCEX_E2E 指向 InferenceX 仓库的 inferencex-e2e 目录）：
  INFERENCEX_E2E=/path/InferenceX/inferencex-e2e python bench/agentx_metrics.py \
      <aiperf artifact dir> --num-gpus 4 --conc 8 --topology tp4 \
      --published results/inferencex_public/dsv41flash_b200_vllm_agentic.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

KEYS = [
    ("total_tput_per_gpu", "total tok/s/GPU"),
    ("output_tput_per_gpu", "output tok/s/GPU"),
    ("median_intvty", "intvty p50"),
    ("p90_intvty", "intvty p90"),
    ("median_ttft", "TTFT p50 s"),
    ("p90_e2el", "E2EL p90 s"),
    ("duration_seconds", "duration s"),
]


def find_artifact_dir(root: Path) -> Path:
    hit = next(root.rglob("profile_export.jsonl"), None)
    if hit is None:
        raise SystemExit(f"no profile_export.jsonl under {root}")
    return hit.parent


def ours(artifact_dir: Path, num_gpus: int) -> dict:
    sys.path.insert(0, os.environ["INFERENCEX_E2E"])
    from infx.results.agentic.artifacts import load_records_with_accounting  # noqa: E402
    from infx.results.agentic.request_metrics import compute_request_metrics  # noqa: E402

    records, accounting = load_records_with_accounting(artifact_dir / "profile_export.jsonl")
    agg_path = artifact_dir / "profile_export_aiperf.json"
    aggregate = json.loads(agg_path.read_text()) if agg_path.exists() else {}
    flat, _ = compute_request_metrics(records, aggregate)
    out = dict(flat)
    for k in ("input_tput_tps", "output_tput_tps", "total_tput_tps"):
        if k in flat:
            out[k.replace("_tps", "_per_gpu")] = flat[k] / num_gpus
    out["num_records"] = len(records)
    out["request_accounting"] = accounting if isinstance(accounting, dict) else str(accounting)
    if aggregate:
        err = (aggregate.get("error_request_count") or {}).get("avg") or 0
        done = (aggregate.get("completed_request_count") or {}).get("avg") or 0
        out["error_rate"] = err / done if done else None
    return out


def published(path: Path, topology: str, conc: int) -> dict | None:
    rows = json.loads(path.read_text())["rows"]
    want_dpa = topology.startswith("dep")
    tp = int("".join(c for c in topology if c.isdigit()))
    for r in rows:
        if r["conc"] == conc and r["decode_tp"] == tp and bool(r["decode_dp_attention"]) == want_dpa:
            m = dict(r["metrics"])
            m["total_tput_per_gpu"] = m["input_tput_per_gpu"] + m["output_tput_per_gpu"]
            m["_id"], m["_date"], m["_image"] = r["id"], r["date"], r["image"]
            return m
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("artifacts")
    p.add_argument("--num-gpus", type=int, required=True)
    p.add_argument("--conc", type=int, required=True)
    p.add_argument("--topology", default="tp4", help="tp4 / tp2 / dep2 / dep4，用来在公开数据里找对应的点")
    p.add_argument("--published", default=None)
    p.add_argument("--out", default=None, help="把我们的指标写成 json")
    args = p.parse_args()

    art = find_artifact_dir(Path(args.artifacts))
    m = ours(art, args.num_gpus)
    if args.out:
        Path(args.out).write_text(json.dumps({"artifact_dir": str(art), "conc": args.conc, "topology": args.topology, **m}, indent=1, default=str))
    pub = published(Path(args.published), args.topology, args.conc) if args.published else None
    print(f"artifacts: {art}  records={m.get('num_records')}  error_rate={m.get('error_rate')}")
    print(f"{'metric':<20}{'ours':>14}{'InferenceX':>14}{'ours/IX':>10}")
    for k, label in KEYS:
        a = m.get(k)
        b = pub.get(k) if pub else None
        ratio = f"{a / b:.3f}" if a is not None and b else "-"
        fa = f"{a:.2f}" if isinstance(a, (int, float)) else "-"
        fb = f"{b:.2f}" if isinstance(b, (int, float)) else "-"
        print(f"{label:<20}{fa:>14}{fb:>14}{ratio:>10}")
    if pub:
        print(f"published point id={pub['_id']} date={pub['_date']} image={pub['_image']}")


if __name__ == "__main__":
    main()
