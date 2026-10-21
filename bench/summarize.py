"""汇总 results.jsonl：每个场景一张 markdown 表 + 一张吞吐-交互性 Pareto 图。

场景按 (数据来源, 输入长度, 输出长度) 分组——同一场景不同并发点用的是不同 seed 的数据集，
文件名不同但属于同一条曲线。

图：横轴 = 单用户解码速度 p50（tokens/s/user，越右越"跟手"），纵轴 = 每卡输出吞吐
（tokens/s/GPU）。和 InferenceX 的 interactivity 视图同一种坐标。

用法：python bench/summarize.py results/results.jsonl [--out-dir results/]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import fmt


def scenario_key(r: dict) -> str:
    m = r.get("dataset_meta") or {}
    if m:
        return f"{m.get('source')}_in{m.get('input_len')}_out{m.get('output_len')}"
    return Path(r["dataset"]).stem


def pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """两个方向都越大越好（交互性、每卡吞吐）：保留没有被任何点同时在两个方向上超过的点。"""
    front = []
    best_y = float("-inf")
    for x, y, tag in sorted(points, key=lambda p: (-p[0], -p[1])):
        if y > best_y:
            front.append((x, y, tag))
            best_y = y
    return front


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--envelope", action="store_true", help="每家只画其所有配置合起来的 Pareto 包络线（各家能做到的最好水平）")
    p.add_argument("--ttft-max", type=float, default=None, help="只保留 TTFT p50 ≤ 这个值（ms）的点，即在 TTFT SLO 下比较")
    args = p.parse_args()
    out_dir = Path(args.out_dir or Path(args.results).parent)

    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    by_scn: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scn[scenario_key(r)].append(r)

    md = []
    for scn, rs in sorted(by_scn.items()):
        rs.sort(key=lambda r: (r["system"], r["config"], r["concurrency"]))
        md.append(f"## {scn}\n")
        md.append(
            "| 系统 | 配置 | GPU | 并发 | 输出 tok/s/GPU | 稳态 tok/s/GPU | TTFT p50 (ms) | TTFT p99 (ms) "
            "| TPOT p50 (ms) | TPOT p99 (ms) | 单用户 tok/s p50 | 平均每步 token | 失败 | 长度不符 |"
        )
        md.append("|" + "---|" * 14)
        for r in rs:
            if "ttft_ms" not in r:
                md.append(f"| {r['system']} | {r['config']} | {r.get('num_gpus', '-')} | {r['concurrency']} | 全部失败 |" + " |" * 9)
                continue
            md.append(
                f"| {r['system']} | {r['config']} | {r['num_gpus']} | {r['concurrency']} "
                f"| {fmt(r['output_throughput_per_gpu'])} | {fmt(r.get('steady_output_throughput_per_gpu'))} "
                f"| {fmt(r['ttft_ms']['p50'], 0)} | {fmt(r['ttft_ms']['p99'], 0)} "
                f"| {fmt(r['tpot_ms']['p50'], 2)} | {fmt(r['tpot_ms']['p99'], 2)} "
                f"| {fmt(r['user_tokens_per_s']['p50'])} | {fmt(r.get('tokens_per_chunk'), 2)} "
                f"| {r['num_failed']} | {r.get('num_output_len_mismatch', '-')} |"
            )
        md.append("")
    report = out_dir / "summary.md"
    report.write_text("\n".join(md))
    print(report.read_text())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("没装 matplotlib，跳过画图")
        return

    for scn, rs in by_scn.items():
        curves: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for r in rs:
            if "ttft_ms" not in r or r["user_tokens_per_s"]["p50"] is None:
                continue
            if args.ttft_max is not None and r["ttft_ms"]["p50"] > args.ttft_max:
                continue
            key = r["system"] if args.envelope else f"{r['system']} / {r['config']}"
            tag = f"{r['config'].removeprefix(r['system'] + '-')} c{r['concurrency']}" if args.envelope else f"c{r['concurrency']}"
            curves[key].append((r["user_tokens_per_s"]["p50"], r["output_throughput_per_gpu"], tag))
        if args.envelope:
            curves = {k: pareto_front(v) for k, v in curves.items()}
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for label, pts in sorted(curves.items()):
            pts.sort()
            xs, ys, cs = zip(*pts)
            ax.plot(xs, ys, marker="o", label=label)
            for x, y, tag in pts:
                ax.annotate(tag, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Interactivity: tokens/s/user (p50)")
        ax.set_ylabel("Output throughput: tokens/s/GPU")
        title = scn + (" — Pareto envelope per system" if args.envelope else "")
        if args.ttft_max is not None:
            title += f", TTFT p50 ≤ {args.ttft_max:g} ms"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        suffix = ("_envelope" if args.envelope else "") + (f"_ttft{args.ttft_max:g}" if args.ttft_max is not None else "")
        png = out_dir / f"pareto_{scn}{suffix}.png"
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
