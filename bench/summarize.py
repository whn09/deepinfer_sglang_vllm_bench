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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--out-dir", default=None)
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
        curves: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
        for r in rs:
            if "ttft_ms" in r and r["user_tokens_per_s"]["p50"] is not None:
                curves[f"{r['system']} / {r['config']}"].append(
                    (r["user_tokens_per_s"]["p50"], r["output_throughput_per_gpu"], r["concurrency"])
                )
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for label, pts in sorted(curves.items()):
            pts.sort()
            xs, ys, cs = zip(*pts)
            ax.plot(xs, ys, marker="o", label=label)
            for x, y, c in pts:
                ax.annotate(f"c{c}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Interactivity: tokens/s/user (p50)")
        ax.set_ylabel("Output throughput: tokens/s/GPU")
        ax.set_title(scn)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = out_dir / f"pareto_{scn}.png"
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
