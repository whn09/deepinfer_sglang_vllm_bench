"""AgentX 汇总：我们测的各配置 + InferenceX 公开点 → markdown 表和 Pareto 图（#7）。

每个配置一条曲线（每个并发一个点）；横轴 interactivity p50（tok/s/user），纵轴每卡总吞吐（tok/s/GPU），
两个方向都越大越好。指标来自 run_agentx.sh 写的 metrics.json（InferenceX 自己的算法）。

  python bench/agentx_report.py results/b200_live/results/agentx \
      --series "DeepInfer EP4=deepinfer-ix-ep4-noeplb:dep4" --series "SGLang (ours)=sglang-ix" --series "vLLM TP4 (ours)=vllm-ix-tp4" \
      --published "vLLM (InferenceX)=results/inferencex_public/dsv41flash_b200_vllm_agentic.json" \
      --published "SGLang (InferenceX)=results/inferencex_public/dsv41flash_b200_sglang_agentic.json" \
      --out-dir results/20261022_agentx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize import pareto_front

# (metrics.json 的名字, 公开 API 的名字, 表头)
COLS = [
    ("total_tput_per_gpu", "tput_per_gpu", "tok/s/GPU"),
    ("output_tput_per_gpu", "output_tput_per_gpu", "out tok/s/GPU"),
    ("p50_intvty", "median_intvty", "intvty p50"),
    ("p50_ttft", "median_ttft", "TTFT p50 s"),
    ("p90_ttft", "p90_ttft", "TTFT p90 s"),
    ("p50_e2el", "median_e2el", "E2EL p50 s"),
]


def load_ours(root: Path, config: str, topo: str | None) -> list[dict]:
    rows = []
    for f in sorted((root / config).glob("c*/metrics.json")):
        if not f.parent.name[1:].isdigit():  # c128-oom-invalid 之类作废的目录
            continue
        m = json.loads(f.read_text())
        rows.append({"conc": m["conc"], "topo": topo or m.get("topology", ""), **{k: m.get(k) for k, _, _ in COLS}})
    return sorted(rows, key=lambda r: r["conc"])


def load_published(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        d = next(v for v in d.values() if isinstance(v, list))
    rows = []
    for r in d:
        topo = f"{'dep' if r['decode_dp_attention'] else 'tp'}{r['num_decode_gpu']}"
        rows.append({"conc": r["conc"], "topo": topo, **{k: r["metrics"].get(api) for k, api, _ in COLS}})
    return sorted(rows, key=lambda r: (r["topo"], r["conc"]))


def fmt(v) -> str:
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 100 else f"{v:.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("agentx_root", help="results/agentx 目录（每个配置一个子目录）")
    p.add_argument("--series", action="append", default=[], help="'显示名=配置目录名[:拓扑]'（拓扑缺省取 metrics.json，即对比口径）")
    p.add_argument("--published", action="append", default=[], help="'显示名=InferenceX 公开数据 json'")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    root, out = Path(a.agentx_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    series: dict[str, list[dict]] = {}
    for s in a.series:
        name, cfg = s.split("=", 1)
        cfg, _, topo = cfg.partition(":")
        series[name] = load_ours(root, cfg, topo or None)
    for s in a.published:
        name, path = s.split("=", 1)
        series[name] = load_published(Path(path))

    lines = ["| system | conc | topo | " + " | ".join(h for _, _, h in COLS) + " |", "|---" * (3 + len(COLS)) + "|"]
    for name, rows in series.items():
        for r in rows:
            lines.append(f"| {name} | {r['conc']} | {r['topo']} | " + " | ".join(fmt(r[k]) for k, _, _ in COLS) + " |")
    (out / "agentx_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, rows in series.items():
        pts = [(r["p50_intvty"], r["total_tput_per_gpu"], str(r["conc"])) for r in rows if r["p50_intvty"] and r["total_tput_per_gpu"]]
        if not pts:
            continue
        dashed = "InferenceX" in name
        all_pts = sorted(pts)
        sc = ax.scatter([x for x, _, _ in all_pts], [y for _, y, _ in all_pts], s=18, alpha=0.5, marker="x" if dashed else "o")
        front = sorted(pareto_front(pts))
        ax.plot([x for x, _, _ in front], [y for _, y, _ in front], "--" if dashed else "-", color=sc.get_facecolor()[0], label=name)
        for x, y, tag in front:
            ax.annotate(tag, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("interactivity p50 (tok/s/user)")
    ax.set_ylabel("total throughput (tok/s/GPU)")
    ax.set_yscale("log")
    ax.set_title("DeepSeek-V4.1-Flash on B200, AgentX (semianalysis_cc_traces), Pareto envelope per system")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "agentx_pareto.png", dpi=130)

    # TTFT p50 vs concurrency: the Pareto plot above has no TTFT axis. Published series mix topologies at the same
    # concurrency (e.g. vLLM tp4 and dep2 at c=8), so they get one line per topology.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rows in series.items():
        dashed = "InferenceX" in name
        topos = sorted({r["topo"] for r in rows}) if dashed else [None]
        for topo in topos:
            sel = sorted((r for r in rows if (topo is None or r["topo"] == topo) and r["p50_ttft"]), key=lambda r: r["conc"])
            if sel:
                label = f"{name} {topo}" if dashed else name
                ax.plot([r["conc"] for r in sel], [r["p50_ttft"] for r in sel], "x--" if dashed else "o-", ms=4, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("TTFT p50 (s)")
    ax.set_title("DeepSeek-V4.1-Flash on B200, AgentX: TTFT p50 by concurrency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "agentx_ttft.png", dpi=130)
    print(f"wrote {out / 'agentx_table.md'}, {out / 'agentx_pareto.png'} and {out / 'agentx_ttft.png'}")


if __name__ == "__main__":
    main()
