"""从 InferenceX 的 srt-slurm recipe 里取出某个并发点的完整服务配置（#4）。

不手抄参数：直接读 InferenceX 仓库里的 agentic.yaml，按 srtctl 的规则展开：
  - base 与 override_* 深度合并（dict 递归合并，标量/列表替换，override 里值为 null 的键删除）
  - zip_override_*：与 benchmark.env.CONC 等长的列表按下标展开成一个个变体
然后挑出 benchmark.env.CONC 等于目标并发的那个变体（同一并发有多个拓扑时用 --topology 选）。

  python bench/ix_recipe.py <agentic.yaml> --conc 128 [--topology dep4] [--format json|cli|env]

输出 json：{"variant", "gpus", "args", "env", "frontend", "setup_script"}；--format cli 输出可直接传给
server 的命令行参数（每行一个），--format env 输出 KEY=VALUE。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if v is None:
                out.pop(k, None)
            elif k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    return copy.deepcopy(override)


def zip_variants(override: dict[str, Any]) -> list[dict[str, Any]]:
    concs = (((override.get("benchmark") or {}).get("env")) or {}).get("CONC")
    n = len(concs) if isinstance(concs, list) else 1

    def pick(node: Any, i: int) -> Any:
        if isinstance(node, dict):
            return {k: pick(v, i) for k, v in node.items()}
        if isinstance(node, list) and len(node) == n and n > 1:
            return node[i]
        return node

    return [pick(override, i) for i in range(n)]


def variants(recipe: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    base = recipe["base"]
    out = []
    for name, ov in recipe.items():
        if name == "base" or not isinstance(ov, dict):
            continue
        if name.startswith("zip_override_"):
            for v in zip_variants(ov):
                out.append((name, deep_merge(base, v)))
        elif name.startswith("override_"):
            out.append((name, deep_merge(base, ov)))
    return out


def select(recipe: dict[str, Any], conc: int, topology: str | None) -> tuple[str, dict[str, Any]]:
    hits = []
    for name, cfg in variants(recipe):
        c = str((((cfg.get("benchmark") or {}).get("env")) or {}).get("CONC"))
        if c == str(conc) and (topology is None or topology in name):
            hits.append((name, cfg))
    if not hits:
        raise SystemExit(f"no variant with CONC={conc}" + (f" and topology '{topology}'" if topology else ""))
    if len(hits) > 1:
        names = ", ".join(n for n, _ in hits)
        raise SystemExit(f"CONC={conc} matches several variants ({names}); pass --topology")
    return hits[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("recipe")
    p.add_argument("--conc", type=int, required=True)
    p.add_argument("--topology", default=None, help="变体名里的子串，如 tp4 / dep4 / dep2")
    p.add_argument("--format", choices=["json", "cli", "env"], default="json")
    p.add_argument("--skip-arg", action="append", default=["served-model-name"], help="不输出的参数（由调用方自己设置）")
    p.add_argument("--engine", choices=["vllm", "sglang"], default=None)
    p.add_argument(
        "--synthetic-al",
        type=float,
        default=None,
        help="按 InferenceX infx/srt_slurm/synthetic_acceptance.py 注入 golden 接受长度（只对开了投机解码的配置）："
        "vLLM 改写 speculative-config 为 rejection_sample_method=synthetic + synthetic_acceptance_length；"
        "SGLang 加 SGLANG_SIMULATE_ACC_LEN / METHOD=match-expected / TOKEN_MODE=real-draft-token",
    )
    args = p.parse_args()

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    name, cfg = select(recipe, args.conc, args.topology)
    role = cfg["roles"]["agg"]
    if args.synthetic_al is not None:
        ra = role.setdefault("args", {})
        if args.engine == "vllm" and "speculative-config" in ra:
            spec = json.loads(ra["speculative-config"])
            spec.update(rejection_sample_method="synthetic", synthetic_acceptance_length=args.synthetic_al)
            ra["speculative-config"] = json.dumps(spec)
        elif args.engine == "sglang" and ra.get("speculative-algorithm"):
            env = role.setdefault("env", {})
            env.update(
                SGLANG_SIMULATE_ACC_LEN=f"{args.synthetic_al:g}",
                SGLANG_SIMULATE_ACC_METHOD="match-expected",
                SGLANG_SIMULATE_ACC_TOKEN_MODE="real-draft-token",
            )
    result = {
        "variant": name,
        "gpus": role.get("gpus"),
        "args": role.get("args") or {},
        "env": role.get("env") or {},
        "frontend": cfg.get("frontend") or {},
        "setup_script": cfg.get("setup_script"),
        "benchmark_env": (cfg.get("benchmark") or {}).get("env") or {},
    }
    if args.format == "json":
        print(json.dumps(result, indent=1))
    elif args.format == "env":
        for k, v in result["env"].items():
            print(f"{k}={v}")
    else:
        for k, v in result["args"].items():
            if k in args.skip_arg or v is False:
                continue
            print(f"--{k}")
            if v is not True:
                print(v if isinstance(v, str) else json.dumps(v))


if __name__ == "__main__":
    sys.exit(main())
