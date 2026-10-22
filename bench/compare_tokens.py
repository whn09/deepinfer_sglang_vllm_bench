"""比较两次 run_deepinfer.py --dump-tokens 的逐请求输出 token 是否一致（例如 YoCo 截断开/关，#16）。

  python bench/compare_tokens.py <records_a.jsonl> <records_b.jsonl>
"""

from __future__ import annotations

import json
import sys


def load(path: str) -> dict[int, list[int]]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["id"]] = r.get("tokens") or []
    return out


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(a) & set(b))
    same = 0
    first_diff = []
    for i in ids:
        if a[i] == b[i]:
            same += 1
            continue
        n = next((k for k, (x, y) in enumerate(zip(a[i], b[i])) if x != y), min(len(a[i]), len(b[i])))
        first_diff.append((i, n, len(a[i])))
    print(f"requests compared: {len(ids)}  identical: {same}  differ: {len(first_diff)}")
    for i, n, total in first_diff[:10]:
        print(f"  request {i}: first differing token at position {n} of {total}")


if __name__ == "__main__":
    main()
