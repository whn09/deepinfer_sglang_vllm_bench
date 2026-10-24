"""Effective decode step time vs per-rank running requests: DeepInfer stats_logger vs SGLang 'Decode batch' lines.
DeepInfer: step = 1000/avg_SPS (window incl. other ranks' prefill stalls) and pure = 1000/cur_SPS.
SGLang: step = running * accept_len / gen_throughput (interval incl. prefill batches)."""
import re, sys, statistics as st, collections
def di(path):
    b = collections.defaultdict(lambda: ([], []))
    for line in open(path, errors="ignore"):
        if "[stats]" not in line: continue
        m = {k: float(v) for k, v in re.findall(r"(cur_SPS|avg_SPS|running) ([0-9.]+)", line)}
        if m.get("running", 0) < 1 or m.get("avg_SPS", 0) <= 0: continue
        r = int(m["running"]); k = min(r // 4 * 4, 32)
        b[k][0].append(1000 / m["avg_SPS"]); b[k][1].append(1000 / m["cur_SPS"] if m["cur_SPS"] < 300 else float("nan"))
    return b
def sg(path):
    b = collections.defaultdict(list)
    for line in open(path, errors="ignore"):
        if "Decode batch" not in line: continue
        r = re.search(r"#running-req: (\d+)", line); al = re.search(r"accept len: ([0-9.]+)", line); th = re.search(r"gen throughput \(token/s\): ([0-9.]+)", line)
        if not (r and al and th) or float(th.group(1)) < 50: continue
        n = int(r.group(1)); k = min(n // 4 * 4, 32)
        b[k].append(1000 * n * float(al.group(1)) / float(th.group(1)))
    return b
d, s = di(sys.argv[1]), sg(sys.argv[2])
print("running/rank | DeepInfer eff step ms (n) | DeepInfer pure decode ms | SGLang eff step ms (n)")
for k in sorted(set(d) | set(s)):
    de, dp = d.get(k, ([], []))
    dp = [x for x in dp if x == x]
    f = lambda v: f"{st.median(v):6.1f} ({len(v)})" if v else "   -   "
    print(f"{k:3d}-{k+3:<3d}       | {f(de):>14} | {f(dp):>14} | {f(s.get(k, [])):>14}")
