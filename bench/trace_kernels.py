"""Kernel time inside DeepInfer decode forward windows (gpu_user_annotation launch_forward_N*_B<small>), averaged over
the profiled steps; compares two traces. Kernel names are shortened (template args dropped)."""
import collections, json, re, sys

def load(path):
    ev = json.load(open(path))["traceEvents"]
    wins = [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") == "gpu_user_annotation"
            and re.match(r"deepinfer:(launch_forward_N\d+_B(\d+)|launch_sample|launch_dflash_decode_forward)$", e["name"])
            and not e["name"].endswith("B32768")]
    fw = [w for w in wins if "launch_forward" in w[2]]
    steps = len(fw)
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    by = collections.Counter(); win_time = collections.Counter(); busy = collections.Counter()
    for a, b, n in wins:
        tag = "forward" if "launch_forward" in n else n.split(":")[1]
        win_time[tag] += (b - a)
        for k in ks:
            if k["ts"] >= a and k["ts"] < b:
                nm = re.sub(r"<.*", "", k["name"]); nm = re.sub(r"\(.*", "", nm)[:70]
                by[(tag, nm)] += k["dur"]; busy[tag] += k["dur"]
    return steps, by, win_time, busy

(sa, a, wa, ba), (sb, b, wb, bb) = load(sys.argv[1]), load(sys.argv[2])
print(f"steps {sa} vs {sb}")
for tag in ("forward", "launch_sample", "launch_dflash_decode_forward"):
    print(f"{tag:30s} window {wa[tag]/sa/1000:6.2f} vs {wb[tag]/sb/1000:6.2f} ms   kernel-busy {ba[tag]/sa/1000:6.2f} vs {bb[tag]/sb/1000:6.2f} ms")
keys = set(a) | set(b)
rows = sorted(keys, key=lambda k: -(b.get(k, 0) / sb))
print(f"\n{'kernel (forward/sample/dflash)':80s} {'A ms/step':>9s} {'B ms/step':>9s} {'B-A':>7s}")
for k in rows[:40]:
    x, y = a.get(k, 0) / sa / 1000, b.get(k, 0) / sb / 1000
    print(f"{k[0][:8]:8s} {k[1]:71s} {x:9.3f} {y:9.3f} {y-x:7.3f}")
