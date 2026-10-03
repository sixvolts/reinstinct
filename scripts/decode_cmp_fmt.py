#!/usr/bin/env python3
"""Per-token kernel time by weight format/role, fork vs reinstinct.
usage: cmp_fmt.py fork.csv ri.csv steps fork_gpu_ms ri_gpu_ms"""
import csv, re, sys, collections, statistics
BIG = 15.0
def key(n):
    m = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+)", n)
    n = re.sub(r"\(.*", "", n)
    if m: return {"23": "iq4_xs", "2": "q4_0", "8": "q8_0", "12": "q4_k", "13": "q5_k", "14": "q6_k", "11": "q3_k"}.get(m[1], "t" + m[1]) + " matvec (generic)"
    for f in ["iq4xs", "iq4_xs", "q4_0", "q8_0", "q3k", "q4k", "q5k", "q6k", "q5_1", "f32", "f16"]:
        if f in n.lower() and ("mat_vec" in n or "matvec" in n or "glu" in n or "gate_up" in n):
            f = f.replace("iq4xs", "iq4_xs").replace("k", "_k") if f not in ("f32", "f16") else f
            moe = "moe" in n.lower() or "<true" in n or "glu<true" in n or "_seg" in n
            return f"{f} {'MoE ' if moe else ''}matvec"
    if re.search(r"fattn|flash_attn|attn_", n): return "attention"
    if re.search(r"gated_delta|gdn|ssm_|conv1d", n): return "GDN/conv"
    if re.search(r"topk|shexp_gate", n): return "router"
    return "glue"
def load(p, steps):
    d = collections.defaultdict(list)
    for x in csv.DictReader(open(p)):
        n = x["Kernel_Name"]
        if "rocclr" in n: continue
        d[n].append((int(x["End_Timestamp"]) - int(x["Start_Timestamp"])) / 1e3)
    big = collections.defaultdict(lambda: [0.0, 0.0]); small = [0.0, 0.0]
    for n, v in d.items():
        k = key(n)
        if statistics.median(v) >= BIG: big[k][0] += sum(v) / steps / 1e3; big[k][1] += len(v) / steps
        else: small[0] += len(v) / steps; small[1] += sum(v) / steps / 1e3
    return big, small
steps = float(sys.argv[3]); gf, gr = float(sys.argv[4]), float(sys.argv[5])
bf, sf = load(sys.argv[1], steps); br, sr = load(sys.argv[2], steps)
print(f"{'big kernels (traced ms/token, calls)':40s} {'fork':>14s} {'reinstinct':>14s} {'delta ms':>9s}")
for k in sorted(set(bf) | set(br), key=lambda k: -(bf[k][0] - br[k][0])):
    print(f"{k:40s} {bf[k][0]:8.3f} ({bf[k][1]:3.0f}) {br[k][0]:8.3f} ({br[k][1]:3.0f}) {bf[k][0]-br[k][0]:9.3f}")
tf, tr = sum(v[0] for v in bf.values()), sum(v[0] for v in br.values())
print(f"{'big total':40s} {tf:14.3f} {tr:14.3f} {tf-tr:9.3f}")
print(f"{'GPU time/token (untraced graph replay)':40s} {gf:14.3f} {gr:14.3f} {gf-gr:9.3f}")
print(f"{'rest = small kernels + GPU dispatch':40s} {gf-tf:14.3f} {gr-tr:14.3f} {(gf-tf)-(gr-tr):9.3f}")
print(f"{'  small kernels per token':40s} {sf[0]:14.0f} {sr[0]:14.0f}")
print(f"{'  -> per small kernel (us)':40s} {1e3*(gf-tf)/sf[0]:14.2f} {1e3*(gr-tr)/sr[0]:14.2f}")
