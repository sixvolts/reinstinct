#!/usr/bin/env python3
"""Aggregate a rocprofv2 kernel trace into time per step by kernel family.

Usage:
  kernel_families.py results_tr.csv --steps N            # N decode steps / forwards in the trace
  kernel_families.py results_tr.csv --marker REGEX --per-step K
      # steps = (calls matching REGEX) / K, e.g. --marker topk_moe --per-step 40 for a 40-MoE-layer model
  kernel_families.py a.csv b.csv --steps 16 16            # side by side (fork vs reinstinct)

Families are matched in order; the first regex that hits a kernel name wins.
Covers ggml (llama fork) kernel names and common reinstinct names; extend FAMILIES
if a family shows up under "rest" with real time in it.
"""
import argparse, csv, re, collections, sys

FAMILIES = [
    ("GDN / linear attention",        r"gated_delta|delta_net|gdn|ssm_|conv1d|ssm_conv"),
    ("attention",                     r"fattn|flash_attn|attn_decode|attn_partial|attn_merge|softmax"),
    ("MoE router / top-k",            r"topk_moe|argsort|router|topk"),
    ("MoE gate/up (decode matvec)",   r"_glu|glu16|mul_mat_vec_q_moe|moe.*gate|expert.*gate"),
    ("MoE down (decode matvec)",      r"q5_1_repacked_seg|moe.*down|expert.*down"),
    ("MoE expert GEMM (prefill)",     r"_id_w|repacked<true|mmq.*_id|grouped|expert_gemm|mm_ids"),
    ("LM head",                       r"lm_head|output_proj|logits"),
    ("dense quant GEMM (MMQ)",        r"mmq|mul_mat_q|gemm_q|tile_gemm"),
    ("dense quant matvec",            r"mul_mat_vec_q|mul_mat_vec_kq|matvec|_repacked|dmmv|vec_dot"),
    ("f32/f16 matmul",                r"mul_mat_vec_f|matvec_f32|gcn_f32_gemm|gemm_f16|gemm_f|Cijk|rocblas"),
    ("norm / elementwise / copies",   r"norm|bin_bcast|unary|silu|gelu|scale|cpy|concat|get_rows|quantize|rope|add|mul\b|hc_"),
]

def load(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        n = r.get("Kernel_Name") or r.get("KernelName") or ""
        if "rocclr" in n:
            continue
        try:
            d = (float(r.get("End_Timestamp") or r["EndNs"]) - float(r.get("Start_Timestamp") or r["BeginNs"])) / 1e6
        except Exception:
            continue
        out.append((n, d))
    return out

def families(rows):
    fam = collections.defaultdict(float); top = collections.defaultdict(float)
    for n, d in rows:
        for name, rx in FAMILIES + [("rest", r".")]:
            if re.search(rx, n, re.I):
                fam[name] += d
                if name == "rest":
                    top[re.sub(r"\(.*", "", n)[:70]] += d
                break
    return fam, top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--steps", type=float, nargs="*")
    ap.add_argument("--marker"); ap.add_argument("--per-step", type=float, default=1.0)
    a = ap.parse_args()
    res = []
    for i, p in enumerate(a.csv):
        rows = load(p)
        if a.steps:
            steps = a.steps[i] if i < len(a.steps) else a.steps[-1]
        elif a.marker:
            steps = sum(1 for n, _ in rows if re.search(a.marker, n)) / a.per_step
        else:
            sys.exit("give --steps or --marker")
        fam, top = families(rows)
        res.append(({k: v / steps for k, v in fam.items()}, top, steps, p))
    names = [n for n, _ in FAMILIES] + ["rest"]
    print("ms per step (kernel time; summed over GPUs if several)")
    print("%-32s" % "family" + "".join("%12s" % ("file%d" % i) for i in range(len(res))))
    for n in names:
        print("%-32s" % n + "".join("%12.3f" % r[0].get(n, 0.0) for r in res))
    print("%-32s" % "TOTAL" + "".join("%12.3f" % sum(r[0].values()) for r in res))
    for i, (_, top, steps, p) in enumerate(res):
        print(f"file{i}: {p}  steps {steps:g}")
        for k, v in sorted(top.items(), key=lambda x: -x[1])[:5]:
            print(f"    rest: {v/steps:8.3f} ms  {k}")

if __name__ == "__main__":
    main()
