# R5 / R6: decode attribution, fork 905021dba + B1..R8 vs reinstinct ee7cbd5

Podcast GPU 1 (MI50), 2026-10-03. Qwen 3.8-27B Q4_K_XL (dense GDN), Gemma 4 31B QAT (Q4_0), Qwen 3.6-35B-A3B Q4_K_XL (MoE).

## Method

1. **Untraced GPU time per token.** A local-only, env-gated timer around the graph launch in each engine records events around `hipGraphLaunch` (GPU time per replay) and the host time from one launch to the next. 128 tokens from an empty context. The timer is not committed to either repo.
2. **Kernel traces.** rocprofv3 kernel trace, 17 forwards (fork: `llama-bench -p 0 -n 16 -r 1`, which includes the warm-up; reinstinct: `generate-text --tokens 1000 -n 16`). Both run with graphs off (`GGML_CUDA_DISABLE_GRAPHS=1` / `REINSTINCT_NO_GRAPH=1`) because both engines crash in graph replay under rocprofv3 on these models. Kernel durations don't depend on graphs.
3. **Profiler bias.** rocprofv3 records each kernel's isolated latency. A no-op kernel reads 5.0-5.8 us traced but costs 1.47 us back-to-back in a graph; a 10 us memory sweep reads 10.4 vs 10.2. So traced time is good for big kernels and inflated for small ones. In traces where the engine keeps the GPU fed, traced busy per token runs 1.1-1.5 us per kernel above the untraced GPU time (fork 27B 36.9 vs 35.1 ms, 35B 12.0 vs 10.5; reinstinct 27B 28.3 vs 28.1, 35B 8.5 vs 7.7).
4. **Avoid traces with idle gaps.** The earlier traces with reinstinct graphs on ran at 49 ms/token under the profiler (8 ms untraced). The GPU idled between kernels and durations inflated by about 5 us per kernel. That is why `R6-L8-attribution-qwen36-35B.md` showed kernel time "level" at ~12.6 ms. The reinstinct side of that note is wrong, and so is its conclusion that the decode gap is all glue.

Scripts: `cmp_fmt.py` (kernel time per token by weight format and role, traced) and the timer diffs. Both are in the reinstinct session scratchpad; ask if you want them in scripts/.

## Host vs GPU

| model | fork GPU / launch-to-launch | reinstinct GPU / launch-to-launch | GPU gap |
|---|---:|---:|---:|
| Qwen 3.8-27B | 35.12 / 35.73 ms | 28.07 / 28.37 ms | 7.05 ms |
| Gemma 31B QAT Q4_0 | 39.06 / 39.31 ms | 30.98 / 31.32 ms | 8.09 ms |
| Qwen 3.6-35B-A3B | 10.54 / 10.93 ms | 7.66 / 7.95 ms | 2.88 ms |

Host overhead is 0.25-0.6 ms per token in both engines. **The whole decode gap is GPU time in the graph**, not ggml graph building, scheduling or CPU launches.

## Where the GPU time goes (traced ms/token, fork - reinstinct)

### Qwen 3.8-27B: 7.05 ms

| item | fork | reinstinct | delta |
|---|---:|---:|---:|
| Q5_K matvec (191 calls, same grids) | 15.08 | 11.70 | **+3.38** |
| IQ4_XS + IQ4_NL + IQ3_S matvec | 6.42 (generic `mul_mat_vec_q`) | 5.63 (repacked IQ4_XS, IQ4_NL/IQ3_S relabelled onto it) | +0.78 |
| Q4_K + Q3_K matvec | 4.37 | 3.44 (Q3_K relabelled exactly onto Q6_K) | +0.93 |
| Q6_K matvec | 4.06 | 3.97 | +0.08 |
| attention (big kernels) | 0.37 | 0 | +0.37 |
| small kernels + dispatch (untraced GPU time minus big kernels) | 4.79 (1025 small kernels) | 3.27 (479) | +1.52 |

Q5_K per launch shape: 2176 WGs (ffn gate/up, ~61 MB) 111 vs 81 us = 552 vs 757 GB/s; 640 WGs 48 vs 32.5 us; 1280 WGs 69 vs 50 us. Neither kernel spills; VGPR 48 (fork) vs 64. The kernels share the repacked layout and the launch shape. The difference is reinstinct **54eaa04** (2026-09-12, before the crossport, never ledgered):
- rows are clamped (`min(row, out_dim-1)`) with a masked store, not `if (row >= ne1) continue`. The guard puts an execz branch between the two rows' loads and a waitcnt behind each, so only one row's planes are in flight at a time;
- Q5_K `spread4` is one multiply by `0x02040810` plus a mask, not four shift/mask/or chains; Q6_K `spread2` is two multiplies.

Measured then on the 27B: Q5_K 540 -> 700 GB/s, Q6_K 590 -> 690, IQ4_XS 560 -> 605, Q4_K flat. The fork's `mul_mat_vec_q5k_repacked`, `q6k`, `q4k`, `q3k` and the `HAS_IDS` variants still have the guarded loads (repack-gcn.cu ~565).

### Gemma 31B QAT (Q4_0): 8.09 ms

| item | fork | reinstinct | delta |
|---|---:|---:|---:|
| Q4_0 matvec | 33.80 (generic `mul_mat_vec_q<Q4_0>`, 351 calls) | 27.07 (`matvec_q4_0_repacked`, 435 calls) | **+6.73** |
| attention (big kernels) | 1.57 (60 calls) | 0.07 | +1.50 |
| small kernels + dispatch | 3.69 (1015) | 3.83 (1162) | -0.14 |

The fork has no repacked Q4_0 decode path (`GGML_CUDA_REPACK_*` covers Q8_0 and Q5_1), so every Q4_0 weight goes through the generic MMVQ. Reinstinct's repacked Q4_0 matvec runs at ~690 GB/s (the kernel-read ceiling on this card is ~720-830). Attention: the fork's decode runs `flash_attn_tile` on Gemma's 256/512-dim heads (60 big calls/token); reinstinct uses GQA flash-decoding (R3 class).

### Qwen 3.6-35B-A3B: 2.88 ms (traced delta 3.5)

| item | fork | reinstinct | delta |
|---|---:|---:|---:|
| glue (norm/elementwise/copy/quantize) | 2.80 (581 kernels) | 1.34 (264) | **+1.46** |
| MoE down Q5_K | 1.42 (37 us/call) | 0.52 (`moe_matvec_q5k_down`, 13.4 us/call) | +0.90 |
| dense Q8_0 matvec (attn/GDN/shared expert) | 4.49 (repacked, one 32 B block per lane) | 3.59 (two-plane layout, 54eaa04) | +0.90 |
| f32 matvec | 0.93 (100x `gcn_f32_matvec_rows` with 8 WGs + 40x fewrows with 1 WG) | 0.78 (70x `matvec_f32_b256`, 256 WGs) | +0.15 |
| router + shared-expert gate | 0.47 | 0.72 | -0.26 |
| GDN, attention, Q4_K gate/up, Q6_K | 1.71 | 1.53 | +0.18 |

Kernels per token: fork 1133, reinstinct 867.

## What closes it, in order of size

1. **Dense matvec load order + spread multiplies** (reinstinct 54eaa04 -> proposed R10): about 4.3 ms of the 27B's 7.05 (Q5_K 3.4, Q4_K/Q3_K 0.9); the MoE `HAS_IDS` variants share the guarded loads. A small, mechanical diff on the fork's existing repacked matvecs.
2. **Repacked Q4_0 decode matvec** (proposed R11): 6.7 of the 31B QAT's 8.1 ms. Reinstinct's `kernels/matvec_q4_0_repacked.cpp` + `quant::q4_0` repack; it also affects any Q4_0 QAT GGUF.
3. **Two-plane Q8_0 layout** (54eaa04, part of R10): 0.9 ms on the 35B; it touches every Q8_0 kernel (MMQ, grouped, MoE, dequant), so it is the larger change.
4. **MoE down kernel** (R4): 0.9 ms on the 35B.
5. **Glue fusion, rest of R6**: 1.5 ms on the 35B, ~1.5 ms on the 27B (1025 vs 479 small kernels).
6. **Gemma decode attention** (R3 class): 1.5 ms on the 31B.
7. **IQ4_NL / IQ3_S relabel onto repacked IQ4_XS** and **Q3_K exact relabel onto Q6_K** (reinstinct qwen35.rs ~295, quant/q3_k.rs): ~0.8 ms on the 27B, which goes through the fork's generic MMVQ for these formats today.

Items 1, 2 and 4 alone would take the fork to roughly 30.8 / 32.4 / 9.6 ms per token (27B / 31B QAT / 35B) from 35.1 / 39.1 / 10.5, against reinstinct's 28.1 / 31.0 / 7.7. Adding 3 and 5 brings the 35B to ~7.2 traced-equivalent, i.e. level. These are traced deltas, which run ~1 us/kernel high, so treat them as upper bounds.
