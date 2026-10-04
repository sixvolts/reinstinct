# S1: llama fork vs reinstinct, podcast, 1x MI50

- Card: podcast GPU 1 (MI50 32 GB, workstation ROM, 225 W, stock clocks, perflevel auto). Distro ROCm 7.1.
- Fork: `gfx906-perf` @ 905021dba, `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, env `GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1`, `-fa 1`. No rocBLAS abort on any model below.
- Reinstinct: `review-fixes` @ e2d4740.
- Date: 2026-10-03. Runs back to back per model, 15 s cool-down between engines.

## Plain decode / prefill

Fork: `llama-bench -m M -ngl 999 -fa 1 -p 512 -n 256 -r 5` (mean ± sd as printed).
Reinstinct: `generate-text M --tokens <512 ids> -n 1 --gpu` with `REINSTINCT_PREFILL=1` (pp512 = 512 / batched-prefill time) and `generate-text M --tokens 1000 -n 256 --gpu` (tg256), one warm-up then 5 runs each, mean ± sd. tg256 decodes from an empty context in both.

| Model (Unsloth UD GGUF) | fork pp512 | reinstinct pp512 | fork tg256 | reinstinct tg256 | decode delta |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 255.3 ± 0.6 | **292.2** ± 0.1 | 27.83 ± 0.01 | **35.32** ± 0.11 | +27% |
| Gemma 4 31B Q4_K_XL | 218.9 ± 0.4 | **254.1** ± 0.1 | 27.35 ± 0.01 | **29.14** ± 0.11 | +7% |
| Gemma 4 31B QAT (Q4_0) | 263.4 ± 0.8 | **319.3** ± 0.1 | 25.58 ± 0.04 | **31.84** ± 0.05 | +24% |
| Gemma 4 26B-A4B Q4_K_XL (MoE) | **1691.7** ± 24.1 | 1098.3 ± 0.3 | **96.14** ± 0.26 | 91.80 ± 0.19 | -5% |
| Gemma 4 E4B Q4_K_XL | **1303.0** ± 11.4 | 917.9 ± 0.5 | 98.16 ± 0.07 | **101.24** ± 0.22 | +3% |
| Qwen 3.6-35B-A3B Q4_K_XL (MoE) | **1576.3** ± 36.1 | 881.3 ± 0.2 | 88.55 ± 0.05 | **118.56** ± 0.46 | +34% |

Prefill: reinstinct leads on the dense models (+14-21%), the fork leads on the MoEs and E4B (+54-79%). pp512 token ids differ (llama-bench random, reinstinct a fixed spread) - irrelevant for timing.

## MTP: Qwen 3.8-27B Q4_K_XL, depth 2, greedy, 256 tokens

Same three raw prompts (no chat template): "Write a Python function that checks whether a string is a palindrome.", "The history of the lighthouse begins in the ancient world, where", "Explain how a refrigerator works, step by step."

Fork: `llama-server -m M -ngl 999 -fa on -c 4096 [--spec-type draft-mtp --spec-draft-n-max 2]`, `/completion` with `temperature 0, n_predict 256, cache_prompt false`, `timings.predicted_per_second`, `draft_n_accepted / draft_n`.
Reinstinct: `qwen-mtp-gen M --k 2 -n 256 --prompt P`, decode-only tok/s (prefill excluded), accepted / drafted.

| Prompt | fork no spec | fork MTP | fork accept | reinstinct plain | reinstinct MTP | reinstinct accept |
|---|---:|---:|---:|---:|---:|---:|
| palindrome | 27.5 | **9.6** | 81.9% | 34.7 | 51.4 | 79.0% |
| lighthouse | 27.5 | **8.8** | 71.0% | 34.4 | 45.8 | 64.7% |
| refrigerator | 27.5 | **9.0** | 74.1% | 34.7 | 48.1 | 70.1% |

**The fork's MTP on this dense GDN model runs ~3x slower than its own plain decode** despite 71-82% acceptance - something on that path is not on the fast kernels here (draft context placement, the dense verify shape, or the GDN snapshot path at n_rs_seq 2?). Reinstinct spec output equals plain greedy on 2 of 3 prompts; the lighthouse prompt diverges at token 74 (a batched-vs-single numerics near-tie), which also explains its lower acceptance.

## Update 2026-10-03: reinstinct after L8a-d (review-fixes @ 96c4223), same card and method

| Model | fork pp512 | reinstinct pp512 (S1 -> now) | fork tg256 | reinstinct tg256 (now) |
|---|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 255.3 | 292.2 -> **300.3** ± 0.2 | 27.83 | **35.28** ± 0.08 |
| Gemma 4 31B Q4_K_XL | 218.9 | **254.0** ± 0.1 | 27.35 | **29.16** ± 0.11 |
| Gemma 4 31B QAT (Q4_0) | 263.4 | **319.7** ± 0.5 | 25.58 | **31.78** ± 0.08 |
| Gemma 4 26B-A4B Q4_K_XL | **1691.7** | 1098.3 -> 1361.1 ± 0.8 | **96.14** | 91.56 ± 0.41 |
| Gemma 4 E4B Q4_K_XL | **1303.0** | 918.6 ± 0.5 | 98.16 | **101.38** ± 0.43 |
| Qwen 3.6-35B-A3B Q4_K_XL | **1576.3** | 881.3 -> 1286.0 ± 1.1 | 88.55 | **118.22** ± 0.45 |

Decode unchanged (no regression). MoE prefill gap: 35B-A3B 1.79x -> 1.23x, 26B-A4B 1.54x -> 1.24x. E4B prefill (dense, 1.42x) is untouched by L8a-d and still open.

E4B follow-up (reinstinct 2579a97): the gap was not MMQ (reinstinct's Q4_K/Q5_K/Q6_K dense tiles were level or ahead: 188/29/46 ms vs fork 196/31/91 per pp512) but the per-layer-embedding projection: F32 weight, converted to fp16 per call and run through a one-column-per-block GEMM, 236 ms vs the fork's gcn_f32_gemm_tn_rb 21 ms. With that kernel: **E4B pp512 919 -> 1483 tok/s** (fork 1303).

## Update 2026-10-03 (later): reinstinct after L6, L8 remainder, L3 (review-fixes @ ee7cbd5), same card and method

One run of the S1 reinstinct script on the current build; fork numbers are the original S1 run (905021dba, unchanged).

| Model | fork pp512 | reinstinct pp512 | fork tg256 | reinstinct tg256 | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 255.3 | **306.3** ± 0.2 | 27.83 | **35.20** ± 0.07 | +20% | +26% |
| Gemma 4 31B Q4_K_XL | 218.9 | **253.9** ± 0.2 | 27.35 | **29.04** ± 0.09 | +16% | +6% |
| Gemma 4 31B QAT (Q4_0) | 263.4 | **319.8** ± 0.6 | 25.58 | **31.76** ± 0.05 | +21% | +24% |
| Gemma 4 26B-A4B Q4_K_XL | **1691.7** | 1641.8 ± 1.0 | **96.14** | 94.40 ± 0.37 | -3% | -2% |
| Gemma 4 E4B Q4_K_XL | 1303.0 | **1507.6** ± 0.5 | 98.16 | **101.48** ± 0.33 | +16% | +3% |
| Qwen 3.6-35B-A3B Q4_K_XL | 1576.3 | **1741.0** ± 1.6 | 88.55 | **124.36** ± 0.42 | +10% | +40% |

What moved since the previous table (reinstinct commits on review-fixes):
- b54a0a9 L6 router top-k (parallel rank on logits, 20 -> 7 us/call): 35B-A3B tg 118 -> 124, 26B-A4B tg 91.6 -> 94.4.
- 766be98 split-K for the skinny F32 GEMMs (router, GDN alpha/beta), 9ae9f04 GDN prefill kernel at 111 VGPRs (2 waves/SIMD), df7c742 MoE prefill chunk 256 -> 1024 tokens: 35B-A3B pp512 1286 -> 1741, 26B-A4B 1361 -> 1642.
- 2579a97 (earlier) E4B per-layer-embedding F32 GEMM: 919 -> 1508.
- ee7cbd5 L3: Qwen verify rows through the flash-decoding attention. MTP K=2 on the 27B with a 13.5K-token prompt: 45.9 tok/s vs plain 30.5 (was 0.44x plain); short prompts unchanged (~1.35-1.57x).

Remaining fork leads: Gemma 26B-A4B prefill (-3%) and decode (-2%).

## Update 2026-10-03 (fork patched): fork 905021dba + B1..R8 vs reinstinct @ ee7cbd5, same card and method

Fork: 905021dba with patches/fork-fixes applied in order B1 B2 R7 R6a R6b R6c R8, same build flags and env as the original S1 run. Reinstinct numbers are the previous table (ee7cbd5, same day, same card).

| Model | fork pp512 (S1 -> patched) | reinstinct pp512 | fork tg256 (S1 -> patched) | reinstinct tg256 | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 255.3 -> 269.3 ± 0.7 | **306.3** | 27.83 -> 28.27 ± 0.03 | **35.20** | +14% | +25% |
| Gemma 4 31B Q4_K_XL | 218.9 -> 243.0 ± 0.8 | **253.9** | 27.35 -> 27.42 ± 0.01 | **29.04** | +4% | +6% |
| Gemma 4 31B QAT (Q4_0) | 263.4 -> 263.3 ± 1.0 | **319.8** | 25.58 -> 25.57 ± 0.04 | **31.76** | +21% | +24% |
| Gemma 4 26B-A4B Q4_K_XL | 1691.7 -> **1728.4** ± 23.9 | 1641.8 | 96.14 -> **96.35** ± 0.12 | 94.40 | -5% | -2% |
| Gemma 4 E4B Q4_K_XL | 1303.0 -> 1471.1 ± 13.1 | **1507.6** | 98.16 -> 98.24 ± 0.07 | **101.48** | +2% | +3% |
| Qwen 3.6-35B-A3B Q4_K_XL | 1576.3 -> 1728.9 ± 46.1 | **1741.0** | 88.55 -> 92.60 ± 0.12 | **124.36** | +1% | +34% |

(prefill / decode columns: reinstinct vs patched fork.) The patches moved fork prefill (R7 Q6_K tile: Gemma 31B Q4_K_XL +11%, E4B +13% with R8; R8 split-K: 35B-A3B +10%; B1: 27B +5%) and 35B-A3B decode (R6 round 1: +4.6%). Nothing in the set touches Q4_0, and dense decode is flat.

### MTP: Qwen 3.8-27B Q4_K_XL, depth 2, greedy, 256 tokens, server -c 16384

Same three raw prompts as above plus a 13,482-token prompt (long-context verify, L3). Fork: decode tok/s from `timings.predicted_per_second`. Reinstinct: `qwen-mtp-gen --k 2 -n 256`, decode-only (prefill excluded), plain = its own non-speculative greedy run.

| Prompt | fork no spec | fork MTP | fork accept | reinstinct plain | reinstinct MTP | reinstinct accept |
|---|---:|---:|---:|---:|---:|---:|
| palindrome | 28.0 | 42.5 (1.52x) | 83.2% | 34.4 | **54.5** (1.59x) | 80.3% |
| lighthouse | 28.1 | 37.8 (1.35x) | 68.4% | 34.4 | **48.0** (1.40x) | 64.7% |
| refrigerator | 28.1 | 38.1 (1.36x) | 69.2% | 34.1 | **51.0** (1.50x) | 71.2% |
| 13.5K-token prompt | 25.9 | 38.0 (1.47x) | 77.9% | 30.3 | **45.8** (1.51x) | 81.1% |

B1 fixed the fork's MTP (was 8.8-9.6 tok/s, ~0.33x plain); both engines now get 1.35-1.59x over their own plain decode, including at 13.5K context. The absolute gap is the plain-decode gap (R5). Reinstinct spec output equals its plain greedy output on 3 of 4 prompts (lighthouse diverges at token 134, a near-tie). Fork prefill of the 13.5K prompt: 249 tok/s (54.1 s); reinstinct 52.8-54.2 s.

## Update 2026-10-03 (fork + L6, R4, R11/R12, R10a): vs reinstinct @ ee7cbd5, same card and method

Fork: the patched build above plus patches/fork-fixes 8-11 in order (L6, R4, R11/R12, R10a). Two runs: (a) through R11/R12, (b) with R10a added. Same flags and env (`GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1`); the Q4_0 and IQ repacks are on by default. Reinstinct numbers are the ee7cbd5 table.

| Model | fork pp512 (B1..R8 -> a -> b) | reinstinct pp512 | fork tg256 (B1..R8 -> a -> b) | reinstinct tg256 | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 269.3 -> 291.7 -> 291.9 ± 0.7 | **306.3** | 28.27 -> 28.67 -> 32.27 ± 0.03 | **35.20** | +5% | +9% |
| Gemma 4 31B Q4_K_XL | 243.0 -> 243.1 -> 243.1 ± 0.7 | **253.9** | 27.42 -> 27.42 -> **30.09** ± 0.01 | 29.04 | +4% | -3% |
| Gemma 4 31B QAT (Q4_0) | 263.3 -> 268.0 -> 268.0 ± 1.2 | **319.8** | 25.57 -> 32.17 -> **32.18** ± 0.01 | 31.76 | +19% | -1% |
| Gemma 4 26B-A4B Q4_K_XL | **1728.4** -> 1727.9 -> 1725.0 ± 35.6 | 1641.8 | 96.35 -> 95.55 -> 83.00 ± 0.07 | **94.40** | -5% | +14% |
| Gemma 4 E4B Q4_K_XL | 1471.1 -> 1472.6 -> 1472.2 ± 20.2 | **1507.6** | 98.24 -> 98.96 -> **111.16** ± 0.08 | 101.48 | +2% | -9% |
| Qwen 3.6-35B-A3B Q4_K_XL | 1728.9 -> 1730.0 -> 1729.4 ± 41.5 | **1741.0** | 92.60 -> 93.59 -> 94.97 ± 0.06 | **124.36** | +1% | +31% |

(prefill / decode columns: reinstinct vs fork (b); negative = fork ahead.)

A/B on build (b), same command:

| Run | pp512 | tg256 |
|---|---:|---:|
| 27B, default | 291.9 | 32.27 |
| 27B, `GGML_CUDA_REPACK_IQ=0` | 272.6 | 31.69 |
| 27B, `GGML_CUDA_NO_Q4K_FENCE=1` | 292.0 | 31.66 |
| 31B QAT, default | 268.0 | 32.18 |
| 31B QAT, `GGML_CUDA_REPACK_Q4_0=0` | 262.9 | 25.55 |

On build (a), `GGML_CUDA_Q3K_RELABEL=0` gave 288.2 / 28.70 on the 27B (relabel: ~1% prefill, decode flat). The single-column repacked IQ4_XS matvec is slower than canonical MMVQ (94.7 vs 86.7 us at 5120x17408, test-repack-bench t=1), but the IQ repack is a net decode win on the 27B because IQ4_NL / IQ3_S speed up more.

**Regression: Gemma 26B-A4B decode with R10a**, 96 -> 83-89 tok/s (noisy run to run with R10a). Graph timer: 9.99 -> 10.85 ms GPU per replay, same capture/update counts (one capture, then replays). Not the Q4_K fence (82.4 with it off). Graphs-off kernel trace: the dense `mul_mat_vec_q8_0_repacked<1,1,false>` at 4096 rows (K = 2816) went 19.5 -> 43.9 us (25 calls/token, +0.61 ms), the 2816-row shape 23.9 -> 17.0 us, the HAS_IDS Q4_K path 37.3 -> 39.5 us. Sent to Furnace; a replacement for R10a is in progress.

**QAT prefill stays 19% behind** after R11: the ported tile `mmq_gemm_nib_repacked<0,4>` runs at the generic MMQ's speed, not reinstinct's (per pp512: 1711 ms vs reinstinct 1407, generic 1743; same grid, 128 VGPRs, occupancy 2). The ISA shows the difference is the activation staging: the 36 B `block_q8_1` leaves `qs` 4 B aligned in LDS, so the tile issues 72 `ds_read2_b32` and 83 `s_waitcnt` where reinstinct's 40 B block issues 32 `ds_read2_b64` + 8 `ds_read2_b32` and 45 waits. Furnace is moving all repacked tiles to a 40 B staged block.

### MTP: Qwen 3.8-27B Q4_K_XL, depth 2, build (b)

| Prompt | fork no spec | fork MTP | fork accept | reinstinct plain | reinstinct MTP | reinstinct accept |
|---|---:|---:|---:|---:|---:|---:|
| palindrome | 32.0 | **55.3** (1.73x) | 92.7% | 34.4 | 54.5 | 80.3% |
| lighthouse | 32.0 | 47.4 (1.48x) | 72.1% | 34.4 | **48.0** | 64.7% |
| refrigerator | 31.9 | 50.6 (1.59x) | 80.5% | 34.1 | **51.0** | 71.2% |
| 13.5K-token prompt | 29.1 | 44.1 (1.52x) | 73.8% | 30.3 | **45.8** | 81.1% |

MTP is roughly level with reinstinct now. Fork acceptance moved with the R10a numerics (not bit-identical to the previous build).

## Update 2026-10-04 (fork + R13): vs reinstinct @ ee7cbd5, same card and method

Fork: the previous build plus patches/fork-fixes 12-13 (R13a bench shapes, R13b MMQ staging/padding + R10 follow-ups), i.e. "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a + R13". Same flags and env. The NO_Q8_HOIST column is the planned patch-14 default (generic Q8_0 hoist opt-in).

| Model | fork pp512 (R10a -> R13) | reinstinct pp512 | fork tg256 (R10a -> R13) | fork tg256, `GGML_CUDA_NO_Q8_HOIST=1` | reinstinct tg256 | prefill | decode (patch-14 behaviour) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 291.9 -> **334.1** ± 1.2 | 306.3 | 32.27 -> 32.13 ± 0.03 | - | **35.20** | -8% | +10% |
| Gemma 4 31B Q4_K_XL | 243.1 -> **255.4** ± 0.7 | 253.9 | 30.09 -> **30.09** ± 0.01 | - | 29.04 | -1% | -3% |
| Gemma 4 31B QAT (Q4_0) | 268.0 -> 295.0 ± 1.2 | **319.8** | 32.18 -> **32.03** ± 0.01 | - | 31.76 | +8% | -1% |
| Gemma 4 26B-A4B Q4_K_XL | 1725.0 -> **1843.2** ± 38.5 | 1641.8 | 83.00 -> 82.94 ± 0.07 | **95.33** ± 0.31 | 94.40 | -11% | -1% |
| Gemma 4 E4B Q4_K_XL | 1472.2 -> **1548.3** ± 16.5 | 1507.6 | 111.16 -> **111.02** ± 0.15 | 110.11 ± 0.10 | 101.48 | -3% | -8% |
| Qwen 3.6-35B-A3B Q4_K_XL | 1729.4 -> **1792.2** ± 48.1 | 1741.0 | 94.97 -> 93.10 ± 0.09 | 93.61 ± 0.09 | **124.36** | -3% | +33% |

(prefill / decode columns: reinstinct vs fork; negative = fork ahead. Decode uses the NO_Q8_HOIST column where measured.) `GGML_CUDA_REPACK_Q4_0=0` on the QAT: 263.3 / 25.53.

- R13b's 8 B-aligned activation staging and padded LDS weight rows lift fork prefill on every model (+4-14%). The fork now leads prefill everywhere except the 31B QAT, which is still 8% behind (1736 vs 1601 ms per pp512) although the Q4_0 tile matches reinstinct's standalone; the remainder is outside the tile and not yet attributed.
- Decode: the fork leads or is level on all three Gemmas and E4B. The open gaps are the 35B-A3B (+33%) and the 27B (+10%).
- 26B-A4B decode with graphs on (tg128, two passes): R13 default 90.7 / 90.5, pre-R10a 97.1 / 96.5, `NO_KQ_HOIST=1` 90.1 / 90.7, `NO_Q8_HOIST=1` 96.9 / 97.2, both 97.2 / 97.3. The whole R10a regression is the generic Q8_0 hoist on the K <= 3072 shapes, invisible in test-repack-bench (every 26B Q8_0 shape within 0.3 us either way). Furnace makes it opt-in (`GGML_CUDA_Q8_HOIST=1`) in patch 14.

MTP on the 27B (depth 2) is unchanged from the R10a build: 55.3 / 47.4 / 50.6 / 44.1 tok/s (no spec 31.8-31.9, 29.0 at 13.5K) vs reinstinct 54.5 / 48.0 / 51.0 / 45.8.

## Update 2026-10-04 (fork + R14): vs reinstinct @ ee7cbd5, same card and method

Fork: the R13 build plus patches/fork-fixes 14 (R14: P3 thread-packed K-quant expert matvec for short K, Q8_0 dense K=4096 at 2 rows per 64-thread block, Q8_0 hoist opt-in, MMQ scale rows unpadded), i.e. "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a + R13 + R14". Same flags and env; the default build is now the patch-14 behaviour, so there is no separate NO_Q8_HOIST column. test-repack-bench --check OK.

| Model | fork pp512 (R13 -> R14) | reinstinct pp512 | fork tg256 (R13 -> R14) | reinstinct tg256 | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 334.1 -> **333.9** ± 1.5 | 306.3 | 32.13 -> 32.13 ± 0.03 | **35.20** | -8% | +10% |
| Gemma 4 31B Q4_K_XL | 255.4 -> **255.4** ± 0.6 | 253.9 | 30.09 -> **30.15** ± 0.01 | 29.04 | -1% | -4% |
| Gemma 4 31B QAT (Q4_0) | 295.0 -> 319.2 ± 0.8 | 319.8 | 32.03 -> **32.03** ± 0.01 | 31.76 | 0% | -1% |
| Gemma 4 26B-A4B Q4_K_XL | 1843.2 -> **1843.5** ± 34.4 | 1641.8 | 95.33 (NO_Q8_HOIST) -> **96.18** ± 0.14 | 94.40 | -11% | -2% |
| Gemma 4 E4B Q4_K_XL | 1548.3 -> **1541.7** ± 27.4 | 1507.6 | 110.11 (NO_Q8_HOIST) -> **110.29** ± 0.14 | 101.48 | -2% | -8% |
| Qwen 3.6-35B-A3B Q4_K_XL | 1792.2 -> **1789.3** ± 52.3 | 1741.0 | 93.61 (NO_Q8_HOIST) -> 118.44 ± 0.12 | **124.36** | -3% | +5% |

(prefill / decode columns: reinstinct vs fork; negative = fork ahead.) A/Bs: `GGML_CUDA_REPACK_Q4_0=0` on the QAT 263.3 / 25.53; `GGML_CUDA_Q8_ROWU=old` on the 35B 1798.7 / 103.27.

- 35B-A3B decode 93.6 -> 118.4 (+27%); the gap to reinstinct falls from 2.7 to 0.40 ms/token. Separate tg256 A/B run: default 118.69, `GGML_CUDA_Q8_ROWU=old` 102.62, `GGML_CUDA_KQ_DOWN_R=0` 106.69. In a rocprof trace (graphs off, 17 forwards) the Q8_0 4096x2048 calls take 14.7 us median on the 2-rows kernel (one-wave was 48.6, reinstinct 11.4 traced), the other Q8_0 calls 23.8 -> 17.3 us, the Q5_K expert down 37.6 -> 13.9 us (`mul_mat_vec_kq_repacked_pack`), and the traced matvec total 95.5 -> 58.2 ms (about 2.2 ms/token). The rest of the gap is glue: 1133 vs 867 kernels per token.
- 26B-A4B decode: the patch-14 default reaches 96.2 and stays ahead of reinstinct; `Q8_ROWU=old` reads 95.75 against 96.72 in the A/B run, so the 2-rows mapping adds about 1% here as well.
- 31B QAT prefill reaches parity: 1604 vs 1601 ms per pp512. The shipped `mmq_gemm_nib_repacked<0,4>` now has the same LDS instruction mix as the harness variant x8p (36 ds_read_b128, 8 ds_read2_b32, 32 ds_read2_b64, 62 s_waitcnt; R13 had 32 / 16 / 32 / 65). In-model the tile drops 8.8%, to about 1399 ms per forward vs reinstinct's 1407. The R13 note's "remainder outside the tile" was wrong: it was the padded scale rows breaking the b128 scale loads (notes/R13-attribution.md).
- Open: 27B decode (+10%: small kernels ~1.5 ms, Q5_K ~0.46, IQ nib matvec ~0.46, attention ~0.38) and the 35B glue (~0.4 ms). Furnace takes the glue next from the R6 table.

MTP on the 27B (depth 2) is unchanged: 55.3 / 47.4 / 50.5 / 44.1 tok/s (no spec 31.8, 29.0 at 13.5K) vs reinstinct 54.5 / 48.0 / 51.0 / 45.8.

## Update 2026-10-04 (fork + R15, final fork state for now): vs reinstinct @ ee7cbd5, same card and method

Fork: the R14 build plus patches/fork-fixes 15 (R15: the GDN conv step runs as one launch, with the conv-cache GET_ROWS elided and CONCAT + state-tail CPYs fused into `conv_step_concat_f32`; fork 3a30ac2c2, production), i.e. "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a + R13 + R14 + R15". Same flags and env; test-repack-bench --check OK. The fork owner has paused fork improvement work, so this is the last fork row until that changes.

| Model | fork pp512 (R14 -> R15) | reinstinct pp512 | fork tg256 (R14 -> R15) | reinstinct tg256 | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 333.9 -> **333.4** ± 1.0 | 306.3 | 32.13 -> 32.41 ± 0.01 | **35.20** | -8% | +9% |
| Gemma 4 31B Q4_K_XL | 255.4 -> **255.4** ± 0.7 | 253.9 | 30.15 -> **30.11** ± 0.01 | 29.04 | -1% | -4% |
| Gemma 4 31B QAT (Q4_0) | 319.2 -> 319.0 ± 1.3 | 319.8 | 32.03 -> **32.01** ± 0.01 | 31.76 | 0% | -1% |
| Gemma 4 26B-A4B Q4_K_XL | 1843.5 -> **1842.7** ± 41.9 | 1641.8 | 96.18 -> **96.95** ± 0.09 | 94.40 | -11% | -3% |
| Gemma 4 E4B Q4_K_XL | 1541.7 -> **1545.4** ± 27.8 | 1507.6 | 110.29 -> **110.74** ± 0.15 | 101.48 | -2% | -8% |
| Qwen 3.6-35B-A3B Q4_K_XL | 1789.3 -> **1789.7** ± 57.5 | 1741.0 | 118.44 -> 121.88 ± 0.17 | **124.36** | -3% | +2% |

(prefill / decode columns: reinstinct vs fork; negative = fork ahead.) A/Bs: `GGML_CUDA_NO_CONV_STEP_FUSION=1` gives tg256 118.47 on the 35B and 32.12 on the 27B; `GGML_CUDA_Q8_ROWU=old` on the 35B 1798.5 / 105.73; `GGML_CUDA_REPACK_Q4_0=0` on the QAT 263.2 / 25.54.

- Kernels per token with graphs off: 35B 1133 -> 1073 (reinstinct 867), 27B 1370 -> 1274. In every GDN layer, `k_get_rows_float`, `concat_cont` and two of the three state-tail `cpy_scalar` launches become one `conv_step_concat_f32<3>`. Decode gains 0.24 ms/token on the 35B (+2.9%) and 0.28 ms/token on the 27B (+0.9%).
- Greedy output on the 35B (2 prompts x 128 tokens) is byte-identical with fusion on and off.
- Remaining decode gaps: the 35B is 0.16 ms/token (+2%) behind. The 27B is 2.45 ms/token (+9%) behind: about 1.2 ms from small kernels after R15, plus Q5_K 0.46, the IQ nib matvec 0.46 and attention 0.38, per notes/R13-attribution.md. The remaining glue rows (norm+quantize pairs, gated norm, q/k norm+rope, split-K placement) are not done.
- MTP on the 27B (depth 2): 57.2 / 49.0 / 52.3 / 45.5 tok/s (no spec 32.1, 29.2 at 13.5K) vs reinstinct 54.5 / 48.0 / 51.0 / 45.8. The fork leads on the three short prompts and is level at 13.5K.
- Harness note: llama-server serving the 35B with `-c 4096` hung twice on shutdown after SIGTERM, R15 fused and unfused alike. It printed "cleaning up before exit" / "Received second interrupt, terminating immediately" and then sat in futex_wait with 4 threads until SIGKILL. The 27B servers in the MTP section shut down cleanly. This is not attributed and is not specific to R15. The S1 server helper now SIGKILLs after 60 s.
