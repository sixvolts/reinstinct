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
