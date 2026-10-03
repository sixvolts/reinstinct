# Fork fixes found by cross-testing

Patches against the fork's public `gfx906-perf` branch (905021dba), for testing on podcast before they reach the fork's GitHub.

**Apply order** (`git am` in a 905021dba checkout):

1. B1
2. B2
3. R7
4. R6a
5. R6b
6. R6c
7. R8
8. L6 (`L6-topk-rank-v3-bafaf7d73.patch`)
9. R4 (`R4-moe-chain-q8-lds-nc-loads-first-69db7bf5d.patch`)
10. R11/R12 (`R11-R12-q4_0-iq-family-q3k-relabel-7f70fff53.patch`)

Items 8-10 were generated from a 905021dba + B1..R8 checkout on furnace (the fork commits they come
from sit on top of other fork work, so the MoE-chain patch's conflict with the missing few-token
expert-dedup kernel is already resolved here). All three `git am` cleanly in this order and the
three touched CUDA translation units pass a syntax-only hipcc compile on that tree. Report fork
numbers as "905021dba + B1..R8 + L6 + R4 + R11/R12".

- **L6**: rank-select top-k for 256/512-expert routing (`GGML_CUDA_NO_TOPK_RANK=1` = serial argmax).
- **R4** bundle: fused MoE down + router-weighted sum (`GGML_CUDA_DOWN_REDUCE_R=0` off), thread-packed
  Q5_1 down (`GGML_CUDA_Q5_1_DOWN_R=0`), gate/up kernel emitting the down's q8_1 (`GGML_CUDA_NO_GLU16_Q8`),
  Q8_0 LDS-staged 4..8-column matvec (`GGML_CUDA_NO_Q8_LDS_NC`), 9..32-column chunking
  (`GGML_CUDA_NO_Q8_NC_CHUNK`), loads-first Q8_0 multi / shared-expert GLU (`GGML_CUDA_NO_Q8_MULTI_UNROLL`,
  `GGML_CUDA_NO_Q8_GLU_UNROLL`), `GGML_CUDA_REPACK_TRACE=1` shape log.
- **R11/R12**: Q4_0 repack end to end (`GGML_CUDA_REPACK_Q4_0=0` off), IQ4_NL / IQ4_XS / IQ3_S on the same
  planes (`GGML_CUDA_REPACK_IQ=0` off), Q3_K -> Q6_K relabel (`GGML_CUDA_Q3K_RELABEL=0` off), 16/32/48-wide
  prefill token tiles for <= 48 columns (`GGML_CUDA_NO_MMQ_NARROW=1` off), `tests/test-repack-host`
  (CPU-only layout check; run with `GGML_CUDA_REPACK=0`).

On furnace the series applies cleanly in this order and compiles (plain `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, no NO_PEER_COPY). Single-GPU smoke run on Qwen3.5-4B UD-Q4_K_XL:

| test | tok/s |
|---|---:|
| pp1 | 88.9 |
| pp3 | 192.8 |
| pp16 | 339.3 |
| pp512 | 1610 |
| tg128 | 120.2 |

pp1 was an aperture abort before B2. Report fork numbers as "905021dba + B1..R8".

## B1: `B1-kq-repacked-nc-0934af928.patch`

The bug: dense repacked Q4_K/Q5_K/Q6_K with 2..16 activation columns went to the 64-wide MMQ tile, so a short batch cost as much as a full tile. MTP verify on Qwen 3.8-27B ran 3x slower than plain decode.

The fix: weight-once multi-column matvec `mul_mat_vec_kq_repacked_nc` (fork commit 0934af928 on qwen4exp-mtp). `GGML_CUDA_NO_KQ_NC=1` restores the old path for A/B runs.

Furnace check, Qwen3.5-4B UD-Q4_K_XL, 1x MI50:

| | old path | new |
|---|---|---|
| plain decode | 108 tok/s | 105 tok/s |
| MTP depth 2 | 31-34 tok/s | 117-129 tok/s |
| 11-14 token prompt | 148-165 ms | 74-96 ms |
| ub3 KLD vs ub512 base | 0.002932 | 0.002906 |

Acceptance is about 70% on both paths. Greedy MTP output diverges from plain decode on both paths, at near-tie words, from batch-size numerics. That is not new.

Still to confirm on podcast: Qwen 3.8-27B `llama-bench -p 1,2,3,4,8,16`, MTP tok/s, and Gemma 31B.

Confirmed on podcast by Reinstinct: Qwen 3.8-27B pp2 8.3 -> 38.1 tok/s, MTP 8.8-9.6 -> 37.6-42.2 tok/s (plain 27.6); Gemma 31B pp2 5.5 -> 47.5.

## B2: `B2-gdn-gather-elision-058d97848.patch`

The bug: the fused GDN state gather read the skipped gather's index tensor after the allocator had reused it. A single-token first ubatch on qwen35 aborted with a memory aperture violation.

The fix: elide only when nothing from the gather to the GDN output overlaps the indices (fork commit 058d97848).

Results:
- Qwen3.5-4B `-ub 1` perplexity now runs and equals the unfused result.
- Flash-Next keeps the elision: tg32 48.3, against 47.3 with the elision off.
- Independent of B1; apply both.

## R7: `R7-q6k-mmq-int8-staging-15d76b19d.patch`

Ported from reinstinct's Q6_K tile: Q6_K weights are expanded to int8 (q6 - 32) at LDS staging, so the inner loop is a plain int8 dot. Fork commit 15d76b19d.

Qwen3.5-4B, 1x MI50:

| model | Q6_K tile per call | pp512 |
|---|---|---|
| UD-Q4_K_XL | 2534 -> 1432 us | 1397 -> 1555 tok/s |
| Q5_K_M | 2277 -> 1311 us | 1263 -> 1468 tok/s |

Output is unchanged: KLD 0.000000.

## R6: decode glue fusions, three patches

All three keep results bit-identical (PPL equal with each one on and off). Each has an off switch for A/B runs.

| patch | what it fuses | off switch |
|---|---|---|
| R6a (8a8f44363) | residual ADD -> RMS_NORM -> MUL with its q8_1 copy; fused groups keep their emitted q8_1 cache entry | `GGML_CUDA_NO_ADD_NORM_FUSION=1` |
| R6b (50798f69b) | GDN gate ADD -> SOFTPLUS -> MUL, beta SIGMOID folded in | `GGML_CUDA_NO_ADD_UNARY_FUSION=1` |
| R6c (a2207dd89) | GDN q/k l2 norm pairs in one launch | `GGML_CUDA_NO_NORM_PAIR_FUSION=1` |

Qwen3.5-4B decode: launches/step 745 -> 610, tg +4-7%. Flash-Next decode: +2%.

## R8: `R8-f32-gemm-splitk-c5bc7f986.patch`

Split-K for skinny F32 GEMMs, reinstinct's policy. Qwen3.5-4B pp512: F32 GEMM 29.4 -> 3.9 ms. `GGML_CUDA_NO_F32_SPLITK=1` disables it.
