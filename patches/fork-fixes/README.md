# Fork fixes found by cross-testing

> 2026-10-04: everything in patches 1-14 is on the fork's public branches `qwen4exp-mtp` (7bac8169e)
> and `gfx906-perf-upstream` (e05405047). The series remains for A/Bs against the 905021dba base.

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
11. R10a (`R10a-kquant-loads-first-b8eb8050f.patch`)
12. R13a (`R13a-bench-26b-shapes-746887a5c.patch`) — test-repack-bench shapes only
13. R13b (`R13b-mmq-staging-padding-r10-followups-ae7fdb2d7.patch`)
14. R14 (`R14-kq-expert-pack-q8-mapping-q8hoist-optin-7bac8169e.patch`)

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
- **R10a**: dense Q4_K/Q5_K/Q6_K/Q8_0 matvecs and the dense Q4_K GLU with clamped rows and hoisted
  plane loads, `__launch_bounds__(256)` on the K-quant matvecs, Q4_K-only scheduling fence
  (`GGML_CUDA_NO_Q4K_FENCE=1` drops it). Qwen3.5-4B, 1x MI50: Q6_K 63.5 -> 53.8 us, Q5_K 32.4 -> 25.8,
  Q4_K GLU 51.9 -> 41.5; tg64 UD-Q4_K_XL 116 -> 131.5, Q5_K_M 101 -> 120. Rounding-close, not
  bit-identical (contraction): KLD 0.0029 at ub1 on the 4B. Report fork numbers as
  "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a".
- **R13b**: (a) every repacked MMQ tile stages activations as a 40 B 8 B-aligned block with widened
  scales and pads its LDS weight/scale rows (the reinstinct ISA finding + the bank-conflict fix): the
  Q4_0 tile goes 8.8 -> 11.25 TMAC/s standalone, identical to reinstinct's; q5_K 7.75 -> 9.4,
  q6_K 6.95 -> 8.3, IQ 8.8 -> 9.7, MoE Q5_1 down 3.3 -> 4.6; bit-identical results. (b) R10a
  follow-ups: expert (HAS_IDS) matvec/GLU paths back on the guarded loop; generic Q8_0 hoist
  K-gated to ne00 <= 3072; A/B switches `GGML_CUDA_NO_KQ_HOIST=1` (dense K-quant matvecs + GLU
  back to the pre-R10a loop) and `GGML_CUDA_NO_Q8_HOIST=1`. Report fork numbers as
  "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a + R13".
- **R14** (= fork 7bac8169e, the production build): (a) P3: thread-packed Q4_K/Q5_K/Q6_K expert
  matvec for K <= 2048 (`mul_mat_vec_kq_repacked_pack`, `GGML_CUDA_KQ_DOWN_R`, 0 = old): 35B-class
  768x2048x128 per call Q5_K 60.8 -> 34.6 us, Q6_K 63.4 -> 48.5, Q4_K 51.6 -> 44.3 (1x MI50 bench).
  (b) dense Q8_0 at >= 512 rows: K = 4096 -> 2 rows per 64-thread block (4096x2048: 67 -> 29.7 us,
  = canonical; 4096x2816 41 -> 34), K = 6144 keeps rowu; `GGML_CUDA_Q8_ROWU=all|r2|old` for A/B.
  (c) generic Q8_0 hoist opt-in (`GGML_CUDA_Q8_HOIST=1`; your 26B A/B). (d) MMQ scale rows: float2
  padded, the nib tile's float rows unpadded (your ISA diff): Q4_0 tile 9.9 -> 11.27 TMAC/s in the
  bench, Q5_1 MoE down tile back at 4.54. Report as "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a +
  R13 + R14".
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
