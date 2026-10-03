# Fork fixes found by cross-testing

Patches against the fork's public `gfx906-perf` branch, for testing on podcast before they reach the fork's GitHub. Apply with `git am` (or `git apply`) in a gfx906-perf checkout.

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
