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

Known, unrelated, pre-existing on furnace's build with this model: `llama-perplexity -ub 1` and `llama-bench -p N` abort with a memory aperture violation; `llama-server` and `llama-bench -n` work. Under investigation.
