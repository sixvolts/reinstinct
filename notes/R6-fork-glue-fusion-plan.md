# R6 on the fork: glue fusion plan

## Where the time goes

Reinstinct's attribution (notes/R6-L8-attribution-qwen36-35B.md) shows its decode lead on Qwen 3.6-35B-A3B comes from fused glue: norm, elementwise and copies take 1.26 ms per step against the fork's 3.16, with fewer launches. Matvecs are level.

Same picture on the fork's production model, Flash-Next, 4x MI50: production binary 300fdfe3b, decode trace, kernel time per step.

**Already fused** (keep):

| kernel | us/step | calls/step | what it fuses |
|---|---:|---:|---|
| `rms_norm_mul_q8_f32` | 687 | 94 | rmsnorm, weight multiply, Q8_1 quantize |
| `unary_gated_q8_kernel<sigmoid>` | 177 | 48 | sigmoid gate, quantize |
| `sigmoid_mul_add_f32` | 136 | 48 | sigmoid, multiply, add |
| `hc_up_pre_f32` / `dsv4_hc_post_f32` | 1112 / 353 | 96 / 96 | the hyper-connection pre/post pairs |

**Unfused glue** (the R6 target): about 2.1 ms per step and roughly 400 launches.

| kernel | us/step | calls/step | us/call |
|---|---:|---:|---:|
| `k_bin_bcast<op_add>` | 298 | 71 | 4.2 |
| `cpy_scalar<f32,f32>` | 280 | 53 | 5.3 |
| `quantize_q8_1` (standalone) | 275 | 101 | 2.7 |
| `scale_f32` | 249 | 33 | 7.5 |
| `rms_norm_f32<256>` (q/k head norms) | 243 | 72 | 3.4 |
| `rms_norm_scale_f32<256>` | 188 | 72 | 2.6 |
| `k_get_rows_float` | 170 | 39 | 4.4 |
| `concat_cont` | 136 | 37 | 3.7 |
| `k_set_rows<f32,i64,f16>` | 122 | 36 | 3.4 |

Each call is mostly launch and latency: 3-8 us for tiny tensors.

## Plan, in order of expected value

1. **Quantize into the producer.** 101 standalone `quantize_q8_1` launches per step. Fold quantize into whichever op produces the matvec input (add, scale, swiglu, sigmoid_mul) the way `rms_norm_mul_q8` already does. Expected ~0.25 ms plus launches.
2. **add + rmsnorm (+ quantize).** The residual add followed by the next norm: one kernel writes the residual and the normed, quantized activation. Covers most `k_bin_bcast<add>` calls.
3. **q/k head norm + scale (+ rope).** `rms_norm_f32<256>` + `rms_norm_scale_f32<256>` + `rope_multi` on the same rows; 180 launches per step between them.
4. **Copies and scales.** Find which `cpy_scalar` / `scale_f32` are layout copies the graph could avoid (views) versus real conversions.
5. **KV write.** `set_rows` into the f16 KV after rope: fold into the rope kernel when the destination is the cache.

## Method

- Each fusion goes in behind a graph-pattern matcher in `ggml-cuda.cu` (same style as the existing ones), with an env switch to disable it.
- Correctness: bit-identity where the math is unchanged (fused add+norm in the same order), else KLD at ub1/ub3 vs the unfused path.
- Speed: `llama-bench` tg on Flash-Next (production layout) and on Qwen3.5-4B. Report the per-family trace before and after with `scripts/kernel_families.py`.
- Reinstinct's fused kernels are the reference designs where they exist.

## Cost

Testing on Flash-Next needs short production stops: each window is about 5-10 minutes.
