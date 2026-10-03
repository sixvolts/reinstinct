# R11: repacked Q4_0 (plan P2)

| file | what | fork target |
|---|---|---|
| `q4_0.rs` `repack_for_matvec` | nibble plane `out*nsp*16` (on-disk qs copied as-is: byte k = weight k low / k+16 high, already sdot4 order) + fp16 d plane `out*nsp*2` | new `repack_q4_0_host`, size switch (repack-gcn.cu ~39), support switch (~55) |
| `matvec_q4_0_repacked.cpp` | decode matvec, entries ROWS=2 (`_f32`), ROWS=1 (`_r1_f32`), ROWS=4 (`_r4_f32`); reinstinct launches ROWS=2 for all shapes today | `mul_mat_vec_q4_0_repacked` |
| `matvec_q4_0_repacked_batched.cpp` | 1..8-row small-batch decoder for `common/matvec_batched_nr.h` (LDS-staged activations for n2..n4) | `mul_mat_vec_kq_repacked_nc` analogue (2..16 cols, chunks of 8) |
| `mmq_gemm_q4_0_repacked.cpp` | prefill tile; the Q4_K tile with a single fp16 scale per block and the -8 folded into the int accumulator; `_narrow_` entry for P <= 16 | `mmq_gemm_q4_0_repacked`; 1407 vs 1743 ms per pp512 on Gemma 31B QAT |
| `embed_lookup_repacked.cpp` | gather from a repacked Q4_0 (and reinstinct-Q8_0) table; needed when token_embd is Q4_0 and repacked (tied LM head) | GET_ROWS on a repacked Q4_0 tensor, or keep token_embd out of the repack buft |
| `dequant_q4_0_repacked_f16.cpp` | repacked -> fp16, for any fallback GEMM path | only if the fork keeps a dequant fallback for repacked types |

## Not here, and needed on the fork

- **A fused gate/up GLU kernel**, plus adding Q4_0 to `ggml_cuda_repack_should_fuse_glu`. Reinstinct runs gate and up as two matvecs plus `geglu`/`swiglu`. Build it from the fork's `mul_mat_vec_q4k_repacked_glu`, using the Q4_0 inner loop below.
- **HAS_IDS variants.** Keep 3D Q4_0 tensors out of the repack buffer type until they exist.

## The inner loop to keep

```
xqsum = sum_g sdot4(0x01010101, xq32[g])      // quantized activation sum, row-independent, hoisted
idot  = sum_j sdot4(q & 0x0F0F0F0F, xq32[j]) + sdot4((q >> 4) & 0x0F0F0F0F, xq32[j+4])
acc  += dw * dx * (float)(idot - 8 * xqsum)
```

Do not use `ds.y` for the offset. The kernel header has the 40x per-matvec accuracy argument; the fork's generic Q4_0 path does use the float sum and works, but this form cancels the activation quantization error. Expect nonzero KLD vs the unrepacked fork by design.

Do not apply the P1 load hoisting to Q4_0. Reinstinct tried it and lost 4-10%; the kernel was already at the bandwidth ceiling (~690 GB/s).
