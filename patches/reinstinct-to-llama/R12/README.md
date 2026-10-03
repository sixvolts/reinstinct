# R12: IQ4_XS family repack + relabels (plan P5)

| file | what |
|---|---|
| `iq4_xs.rs` `repack_for_matvec` | nibble plane `out*nsp*16` (per 32-weight sub-block; nibbles index the IQ4_NL codebook) + fp16 scale plane `out*nsp*2` holding `d*(ls-32)` folded per sub-block. **Not bit-exact** vs ggml dequant (fp16 rounding of the product, <= 2^-11 relative) |
| `iq4_nl.rs` | IQ4_NL -> the same layout: d is already fp16 per 32, so **exact** |
| `iq3_s.rs` `repack_for_matvec`, `kernel_source` | IQ3_S -> the same layout with its own 16-entry codebook (`(v+15)/2` nibbles, `db = d*(1+2s)` folded to fp16, not exact). The kernels are the IQ4_XS sources compiled with `IQ4NL_KV_*` / `IQ4NL_KV_TABLE` redefined (see `kernel_source`) |
| `q3_k.rs` | Q3_K -> Q6_K relabel, **exact** (q6 = q3 + 28, sc6 = sc3 - 32; tests compare dequant bit for bit). The fork's existing Q6_K repack/kernels then serve it |
| `matvec_iq4xs_repacked.cpp` | decode matvec (ROWS=2, branchless rows; 605 GB/s on the 27B) |
| `matvec_iq4xs_repacked_batched.cpp` | small-batch decoder for `common/matvec_batched_nr.h` |
| `mmq_gemm_iq4xs_repacked.cpp` | prefill tile + `_narrow_`; on the 27B, 351 ms for IQ4_XS+NL+IQ3_S vs the fork's generic `mul_mat_q` 501 ms per pp512 |
| `dequant_iq4xs_repacked_f16.cpp` | fallback dequant |
| `qwen35-relabel-excerpt.rs` | reinstinct's dtype routing for the UD-XL formats (which relabel onto which layout) |

**Layout choice (plan P5).** Keep reinstinct's folded fp16 sub-block scale, with one kernel set for three formats and KLD validation for IQ4_XS/IQ3_S; or use a u16 `ls` plane + fp16 d per superblock (the fork's Q4_K plane style), which is exact for IQ4_XS but needs a separate scale plane for IQ4_NL. Reinstinct runs the folded form for the Qwen 3.8-27B UD-XL file in every S1 number; it was not separately KLD-checked against an exact layout.

**Order.** Do the Q3_K -> Q6_K relabel first. It is exact, small, and saves ~33 ms per pp512 on the 27B (the q3k tile is 13.9 ms/call).
