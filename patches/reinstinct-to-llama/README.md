# reinstinct -> llama fork: source extracts

Verbatim copies of reinstinct sources (github.com/sixvolts/reinstinct, branch `review-fixes` @ ee7cbd5) for the items in `notes/fork-gap-plan.md`. These are references to port from, not drop-in patches. Reinstinct compiles kernels at runtime from these files with hipcc for gfx906; launches are from Rust.

## Conventions that differ from the fork

- **Activation block.** Reinstinct's `BlockQ8` (40 B, `common/quantize_q8.cpp`) is `{ f32 d; f32 xsum; int8 qs[32]; }`, where `xsum = d * sum(q)` is the sum of the *quantized* values. The fork's `block_q8_1` (36 B) is `{ half2 ds; int8 qs[32]; }` with `ds.y` = the *unquantized* float sum, stored as fp16 (quantize.cu:86-100). Port the arithmetic, not the struct: `dx = __low2float(ds)`. Where a kernel needs the quantized sum (Q4_0's -8 offset), compute it with `sdot4(0x01010101, qs_word)` as these kernels do.
- **Weight layouts.** Planes per format, row-major over `[out_dim][nsp]`. `nsp = n_sub` padded by one when it is a power of two (anti-alias; the same rule as the fork's `repack_q4k_nsp`). The exact byte layout is in each `*.rs` `repack_for_matvec`.
- **Launch contracts.**
  - Decode matvec: block 256 = 4 waves, ROWS rows per wave, grid = ceil(out_dim / (4*ROWS)).
  - Small batch: `common/matvec_batched_nr*.h`, entries n1..n4 + w8. The host picks rows per wave (prefill.rs `SmallBatchMatvec::entry`): n1 -> 2, n2..n4 -> 4 (Q5_K n4 and Q6_K n3/n4 -> 2), w8 -> 2, with grid = ceil(out_dim / (4*rows)).
  - MMQ: block 256, grid = (ceil(out_dim/BM), ceil(P/BN)). BM = BN = 64 normally; the `_narrow_` entry (BM = BN = 16) is used when P <= 16.
- `common/gfx906_dpp.h` holds the wave64 DPP reductions (`wave64_reduce_add_f32`); the fork has `warp_reduce_sum<64>`.
