# Decode / prefill attribution on the R13 fork build

Podcast GPU 1, 2026-10-04. Fork "905021dba + B1..R8 + L6 + R4 + R11/R12 + R10a + R13" with `GGML_CUDA_NO_Q8_HOIST=1` (patch-14 behaviour) vs reinstinct ee7cbd5 (R5 traces, unchanged engine). Method as in `R5-decode-attribution.md`: rocprofv3 kernel trace with graphs off, 17 decode forwards. Fork GPU time per token is estimated as 1000 / tg256 minus 0.35 ms host (no graph-timer build this round), so the "small kernels + dispatch" rows carry roughly ±0.2 ms.

## Qwen 3.6-35B-A3B: fork ~10.3 vs reinstinct 7.66 ms GPU per token

This build is per-kernel identical to the R5 fork apart from top-k (11.7 -> 10.1 us): 1133 kernels/token, same calls and same per-call times. The R4 MoE-chain pieces are gated to Flash-Next shapes (ne00 2560), and the 35B's ne00 is 2048.

| item | fork | reinstinct | delta |
|---|---:|---:|---:|
| dense Q8_0, 2048 rows x K 4096 (`ssm_out` x30, `attn_output` x10) | 40 x 48.6 us = 1.94 ms | 40 x ~11.4 us (2 rows per 64-thread WG) | ~+1.0-1.4 |
| MoE down Q5_K (`mul_mat_vec_q5k_repacked<true,true>`) | 36 x 38.1 us = 1.37 ms | 13.4 us/call, 0.48 ms | +0.9 |
| MoE gate/up Q4_K GLU (generic per-slot HAS_IDS path, not glu16) | 39 x 22.4 us = 0.87 ms | 0.81 ms | +0.06 |
| glue (kernels per token) | 1133 | 867 | ~+1.5 (R5) |

The Q8_0 shape is slow standalone too: test-repack-bench 4096x2048 runs at 38.6-41.4 us/op untraced, against 26 us/op for 2048x4096, which reads the same bytes; traced standalone it is 55 us steady. One 64-lane wave walks 128 blocks per row, so the kernel is latency-bound. The fork's `rowu<6,64>` special case for ne00 == 6144 addresses the same problem on Flash-Next.

## Qwen 3.8-27B: fork ~30.8 vs reinstinct 28.07 ms GPU per token (gap 2.7, was 7.05 in R5)

| item | fork | reinstinct | delta |
|---|---:|---:|---:|
| Q5_K matvec | 12.16 | 11.70 | +0.46 (was +3.38) |
| IQ4_XS / IQ4_NL / IQ3_S (nib kernels vs repacked IQ4_XS) | 6.09 | 5.63 | +0.46 |
| Q4_K + Q6_K | 7.31 | 7.41 | -0.10 |
| attention | 0.38 | 0 | +0.38 |
| small kernels + dispatch | ~4.8 (983 kernels) | 3.27 (479) | ~+1.5 |

## Gemma 4 31B QAT pp512: fork 1729 vs reinstinct 1595 ms traced per forward

| item | fork | reinstinct |
|---|---:|---:|
| Q4_0 GEMM | 1535 (`mmq_gemm_nib_repacked`, was 1711 at R10a, generic 1743) | 1407 |
| attention | 117 | 105 |
| glue + quantize | 76 | ~82 |

The fork's tile runs 8-11% behind reinstinct's on every shape in-model:

| shape | fork (us per call) | reinstinct (us per call) |
|---|---:|---:|
| gate/up | 5807 | 5339 |
| down | 4553 | 4177 |
| 4096-row | 1215 | 1092 |
| 8192-row | 2361 | 2151 |
| 16384-row | 4531 | 4180 |
| 2048-row | 708 | 635 |

Resource use: fork VGPR 124 / LDS 23552 B, reinstinct 128 / 22016. Furnace's standalone harness showed parity on its own card. Whether the remaining gap is card-specific or comes from in-model strides is open.
