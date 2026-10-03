> **Decode section superseded by R5-decode-attribution.md.** The reinstinct decode trace here ran with idle gaps under the profiler (graphs on, 49 ms/token traced vs 8 untraced), which inflated its small kernels by ~5 us each, so "kernel time level, the lead is glue" is wrong: fork dense/MoE matvecs are slower too. The prefill section stands.

# R6 / L8 attribution: Qwen 3.6-35B-A3B Q4_K_XL, podcast GPU 1

Kernel traces via rocprofv3 (AMD rocprofiler-sdk 7.1 + AMD HIP/HSA runtime libs unpacked under ~/opt on podcast; the distro runtimes have no rocprofiler-register hook). Fork 905021dba (`GGML_CUDA_DISABLE_GRAPHS=1` for the decode trace - the fork crashes in graph compute under the profiler), reinstinct 3a46560+. Aggregated with scripts/kernel_families.py. file0 = fork, file1 = reinstinct. Profiled kernel times run above untraced wall time (tg: fork 11.3 ms, reinstinct 8.4 ms per token untraced); compare families, not totals against tok/s.

## Decode (17 steps: llama-bench -p 0 -n 16 -r 1 incl. warm-up / generate-text --tokens 1000 -n 16)

```
ms per step (kernel time; summed over GPUs if several)
family                                 file0       file1
GDN / linear attention                 0.453       0.560
attention                              0.210       0.307
MoE router / top-k                     0.472       0.843
MoE gate/up (decode matvec)            0.000       1.559
MoE down (decode matvec)               0.000       0.782
MoE expert GEMM (prefill)              0.000       0.000
LM head                                0.000       0.000
dense quant GEMM (MMQ)                 0.000       0.000
dense quant matvec                     8.021       6.774
f32/f16 matmul                         0.000       0.000
norm / elementwise / copies            3.159       1.263
rest                                   0.294       0.500
TOTAL                                 12.610      12.588
file0: fork_tg_kernel_trace.csv  steps 17
    rest:    0.210 ms  moe_weighted_reduction_f32
    rest:    0.084 ms  void k_set_rows<float, long, __half>
file1: ri_tg_kernel_trace.csv  steps 17
    rest:    0.238 ms  moe_combine_f32
    rest:    0.205 ms  swiglu_q8_f32
    rest:    0.049 ms  sigmoid_mul_q8_f32
    rest:    0.005 ms  swiglu_mul_f32
    rest:    0.003 ms  embed_lookup_q8_0_v_f32
```

## Prefill pp512 (fork: warm-up + 1 run = 2 forwards; reinstinct: 1 forward)

```
ms per step (kernel time; summed over GPUs if several)
family                                 file0       file1
GDN / linear attention                44.891      95.243
attention                              7.208       0.000
MoE router / top-k                     1.897       1.594
MoE gate/up (decode matvec)            0.000       1.544
MoE down (decode matvec)               0.000       0.000
MoE expert GEMM (prefill)             62.570     250.664
LM head                                0.000       0.000
dense quant GEMM (MMQ)               134.790      90.627
dense quant matvec                     0.661      32.368
f32/f16 matmul                        30.636      59.014
norm / elementwise / copies           23.986      17.778
rest                                  16.418      26.868
TOTAL                                323.057     575.701
file0: fork_pp_kernel_trace.csv  steps 2
    rest:   13.976 ms  moe_weighted_reduction_f32
    rest:    1.678 ms  void soft_max_f32<true, 256, 256, float>
    rest:    0.252 ms  void repack_tile_map<16>
    rest:    0.173 ms  void reduce_rows_f32<false>
    rest:    0.172 ms  void op_clamp_kernel<float>
file1: ri_pp_kernel_trace.csv  steps 1
    rest:    7.134 ms  attn_prefill_tiled_f32
    rest:    5.490 ms  moe_combine_f32
    rest:    4.156 ms  moe_scatter_rows
    rest:    2.208 ms  moe_sort_scan
    rest:    1.817 ms  swiglu_mul_f32
```

## Top kernels

```
#### fork_pp_kernel_trace.csv (ms per forward, top 12)
      66.323    250x  void mmq_gemm_q8_0_repacked<false, 4>
      60.835     78x  void mmq_gemm_q4k_repacked_id_w1<2>
      53.872     38x  void mmq_gemm_q5k_repacked<true, 1>
      40.332     30x  void gated_delta_net_lds_wave64<16, false>
      29.657    100x  void gcn_f32_gemm_tn_rb<true>
      14.595      4x  void mmq_gemm_q6k_repacked<true, 1>
      13.976     40x  moe_weighted_reduction_f32
       5.670     10x  void flash_attn_tile<256, 256, 4, 8, false>
       4.560     30x  void ssm_conv_long_token_f32<true, 128ul, 4ul, 32l>
       3.302     30x  void concat_dim0_transpose_src1<unsigned int>
       2.648     79x  void rms_norm_mul_q8_f32<1024>
       2.511    122x  quantize_q8_1
#### ri_pp_kernel_trace.csv (ms per forward, top 12)
     140.943     72x  mmq_gemm_q5k_grouped_f32
      97.857    156x  mmq_gemm_q4k_grouped_f32
      90.627    370x  mmq_gemm_q8_0_repacked_f32
      87.033     30x  gdn_recurrent_batched_v2_f32
      59.014    140x  gemm_f16_rows_f32
      20.796      2x  moe_matvec_q6k_repacked_f32
      11.865      6x  mmq_gemm_q6k_grouped_f32
      10.916      4x  moe_matvec_q5k_repacked_f32
       8.210     30x  conv1d_step_silu_batched_f32
       7.134     10x  attn_prefill_tiled_f32
       6.971    531x  quantize_q8_f32
       5.490     80x  moe_combine_f32
#### fork_tg_kernel_trace.csv (ms per forward, top 12)
       4.183    131x  void mul_mat_vec_q8_0_repacked<1, 1, false>
       1.403     38x  void mul_mat_vec_q5k_repacked<true>
       0.865     39x  void mul_mat_vec_q4k_repacked_glu<true, 1, 1>
       0.756    100x  void gcn_f32_matvec_rows<0>
       0.522     80x  void rms_norm_mul_q8_f32<1024>
       0.521    110x  void k_bin_bcast<&
       0.472     40x  void topk_moe_cuda<256, false>
       0.349     81x  quantize_q8_1
       0.323     30x  void gated_delta_net_cpw<2, false>
       0.291     40x  void mul_mat_vec_q8_0_repacked_glu<1>
       0.253     60x  void rms_norm_scale_f32<256>
       0.212     50x  void rms_norm_f32<256, true, false>
#### ri_tg_kernel_trace.csv (ms per forward, top 12)
       3.779     71x  matvec_q8_0_repacked_r1_f32
       2.098    180x  matvec_q8_0_repacked_f32
       1.185     39x  moe_gate_up_swiglu_q4k_repacked_f32
       0.861     70x  matvec_f32_b256
       0.843     40x  moe_topk_f32
       0.701     36x  moe_matvec_q5k_down_f32
       0.559     80x  add_rmsnorm_q8_f32
       0.386     30x  gdn_recurrent_step_v2_f32
       0.373     40x  moe_shexp_gate_f32
       0.254     10x  attn_decode_gqa_f32
       0.238     40x  moe_combine_f32
       0.221     41x  add_inplace_f32
```

## Reading (from the top-kernel lists; the family buckets misfile some ggml MoE kernels - see last bullet)

- **L8, prefill pp512, per forward** (fork vs reinstinct):
  - MoE experts 2.2x: fork ~129 ms (q4k_repacked_id_w1 gate+up 60.8, q5k_repacked<true,1> down 53.9, q6k 14.6) vs reinstinct ~282 ms (q5k_grouped down 140.9, q4k_grouped gate+up 97.9, q6k_grouped 11.9, plus 31.7 of moe_matvec fallback on 6 layers).
  - GDN recurrence 2.2x: gated_delta_net_lds_wave64<16> 40.3 vs gdn_recurrent_batched_v2 87.0.
  - F32 GEMMs (router 2048x256, alpha/beta) 2x: gcn_f32_gemm_tn_rb 29.7 vs gemm_f16_rows 59.0.
  - Dense Q8_0 GEMM 1.4x: mmq_gemm_q8_0_repacked<false,4> 66.3 vs ours 90.6.
  - Reinstinct to take, in order: the fork's MoE id-GEMM (q4k_id_w1 / q5k repacked down), its GDN prefill kernel, its F32 GEMM, its Q8_0 MMQ tile.
- **R6, decode**: summed kernel time per step is level (~12.6 ms both, traced). Reinstinct's untraced lead comes from fewer, fused small kernels: norm/elementwise/copies 1.26 vs 3.16 ms/step (add_rmsnorm_q8, sigmoid_mul_q8, swiglu_q8 fused vs rms_norm + bin_bcast + quantize_q8_1 + concat), and fewer launches. Kernel by kernel the fork is ahead in places: dense Q8_0 matvec 4.18 ms (131 launches) vs ours 5.88 (251: matvec_q8_0_repacked_r1 + _f32), top-k 0.47 vs 0.84, F32 matvec 0.76 vs 0.86. Ours ahead: MoE down 0.70 vs 1.40 (q5k), GDN step 0.39 vs 0.32+ (gated_delta_net_cpw) about level.
- Family regexes: ggml's MoE kernels (`*_id_w*`, `mul_mat_vec_*_glu`, `mmq_gemm_q5k_repacked<true,...>` used for ids) and reinstinct's `matvec_f32_b256`/`gemm_f16_rows` need entries in kernel_families.py for the bucket view to be right.
