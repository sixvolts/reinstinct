# Cross-port ledger

Direction: **R->L** reinstinct to llama fork, **L->R** llama fork to reinstinct, **S** shared.
Owner: who does the port/evaluation. See README for status values.

## R->L

| id | item | source | owner | status | notes / evidence |
|---|---|---|---|---|---|
| R1 | `__launch_bounds__` / VGPR-spill audit of batched matvec and other kernels (hipcc assumes 1024-thread WGs without it, caps VGPRs at 64, spills silently) | reinstinct 1d23621, f000b64 | Furnace | proposed | First: ggml `mul_mat_vec_q` ncols_dst 2..8 and the fork's repack-gcn kernels via `-Rpass-analysis=kernel-resource-usage`. Directly on the MTP verify path (3 rows at n_max 2). |
| R2 | small-batch matvec n=1..8: exact row counts n1..n4 + masked n8, loads behind `sched_barrier(0)`, 4 output rows/wave, activations staged in LDS per 64-sub-block chunk; Q8_0 batched path | reinstinct 1d23621, f000b64 | Furnace | proposed | Reinstinct: Q4_K n4 21504x5376 0.228->0.114 ms, Q6_K n4 0.438->0.199, Q8_0 n4 0.516->0.185; Qwen3.8-27B MTP K=2 17.0->38.4 tok/s. Compare against fork's dense Q8_0 nc/broadcast matvec and repacked glu16 first. |
| R3 | int8-KV GQA flash-decoding: scores-pass lane mapping with 16-32 B loads, device-side split count (graph-safe), 32 splits for long full-attention layers | reinstinct 1e11d8f, 7344c46 | Furnace | proposed | Gemma 31B global layer @32K 1.07->0.61 ms. Fork runs F16 KV today; evaluate against fattn_dec_chunk/combine. |
| R4 | MoE down-proj, 2 rows per group | reinstinct (earlier) | Furnace | proposed | 35B MoE +7.9%. Fork's down path is repacked Q5_1 seg kernel (Flash-Next); our dedup attempt on down was slower (96 vs 71 us). |

## L->R

| id | item | source | owner | status | notes / evidence |
|---|---|---|---|---|---|
| L1 | Adaptive MTP serving policy: drafting on/off globally by number of generating slots (`--spec-draft-max-slots N`, env `LLAMA_SPEC_MAX_GEN`), fixed depth 2, drafter keeps ingesting every target batch while off | fork qwen4exp-mtp a6c082a43, af566d8a1 | Reinstinct | proposed | Same protocol as Rune Prod (GLM-5.3, 10x MI100). See notes/llama-mtp.md. |
| L2 | MTP draft-loop details for a single trained head (Qwen NextN): catch-up of every target batch through the MTP layer with target hidden states, carry-over row, chained steps on the head's own hidden, accept() picks the verified row | fork common/speculative.cpp (PR 28243 port) | Reinstinct | proposed | Reinstinct sees ~35% acceptance at K=3; fork sees ~76% (greedy) / ~53% (temp 1.0) at K=2 on Flash-Next. notes/llama-mtp.md lists the likely causes. |
| L3 | Multi-query FA decode: verify rows (up to 16 per seq) through the decode gather/chunk/combine path | fork 314cab9ca / c85fd8185 | Reinstinct | proposed | pp3 @32K attention 5.53->1.55 ms/step on Flash-Next. |
| L4 | GDN decode: 4 state columns per wave for >=2 seqs; state-gather elision only when n_seqs==1 or n_seq_tokens==1 (race otherwise) | fork 82fb75a6f, abe4902f6 | Reinstinct | proposed | Applies to Qwen 3.6/3.8 GDN layers. The race made multi-seq outputs nondeterministic. |
| L5 | MoE expert dedup for few-token gate/up (wave-wide ids + ballot, first (token,slot) owns the expert) | fork 99e97eafc (scheme from Rune Prod) | Reinstinct | proposed | glu16 91.4->80.8 us/call, bit-identical; MTP 1 user +0.9%, 2 users +1.7%. Matters for 35B-A3B / 26B-A4B verify. |
| L6 | topk_moe rank kernel | fork 4a8dee414 | Reinstinct | proposed | 18 -> ~4 us per call. |
| L7 | MoE dispatch rejection logging: one-time log of why a fast kernel did not dispatch | Rune Prod tip | both | proposed | Rune found a dedup kernel silently not dispatching for months. |

## S

| id | item | owner | status | notes |
|---|---|---|---|---|
| S1 | Like-for-like baselines: fork vs reinstinct on the same MI50, same models (Gemma 4 31B/26B-A4B/E4B, Qwen 3.8-27B, Qwen 3.6-35B-A3B), same settings | Reinstinct runs on podcast; Furnace supplies build + flags | proposed | Fork build: branch `gfx906-perf`, `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, env `GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1`, `-fa on`. |
| S2 | DFlash | Reinstinct leads | proposed | Fork evaluated it 2026-08 for Qwen3.5-122B: no llama.cpp/GGUF path then, vLLM/SGLang only. Reinstinct has a WIP; fork follows. |
