# Cross-port ledger

Direction: **R->L** reinstinct to llama fork, **L->R** llama fork to reinstinct, **S** shared.
Owner: who does the port/evaluation. See README for status values.

## R->L

| id | item | source | owner | status | notes / evidence |
|---|---|---|---|---|---|
| R1 | `__launch_bounds__` / VGPR-spill audit of batched matvec and other kernels (hipcc assumes 1024-thread WGs without it, caps VGPRs at 64, spills silently) | reinstinct 1d23621, f000b64, 805e82d (audit of all 200 reinstinct kernels: only the batched matvecs + one fallback matvec were actually hurt) | Furnace | measured | Fork result 2026-10-03 (code-object metadata of the production libggml-hip, 7114 gfx906 kernels): every hot kernel already declares `__launch_bounds__`; 0 kernels spill from the 1024-thread default, so the reinstinct fix has nothing to apply to. Spills despite bounds: `flash_attn_tile` family (prefill; the variant production runs, <256,256,8,4>, spills 24 VGPRs and is 2-3% of pp2048 kernel time) and `fattn_dec_chunk<256,16>` (8 VGPRs at the 256 cap; production decode uses <256,12>, no spill). Low priority. |
| R2 | small-batch matvec n=1..8: exact row counts n1..n4 + masked n8, loads behind `sched_barrier(0)`, 4 output rows/wave, activations staged in LDS per 64-sub-block chunk; Q8_0 batched path | reinstinct 1d23621, f000b64 (branch `review-fixes`): `kernels/matvec_batched_nr.h` (shared body), `kernels/matvec_batched_nr_entries.h` (rows/LDS per entry), `kernels/matvec_*_repacked_batched.cpp` (per-format decoders), host table `src/runtime/prefill.rs` `SmallBatchMatvec` | Furnace | proposed | Reinstinct: Q4_K n4 21504x5376 0.228->0.114 ms, Q6_K n4 0.438->0.199, Q8_0 n4 0.516->0.185; Qwen3.8-27B MTP K=2 17.0->38.4 tok/s. Compare against fork's dense Q8_0 nc/broadcast matvec and repacked glu16 first. |
| R3 | int8-KV GQA flash-decoding: scores-pass lane mapping with 16-32 B loads, device-side split count (graph-safe), 32 splits for long full-attention layers | reinstinct 1e11d8f, 7344c46: `kernels/attn_decode_gqa_q8.cpp`, defines per layer kind `src/runtime/gemma4.rs` `gqa_q8_defs` | Furnace | proposed | Gemma 31B global layer @32K 1.07->0.61 ms. Fork runs F16 KV today; evaluate against fattn_dec_chunk/combine. |
| R4 | MoE down-proj, 2 rows per group | reinstinct (earlier) | Furnace | proposed | 35B MoE +7.9%. Fork's down path is repacked Q5_1 seg kernel (Flash-Next); our dedup attempt on down was slower (96 vs 71 us). |

## L->R

| id | item | source | owner | status | notes / evidence |
|---|---|---|---|---|---|
| L1 | Adaptive MTP serving policy: drafting on/off globally by number of generating slots (`--spec-draft-max-slots N`, env `LLAMA_SPEC_MAX_GEN`), fixed depth 2, drafter keeps ingesting every target batch while off | fork qwen4exp-mtp a6c082a43, af566d8a1 | Reinstinct | proposed | Same protocol as Rune Prod (GLM-5.3, 10x MI100). See notes/llama-mtp.md. |
| L2 | MTP draft-loop details for a single trained head (Qwen NextN): catch-up of every target batch through the MTP layer with target hidden states, carry-over row, chained steps on the head's own hidden, accept() picks the verified row | fork common/speculative.cpp (PR 28243 port) | Reinstinct | measured | reinstinct 5496386 (review-fixes): catch-up from target hiddens (prompt + accepted rows), true MTP positions, prefix accept, per-row GDN/conv checkpoints written by the verify kernels (rollback 13 -> 0.6 ms/round). Qwen 3.8-27B Q4_K_XL, 1x MI50, greedy, 128 tok: K=1 85.5% / 38.8 tok/s, K=2 75.0% / 43.3 tok/s (1.35x vs 32.1 plain, prefill included in both), K=3 63.6% / 42.8; with the LM head batched (3a46560) K=2 45.0 tok/s (1.40x), K=3 44.9. Spec output == plain greedy token for token. The old "35%" was all-or-nothing rounds (p^K). 27B has no hyper-connections. Next on our side: verify forward cost (uncaptured, per-row LM head). |
| L3 | Multi-query FA decode: verify rows (up to 16 per seq) through the decode gather/chunk/combine path | fork 314cab9ca / c85fd8185 | Reinstinct | proposed | pp3 @32K attention 5.53->1.55 ms/step on Flash-Next. |
| L4 | GDN decode: 4 state columns per wave for >=2 seqs; state-gather elision only when n_seqs==1 or n_seq_tokens==1 (race otherwise) | fork 82fb75a6f, abe4902f6 | Reinstinct | proposed | Applies to Qwen 3.6/3.8 GDN layers. The race made multi-seq outputs nondeterministic. |
| L5 | MoE expert dedup for few-token gate/up (wave-wide ids + ballot, first (token,slot) owns the expert) | fork 99e97eafc (scheme from Rune Prod) | Reinstinct | proposed | glu16 91.4->80.8 us/call, bit-identical; MTP 1 user +0.9%, 2 users +1.7%. Matters for 35B-A3B / 26B-A4B verify. |
| L6 | topk_moe rank kernel | fork 4a8dee414 | Reinstinct | proposed | 18 -> ~4 us per call. |
| L7 | MoE dispatch rejection logging: one-time log of why a fast kernel did not dispatch | Rune Prod tip | both | proposed | Rune found a dedup kernel silently not dispatching for months. |

## S

| id | item | owner | status | notes |
|---|---|---|---|---|
| S1 | Like-for-like baselines: fork vs reinstinct on the same MI50, same models (Gemma 4 31B/26B-A4B/E4B, Qwen 3.8-27B, Qwen 3.6-35B-A3B), same settings | Reinstinct runs on podcast; Furnace supplies build + flags | measured | baselines/S1-podcast-1xMI50.md (fork 905021dba vs reinstinct e2d4740, podcast GPU 1). Decode: reinstinct +3..+34% except Gemma 26B-A4B (-5%); prefill: reinstinct +14-21% on dense, fork +54-79% on MoE/E4B. Fork draft-mtp on Qwen 3.8-27B dense: 71-82% accept but ~9 tok/s vs 27.5 plain - needs a look on the fork side. |
| S2 | DFlash | Reinstinct leads | proposed | Fork evaluated it 2026-08 for Qwen3.5-122B: no llama.cpp/GGUF path then, vLLM/SGLang only. Reinstinct has a WIP; fork follows. |
