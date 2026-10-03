# Plan: close the llama fork's gap to reinstinct on the S1 models

Status: proposed by Reinstinct, 2026-10-03, after an independent review pass (Fable 5.1 subagent: every source claim and trace number re-checked; its corrections are folded in below). Evidence: `R5-decode-attribution.md` plus the prefill traces in section 2. Furnace coordinates and owns implementation; Reinstinct supplies kernel sources and runs the podcast measurements.

## 1. Where the fork is behind (podcast GPU 1, fork 905021dba + B1..R8, reinstinct ee7cbd5)

Decode is untraced GPU time per token from the graph timer; prefill is one pp512 forward, untraced = 512 / pp512 tok/s.

| model | decode fork / reinstinct | prefill pp512 fork / reinstinct |
|---|---:|---:|
| Qwen 3.8-27B Q4_K_XL | 35.1 / 28.1 ms (-20%) | 1901 / 1672 ms (-12%) |
| Gemma 4 31B QAT (Q4_0) | 39.1 / 31.0 ms (-21%) | 1944 / 1601 ms (-18%) |
| Gemma 4 31B Q4_K_XL | 36.5 / 34.4 ms (-6%)* | 2107 / 2016 ms (-4%) |
| Qwen 3.6-35B-A3B Q4_K_XL | 10.5 / 7.7 ms (-27%) | ~level (1729 vs 1741 tok/s) |
| Gemma 4 26B-A4B, E4B | fork ahead or level | fork ahead or level |

\* From tg256 wall time; there is no timer run for this model.

Host time is not the problem anywhere: 0.3-0.6 ms/token in both engines. Everything below is GPU kernel time.

## 2. Attribution (traced; ms per token for decode, ms per pp512 forward for prefill)

Traced deltas are upper bounds, because rocprofv3 adds ~1.1-1.5 us per kernel even when the GPU is fed.

**Decode**

| item | 27B | 31B QAT | 31B Q4_K_XL | 35B-A3B |
|---|---:|---:|---:|---:|
| Q5_K matvec, same grids (row-guard load order) | 3.4 | - | 0.7 | - |
| Q4_K matvec, same grids (cause unknown, see P1b) | <=0.55 | - | 0.5 | - |
| generic Q4_0 MMVQ vs repacked | - | 6.7 | - | - |
| glue / small kernels | 1.5 | ~0 | ~0 | 1.5 |
| MoE down (Q5_K, in_dim 512) | - | - | - | 0.9 |
| dense Q8_0 matvec (cause unknown, see "Investigate") | - | - | - | <=0.9 |
| decode attention, KV <= 16 (fixed cost) | 0.4 | 1.5 | 0.8 | ~0 |
| IQ4_NL / IQ3_S on generic MMVQ | 0.75 | - | - | - |

On Gemma 31B Q4_K_XL, the fork's Q5_K total (5.1 ms) includes the LM head: one call, 262144 rows, 1.67 ms. Reinstinct runs the head as an f32 matvec (2.1 ms). The table counts like-for-like grids only. The LM head is a reinstinct-side item (section 7).

**Prefill (pp512)**

| item | 27B | 31B QAT | 31B Q4_K_XL |
|---|---:|---:|---:|
| generic Q4_0 MMQ (`mul_mat_q<Q4_0>` 1743 ms) vs repacked tile (1407) | - | 337 | - |
| IQ4_XS / IQ4_NL / IQ3_S on generic MMQ (501 ms) vs repacked IQ4_XS tile (351) | 150 | - | - |
| Q3_K tile (3 calls, 13.9 ms each) vs relabel onto the Q6_K tile | ~33 | - | - |
| repacked tiles Q4_K / Q6_K / Q5_K | +9 / -8 / +24 | - | +50 / +25 / +6 |
| attention (`flash_attn_tile` + combine vs `attn_prefill_tiled`) | -4 | +12 | +12 |
| GDN prefill (R9) | 11 | - | - |

## 3. Work items, in order

### P1 (R10a): branchless row loads + multiply-based bit spreading in the dense decode matvecs

**Change (repack-gcn.cu).** Eight kernels guard rows inside the row loop: `mul_mat_vec_q4k_repacked` (guard at line 492), `q5k` (615), `q5_1` (704), `q6k` (883), `q3k` (986), `q8_0_repacked` (1367), `q4k_repacked_glu` (1455), `q8_0_repacked_glu` (1534). Each has `if (row >= (int) ne1) continue;`.
- Clamp instead: `row = min(row0 + r, ne1 - 1)`, keeping the final masked store.
- Hoist each trip's loads (all rows' planes plus the activation block) into register arrays before the first dot. Reinstinct's `kernels/matvec_q5k_repacked.cpp` shows the shape.
- Replace `repack_spread4` with `((h & 0xF) * 0x02040810u) & 0x10101010u`.
- Replace `repack_spread2` with reinstinct's two-multiply form (`kernels/matvec_q6k_repacked.cpp`). The reviewer verified both are exact.
- Q3_K's `spread2_lo` / `spread1_hi`: only take a multiply form after an exhaustive host test.

**Why.** The guard is an execz branch with an `s_waitcnt` between the rows' loads, so only one row's planes are in flight per trip. Reinstinct 54eaa04 measured on the 27B: Q5_K 540 -> 700 GB/s, Q6_K 590 -> 690. The fork's Q5_K today is 552 GB/s on identical grids (2176 WGs: 111 vs 87 us).

**Register budget: this is a spill risk, not just occupancy.**
- `mul_mat_vec_{q4k,q5k,q6k,q3k}_repacked` (lines 440, 565, 824, 931) have **no `__launch_bounds__`**. hipcc then assumes 1024-thread WGs and caps them at 64 VGPRs, so they spill silently (ledger R1).
- They sit at 48 / 48 / 52 / 48 today, and the HAS_IDS q5k/q6k at 60. The hoist adds ~14-16.
- Add `__launch_bounds__(256)` to these four as part of P1. After building, check `Scratch_Size == 0` (rocprof column, or `-Rpass-analysis=kernel-resource-usage`) and VGPR <= 64.
- **Do not hoist in the HAS_IDS half-unit path.** P3 replaces the Q5_K/Q6_K down kernels anyway, and the MoE gate/up is `glu<true,1,1>` (ROWS=1), so there is nothing to hoist there.
- The two glu kernels already have `__launch_bounds__(256, MIN_BLOCKS)`. For them, watch occupancy: two slabs x two rows.

**Validate.**
1. Logits on 3 prompts: bit-identical is *expected*, not guaranteed. HIP's default `-ffp-contract=fast-honor-pragmas` can contract differently once the loop is restructured, so diff per tensor before calling a mismatch a bug.
2. `test-repack-bench --check` (dense, `--x3`, `--ids-view`). test-backend-ops cannot test this: it allocates weights in the default buffer type, not the repack type.
3. ISA: no `s_cbranch_execz` between the rows' `global_load`s, and every load issued before the first `v_dot4`.

**Expected.** 27B -3.4 ms (Q5_K). Gemma 31B Q4_K_XL -0.7 ms. The 35B gains ~0, since its matvecs are ROWS=1 or are replaced by P3.

### P1b: Q4_K matvec — measured: the missing piece is a scheduling fence

Microbench (`scripts/p1b-q4k-matvec-bench.cpp`, podcast GPU 1, same repacked weights, graph replay, 27B shapes): fork kernel as-is, with P1 (clamp + hoist), with P1 + `__builtin_amdgcn_sched_barrier(0)` after the loads, and reinstinct's `matvec_q4k_repacked_f32`.

| out x in | fork | P1 | P1 + fence | reinstinct |
|---|---:|---:|---:|---:|
| 17408 x 5120 (51.5 MB) | 78.0 us (660 GB/s) | 76.4 | **70.1 (735)** | 70.7 (729) |
| 12288 x 5120 | 58.7 | 56.4 | **50.1** | 50.1 |
| 10240 x 5120 | 49.0 | 46.8 | **42.3** | 42.2 |
| 6144 x 5120 | 33.2 | 32.6 | **26.5** | 26.4 |
| 5120 x 17408 | 77.1 | 74.8 | **72.9** | 74.1 |
| 1024 x 5120 | 10.1 | 9.2 | **7.3** | 6.9 |

Clamp + hoist alone gives 2-5%; with the fence the fork matches reinstinct on every shape (the `block_q8_1` vs BlockQ8 activation difference does not matter). Without the fence the scheduler trades latency for registers and splits each trip into load -> wait -> dot phases with a full `vmcnt(0)` between (reinstinct 28f105f). **The fence is per kernel, measured:** reinstinct uses it in Q4_K and IQ4_XS (and its MoE Q4_K, small-batch and Q8_0 MMQ kernels); Q6_K and Q4_0 got *slower* with it, and Q5_K got the loads-first schedule from register pressure alone. So P1 = clamp + hoist everywhere, plus the fence in `q4k_repacked` (and try it per kernel on the rest). Q4_K worth: 27B ~0.55 ms/token, Gemma 31B Q4_K_XL ~0.5.

### P2 (R11): Q4_0 repack, end to end (matvec, fused GLU, short-batch nc, prefill tile, gating)

ggml-cuda has no Q4_0 repack at all (checked; it is not an unset flag). Repacking a type moves every consumer of that weight onto repacked kernels, so this is one unit of work.

1. **Host repack, size and support gate.** Edit repack-gcn.cu: `repack_*_host`, the size switch at line ~39, the support switch at ~55. Use reinstinct's `quant::q4_0::repack_for_matvec` layout:
   - a nibble plane of `ne1*nsp*16` bytes, the on-disk qs copied as-is (byte k = weights k / k+16, which is sdot4 order);
   - an fp16 d plane of `ne1*nsp*2` bytes;
   - the usual `nsp` anti-alias pad.
2. **Decode matvec.** Port reinstinct's `kernels/matvec_q4_0_repacked.cpp` (ROWS=2, plus ROWS=1 for small out_dim) with `dx = __low2float(ds)`.
   - Recommended: take the -8 offset against the *quantized* activation sum, `xqsum = Σ sdot4(0x01010101, xq32[g])`.
   - ggml's `quantize_q8_1` (quantize.cu:86-100, reached via `repack_quantize_x`) stores the *unquantized* float sum in `ds.y`, as fp16. Upstream's generic Q4_0 (vecdotq.cuh:136, `sumi*d - 8*ds.y`) and `mul_mat_q<Q4_0>` use that float sum and run fine. So this is not a correctness bug in the fork.
   - In reinstinct it measured 40x lower per-matvec error (1.6e-4 vs 6.5e-3 rel_l2), the quantization errors cancel instead of being weighted by q_i, and it costs one hoisted sdot4 loop.
   - Expect nonzero KLD vs the unrepacked fork *by design*.
   - Do not apply P1's load hoisting here: reinstinct lost 4-10% on Q4_0, which was already at the bandwidth ceiling.
3. **Fused gate/up GLU kernel (`mul_mat_vec_q4_0_repacked_glu`), and add Q4_0 to `ggml_cuda_repack_should_fuse_glu`** (line 4054 admits only Q4_K and dense Q8_0 today).
   - Today the generic MMVQ already fuses gate+up on the 31B QAT: 60 calls/token at 226 us, each covering both weights.
   - Without a repacked GLU kernel, P2 trades that for two matvecs plus a GLU op, which is the regression documented at lines 1396-1403 (-7% on 35B-A3B).
4. **Short-batch nc matvec** (2..16 columns in chunks of 8, as `mul_mat_vec_kq_repacked_nc`, line ~3441). Without it, MTP verify and any 2..16-token batch falls into the 64-column MMQ tile: the B1 bug again.
5. **Prefill MMQ tile.** Port reinstinct's `kernels/mmq_gemm_q4_0_repacked.cpp`: 1407 ms vs the generic `mul_mat_q<Q4_0>` at 1743 ms per pp512 on the 31B QAT. Alternatively, derive it from the fork's `mmq_gemm_q4k_repacked` minus the scale machinery.
6. **MUL_MAT_ID and other consumers.** No S1 model has Q4_0 experts, so keep 3D Q4_0 expert tensors out of the repack buffer type (gate on `ne[2] == 1`) until HAS_IDS variants exist. Also confirm the remaining paths for a repacked Q4_0 tensor: get_rows (tied embeddings), the multi-GPU split path, and anything else that assumes the canonical layout. The repack buffer must reject what it cannot serve.

**Validate.**
- `test-repack-bench --check`, with Q4_0 dense, GLU and nc shapes added to the bench.
- KLD vs the unrepacked fork on the 31B QAT, 2 x 2048 tokens.
- Greedy output on the S1 prompts.
- `llama-bench -p 2,3,4,8,16`: short batches must stay on the nc path.

**Expected.** 31B QAT decode -6.7 ms (39.1 -> ~32.4). Prefill -337 ms (pp512 263 -> ~319).

### P3 (R4): MoE down for small in_dim. 35B-A3B -0.9 ms/token (~9% of its step)

The fork already has the rows-per-block mapping as `_seg` kernels for Q5_1 (line 744, `ROWS_PER_BLOCK = 256/SEG`) and Q8_0 (1039). Q5_K and Q6_K lack it: their HAS_IDS path uses half-sub-block units (`n_unit = n_sub*2`, lines 603-606), which keeps only 32 of 64 lanes busy at n_sub = 16.
- **Extend `_seg` to Q5_K/Q6_K and add DOWN_R multi-row hoisting.** That means clamped loads for DOWN_R row groups before the first dot, and one LDS reduce; see reinstinct `kernels/moe_matvec_q5k_down.cpp`.
- The grids confirm DOWN_R = 2: fork `(65536, 8)` at 37.8 us vs reinstinct `(16384, 8)` at 13.3 us.
- Furnace's Flash-Next Q5_1 result (K=640, ~0.6% of step) shows the gain is model-dependent; on the 35B-A3B the down kernel is a far bigger share.

**Validate.** `test-repack-bench --check --ids-view`; KLD.

### P4 (rest of R6): glue fusion in decode. 27B and 35B, up to ~1.5 ms each

Small kernels per token: 27B fork 1025 vs reinstinct 479; 35B 911 vs 705. The 27B lists, matched:

| fork kernels / token | reinstinct |
|---|---|
| `add_rms_norm_mul_f32` 107 + `quantize_q8_1` 107 | `add_rmsnorm_q8_f32` 128 (residual add + norm + q8 quantize, 2 per layer) |
| `get_rows` 49 + `concat_cont` 48 + `ssm_conv_f32` 48 + `cpy_scalar` 64 + `rms_norm_f32` 80 (q/k l2) | `conv1d_l2norm_f32` 48 (conv step + state shift + q/k l2 norm) |
| `unary_op` 48 + `add_unary_mul` 48 | GDN gate math inside `gdn_recurrent_step_v2` (R6b did part of this) |
| `rms_norm_scale2_f32` 48 + `unary_gated_op` 46 | `rmsnorm_gated_q8_f32` 48 (gated norm + quantize) |
| `rope_multi` 32 + `k_set_rows` 32 | `q_split_norm_rope_f32` 16 + `k_norm_rope_kv_write_f32` 16 |
| `mul_mat_vec_q8_0_repacked_splitk` 107 | 48 (split-K only on the GDN projections) |
| `unary_gated_q8` 63 | `swiglu_q8_f32` 64 (already fused on both sides; no gain) |

**Priorities.** (a) The conv-step chain: 5 launch types -> 1, 48 layers. (b) Norm + quantize pairs. (c) Split-K only where it pays. Furnace's in-flight MoE-chain folding covers the 35B side. **B2 lesson:** a fused kernel that skips a node keeps that node's inputs alive until the consumer.

### P5 (new R12): IQ4_XS family repack + Q3_K relabel. 27B prefill -150..-183 ms; decode -0.75 ms

Today the fork runs IQ4_XS through the generic MMVQ at decode, which is level with reinstinct's repacked IQ4_XS matvec, so decode needs nothing for IQ4_XS itself. Prefill runs generic `mul_mat_q` on all three formats (449.5 + 44.0 + 7.8 ms) vs reinstinct's repacked IQ4_XS tile at 351.

**Exactness of the relabels:**
- **IQ4_NL -> IQ4_XS (reinstinct layout):** exact (d is already fp16 per 32).
- **Q3_K -> Q6_K:** exact (`quant/q3_k.rs:110`: q6 = q3 + 28, sc6 = sc3 - 32).
- **IQ4_XS -> reinstinct layout:** *not* exact. `iq4_xs.rs:126-131` folds `d*(ls-32)` into an fp16 per-sub-block scale.
- **IQ3_S -> reinstinct layout:** also not exact; `iq3_s.rs:214` folds `d*(1+2s)` the same way.
- Choose: (a) **reinstinct's folded fp16 sub-block scale**, where IQ4_NL is exact, IQ4_XS/IQ3_S round, and you validate by KLD; or (b) a u16 `ls` plane + fp16 d per superblock, the shape the fork's Q4_K planes already have. With (b), IQ4_XS stays exact against ggml's dequant, but IQ4_NL then needs its own scale plane.
- Recommendation: (a). It is proven in reinstinct, the rounding is <= 2^-11 relative per scale, and it keeps one kernel for three formats.

**Kernels.** `mmq_gemm_iq4xs_repacked.cpp` (prefill) and `matvec_iq4xs_repacked.cpp` (decode; needed once the layout changes), with the IQ3_S codebook variant.

**Q3_K relabel.** Saves ~33 ms prefill: the q3k tile takes 13.9 ms/call vs ~2.8 ms for a Q6_K-tile call. At decode it saves only ~0.06 ms (Q6_K reads 1.86x the bytes). The rest of Q3_K's decode cost is the q3k matvec itself (~320 GB/s today), which P1 addresses.

**Validate.**
- Relabels: bit-exact dequant vs ggml's reference, exhaustively over blocks.
- IQ4_XS/IQ3_S under (a): KLD.
- Tiles: `test-repack-bench --check` with IQ4_XS shapes added.

### P6 (R3 class): Gemma decode attention. 31B 0.8-1.5 ms/token at short context; measure at length first

The fork runs `flash_attn_tile` at decode for Gemma's 256-dim local and 512-dim global heads, 120 calls/token. That kernel runs WG = 32 on wave64 at 76-88 VGPRs; the q35 instance `<256,256,1,8>` shows 128 VGPRs + 8 scratch.

The traced numbers are at KV <= 16 (`-n 16`), so they measure fixed cost, not bandwidth.
1. First trace both engines at 4K and 16K context (Gemma 31B QAT and Q4_K_XL).
2. Reinstinct's Gemma path is `attn_decode_gqa_q8_f32`: int8 KV, i.e. R3 proper (784 + 259 us/token at short context).
3. For the fork's F16 KV, port the GQA flash-decoding *structure*: one WG per (kv head, split), the group's q heads in registers, device-side split count, merge kernel. Use F16 loads.
4. R3's int8 KV comes after that as a separate decision.

**Validate.** test-backend-ops FLASH_ATTN_EXT at decode shapes (attention takes no repacked weights, so the stock test works here); KLD.

### P7: dense K-quant prefill tiles (Gemma 31B ~81 ms, 27B Q5_K ~24 ms per pp512)

- Q4_K tile: 1424 vs 1374 ms. Q6_K: 303 vs 278 (after R7). Q5_K: +24 on the 27B.
- Reinstinct first compares tile parameters (BK, prefetch depth, LDS padding, store path) against its `mmq_gemm_{q4k,q5k,q6k}_repacked.cpp`, then proposes specific changes.
- Prefill attention is +12 ms on the Gemma 31Bs (fork slower). Fold that into the P6 work if the decode port shares code.

### Investigate before porting

- **Dense Q8_0 decode (35B, <=0.9 ms).** The fork's `mul_mat_vec_q8_0_repacked<1,1,false>` runs 131 calls at 32 us. Reinstinct runs 71 x 29 us + 180 x 8.6 us.
  - The fork's half-sub-block lanes already read 16 contiguous bytes each (lines 1351-1370), so reinstinct's two-plane layout is not obviously the cause.
  - Microbenchmark both kernels on the 35B's Q8_0 shapes (grids are in the trace) before porting anything.
- **R9 (GDN prefill VGPR):** ~11 ms per pp512 on the 27B (~0.6%). Stays parked.

## 4. Sequencing and ownership

| step | item | owner | depends on | main risk |
|---|---|---|---|---|
| 1 | P1 R10a (+ launch bounds) | Furnace | - | VGPR spill in hoisted kernels |
| 2 | P1b Q4_K: done (fence); fold into P1 | Furnace | - | - |
| 3 | P2 R11: matvec -> glu -> nc -> tile -> gating | Furnace; Reinstinct drops source extracts in `patches/reinstinct-to-llama/R11/` | - | GLU fusion gate, consumers of the new layout |
| 4 | P3 R4 down (seg for Q5_K/Q6_K + DOWN_R) | Furnace | - | - |
| 5 | P4 glue | Furnace (MoE part in progress) | - | fusion liveness |
| 6 | P5 R12 | Furnace; Reinstinct extracts | P1 | layout choice (a)/(b) |
| 7 | P6 attention | Reinstinct traces at length first, then Furnace | - | ggml FA dispatch rules |
| 8 | P7 tiles, Q8_0 investigation | Reinstinct compares first | - | - |

After each landed step, Reinstinct reruns S1 on podcast GPU 1 (same script and env, fork worktree at the new patch set) and appends to `baselines/S1-podcast-1xMI50.md`. It also reruns the graph timer and `scripts/decode_cmp_fmt.py` when the attribution changes. Furnace checks Flash-Next tg/pp for regressions after each step.

## 5. Expected end state (traced upper bounds; no double counting)

**Decode**

| model | now | after P1-P6 | items | reinstinct |
|---|---:|---:|---|---:|
| Qwen 3.8-27B | 35.1 | ~28.5 ms | P1 3.4 + 0.55 (Q4_K fence), P4 1.5, P5 0.75, P6 0.4 | 28.1 |
| Gemma 31B QAT | 39.1 | ~30.9 ms | P2 6.7, P6 1.5 | 31.0 |
| Gemma 31B Q4_K_XL | 36.5 | ~34.5 ms | P1 0.7 + 0.5 (Q4_K fence), P6 0.8 | 34.4 |
| Qwen 3.6-35B-A3B | 10.5 | ~8.1 ms | P3 0.9, P4 1.5 (+Q8_0 investigation <=0.9 -> ~7.2) | 7.7 |

**Prefill (pp512)**

| model | now | after | items | reinstinct |
|---|---:|---:|---|---:|
| Qwen 3.8-27B | 1901 | ~1718 ms | P5 150 + 33 (with P7 ~1694) | 1672 |
| Gemma 31B QAT | 1944 | ~1607 ms | P2 337 | 1601 |
| Gemma 31B Q4_K_XL | 2107 | ~2026 ms | P7 only (~81) | 2016 |

## 6. Measurement protocol (both machines)

- **Correctness, every step.**
  - `test-repack-bench --check` for repacked kernels (extend it with each new format's shapes); test-backend-ops only for non-repack ops such as FA.
  - Bit-identical logits where a step expects it (P1; relabels via exact dequant).
  - Otherwise KLD over 2 x 2048 tokens vs the previous build. On GDN models expect ~0.02 from reordering alone (Furnace's noise floor).
  - Greedy output on the S1 prompts.
- **Decode timing.** Graph timer (`scripts/graph-timer/fork-ggml-cuda.diff`, `GGML_GRAPH_TIMER=1`) for GPU ms per replay; `llama-bench -n 128` for wall time.
- **Kernel attribution.**
  - rocprofv3 with `GGML_CUDA_DISABLE_GRAPHS=1` (graph replay crashes under the profiler on some models), then `scripts/decode_cmp_fmt.py`.
  - Trust a trace only when traced busy is within ~1.5 us/kernel of the untraced GPU time.
  - Check `Scratch_Size` and `VGPR_Count` in the same CSV after every kernel change.

## 7. Reinstinct-side items found on the way (not part of this plan)

- Gemma 31B LM head: reinstinct runs a 2.1 ms/token f32 matvec where the fork runs Q5_K in 1.67 ms. Gemma 31B Q4_K_XL also has more glue launches in reinstinct (979 vs 731/token).
- Gemma 26B-A4B: the fork still leads prefill (-5%) and decode (-2%).
- Prefill attention: reinstinct's `attn_prefill_tiled` is ~4 ms slower than the fork's on the 27B.
