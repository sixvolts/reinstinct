# Porting reinstinct's gfx906 Wins to a llama.cpp Branch

Status: **proposal** (no code yet). Written 2026-06-09 against reinstinct
`7511836` and llama.cpp master `b22ff4b7b` (local checkout at
`~/llama.cpp`, built for gfx906 with `GGML_HIP_GRAPHS=ON`; carries two
local patches to `solve_tri.cu` + `ggml-cuda.cu` capping GCN triangular
solves at 64×64).

## Goal

reinstinct beats llama.cpp on MI50 by **+17-43% decode** and up to
**+24% prefill** (README tables, measured against a current HIP-graphs
build). reinstinct only serves two model families; llama.cpp serves
everything. A llama.cpp branch carrying the transferable kernels gives
the gfx906 community most of the win without adopting a whole new
engine — and tells us which wins are engine-architecture wins vs
kernel wins.

## What llama.cpp already has (don't port these)

Audit of master `b22ff4b7b`, with file references:

| Capability | Where | Status on gfx906 |
|---|---|---|
| dp4a via `__builtin_amdgcn_sdot4` | `common.cuh:690-728` (explicit `__gfx906__` case) | yes |
| Int8 MMQ tile GEMM for prefill | `mmq.cu` — `ggml_cuda_should_use_mmq` enables it unconditionally for GCN (falls through the CDNA/WMMA gates) | yes, dp4a path, wave64-aware (4 warps × 64) |
| Split-K decode attention | `fattn-common.cuh:1061-1127` — `parallel_blocks` KV-split with partial-result merge | yes (vec path) |
| Quantized KV cache in FA | `fattn-vec.cuh:572+` — F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/BF16 | yes |
| HIP graph capture | `GGML_HIP_GRAPHS=ON` (already on in the local build) | yes |
| MoE grouped dispatch | `mmid.cu` — `mm_ids_helper` sorts tokens per expert, feeds MMQ with permutation indices | yes |
| GCN-specific launch params | `mmvq.cu:64-100` — `MMVQ_PARAMETERS_GCN` table | yes (shared with CDNA, untuned for Vega20) |

So the structural ideas — MMQ, split-K, KV quant, graphs, MoE grouping
— all exist. The decode gap is **not** a missing-feature gap. It comes
from layout and per-op overhead, which is exactly what the portable
items below address.

## Where reinstinct's decode win actually comes from

For a bandwidth-bound decode, the only things that matter are
(a) bytes moved, (b) achieved bandwidth on the weight stream,
(c) overhead per token. reinstinct wins on (b) and (c):

1. **K-quant plane repacking** (`src/quant/q4_k.rs:126-176` + the
   `matvec_*_repacked` kernel family). On-disk K-quant superblocks
   interleave nibbles/scales/minima, so a streaming wave64 read is only
   ~58% coalesced. reinstinct repacks at load into three contiguous
   planes (nibbles / sub-block scales / superblock d+dmin) with one
   sub-block of padding when the count is a power of two (kills a 3×
   bank-alias). Measured: **~470 → ~740 GB/s effective** on the Gemma
   FFN shape — +53% on the dominant decode cost. llama.cpp's `mmvq.cu`
   reads the on-disk layout directly; this is the largest single delta.
2. **One fused step ≈ no dispatch overhead.** reinstinct decodes a
   token in ~180 kernel launches captured in a HIP graph with fused
   norm/rope/write paths. llama.cpp's graph capture helps but its op
   granularity is fixed by ggml graphs.
3. **GCN-tuned reductions** (`kernels/gfx906_dpp.h`) — DPP/ds_swizzle
   wave64 reductions instead of `__shfl_xor` cascades, used in every
   softmax/norm/dot epilogue.

## Port plan, ranked by expected ROI

### P1 — Repacked K-quant decode matvec (the big one)

**What:** load-time repack of Q4_K/Q5_K/Q6_K (+Q8_0's simpler variant)
into the three-plane layout, plus the matching matvec kernels, gated to
GCN.

**Integration shape:** llama.cpp already has the precedent — the CPU
backend's `repack` buffer type (formerly aarch64). Add an opt-in HIP
buffer type (`GGML_HIP_REPACK=1` or auto-on for `GGML_CUDA_CC_IS_GCN`)
that transforms supported tensor types in `set_tensor`, and route
`mul_mat_vec` for repacked tensors to ported kernels. The repack is
self-contained host code (a few hundred lines of Rust in
`src/quant/q{4,5,6}_k.rs` that translates mechanically to C++).

**Kernels to port:** `matvec_q4k_repacked.cpp`, `_q5k_`, `_q6k_`,
`matvec_q8_0_repacked.cpp` (with its ROWS=1/ROWS=2 out_dim dispatch,
+13% on MoE shapes), and the batched ≤4-row variants if spec-decode
batching matters.

**Complications:** MMQ prefill must either keep a second (on-disk
layout) copy — VRAM cost — or also learn the repacked layout, the way
reinstinct's `mmq_gemm_*_repacked.cpp` kernels already do. Port both
and the layout is consistent end to end. `get_rows`, `cpy`, and
quant-aware ops touching weight tensors must be blocked or taught the
layout (the buffer-type mechanism handles this: unsupported ops fall
back per ggml's buft rules — verify the fallback set is decode-clean).

**Expected:** the +53% effective-bandwidth delta on the matvec is the
dominant share of reinstinct's +33% decode win on Gemma 31B. Even
landing half of it makes this the highest-value item by far.

**Effort:** 1-2 weeks. The kernels are written; the work is the buffer
type, the MMQ-side layout unification, and `test-backend-ops` coverage.

### P2 — GCN DPP reduction header

**What:** port `gfx906_dpp.h` into `common.cuh`'s `warp_reduce_sum` /
`warp_reduce_max` under `#if defined(GCN)` (the define already exists
in `vendors/hip.h:182-192`). Five DPP/swizzle ops replace six shfl
rounds in every warp reduction backend-wide.

**Expected:** small per-kernel (these reductions hide behind memory
latency in the big kernels) but it lands everywhere — fattn epilogues,
norms, MMVQ dots. Low single digits on decode; nearly free to do.

**Effort:** 1-2 days incl. `test-backend-ops` on gfx906.
**Upstreamable:** yes — clean, self-contained, arch-gated. Best
candidate for a mainline PR rather than a branch-only patch.

### P3 — GDN fused decode step for Qwen3.5/Next-style hybrids

**What:** llama.cpp runs Gated-DeltaNet decode as a chain of generic
ggml ops (the local `solve_tri.cu` patches are part of this path's
chunked prefill). reinstinct fuses the whole per-token recurrence —
beta quantization, state decay, rank-1 update, Q·S — into one kernel
with LDS-resident state and a bank-conflict-padded layout
(`gdn_recurrent_step_fused*.cpp`, 3.84× on the op after the padding
fix).

**Integration shape:** a fused `GGML_OP` path is invasive; the
realistic version is a custom op / graph-rewrite in the HIP backend
that pattern-matches the delta-net decode subgraph, the way backends
already fuse rope+rms patterns. Branch-only material.

**Expected:** on MI50, reinstinct's Qwen 27B GDN decode is +17-21% vs
llama.cpp; the GDN chain is the structural difference. Pairs naturally
with the existing local solve_tri work.

**Effort:** 1-2 weeks; needs careful state-layout mapping between
ggml's ssm tensors and the fused kernel's expectations.

### P4 — dp4a Q·K for fattn-vec on GCN

**What:** llama.cpp's `fattn-vec` with Q8_0 KV dequantizes per element
for the Q·K dot. reinstinct's `attn_partial_q8.cpp` quantizes Q to int8
per head and uses `sdot4` — 2× on the attention kernel itself at 48 dB
SNR.

**Caveat from our own data:** on reinstinct workloads attention is a
small slice of decode (GDN models: ~1.7%; an int8-KV port to Qwen was
measured byte-identical but 0% wall-clock and reverted). It matters at
long context and for dense attention-heavy models only. Port after P1
proves out, measure honestly, be prepared to drop it.

**Effort:** ~1 week (vec_dot specialization inside fattn-vec's
template machinery).

### P5 — gfx906 launch-parameter tuning sweep

**What:** `MMVQ_PARAMETERS_GCN` is shared between GCN and CDNA and was
never Vega20-tuned; same for MMQ tile counts on GCN. Run the standard
sweep methodology (3-pass tok/s averages) over mmvq rows-per-block /
nwarps and MMQ tile configs on MI50, contribute a Vega20 table.

**Effort:** 2-3 days, mostly machine time. **Upstreamable:** yes.

### Not worth porting

- **HIP graphs, split-K FA, MoE grouping** — already present (see
  table above).
- **SuperQuant tiered KV** — capacity feature with a −30% decode cost,
  architecturally bespoke (tier cascade defeats ggml's KV model).
- **Adaptive-K MTP** — llama.cpp has its own speculative framework;
  the *idea* (rolling-α auto-disable of speculation) could be filed as
  a llama-server heuristic proposal, but our drafter and verify path
  don't map.
- **PLE/KV-sharing (Gemma E4B)** — llama.cpp already supports these
  models.

## Branch strategy

gfx906 is deprecated in ROCm and mainline llama.cpp treats it as
legacy; a large GCN-only buffer type (P1) is unlikely to be accepted
upstream, and the community already maintains gfx906 forks.

Recommendation:

1. **Branch `gfx906-perf` off the local checkout** (it already carries
   the solve_tri GCN patches — commit those first, they're currently
   uncommitted local modifications).
2. Land P1, P2, P5 there; benchmark each against the
   `tests/golden/llama_bench` harness (same prompts/settings as the
   README tables, so reinstinct / stock-llama.cpp / patched-llama.cpp
   are three columns of one table).
3. **Upstream P2 and P5 as mainline PRs** — small, arch-gated, no
   maintenance burden argument against them.
4. P3/P4 stay branch-only unless the maintainers signal interest.

## Validation checklist (every item)

- `test-backend-ops -o MUL_MAT` (and FA ops for P4) on gfx906.
- Perplexity run on a K-quant model before/after — repack must be
  bit-exact in dequantized values, so PPL must be identical.
- `llama-bench` decode+prefill vs the golden harness numbers; report
  tok/s (never ms/tok).
- A long-context decode (≥8k) to catch split-K / KV interactions.
