# reinstinct — architectural overview

A custom HIP inference engine that brings AMD Instinct MI50/MI60 (gfx906,
Vega 20) back from datacenter abandonware to single-card LLM inference.
This document explains how the code is structured and why each
non-obvious choice was forced by the hardware. It is meant to be read
once, end-to-end, by someone trying to understand the engine before
modifying it — or by someone porting the lessons to a different
architecture and trying to figure out what's gfx906-specific.

The companion `gfx906-inference-engine-design.md` is the *original*
design document written before implementation. This one describes what
actually shipped.

---

## 1. What "gfx906" means and why it forces specific choices

The MI50 and MI60 are 2018 AMD datacenter cards built on the Vega 20
die. AMD officially declared the architecture EOL in 2023 and stopped
optimizing inference stacks for it. The hardware itself remains
genuinely capable, but inherits a particular set of constraints that
modern kernels (CDNA / RDNA) sidestep.

The constraints that actually drove engineering decisions in this
codebase:

| What gfx906 has | What it doesn't have |
|---|---|
| 60-64 CUs, 1 TB/s HBM2, 32 GB | No MFMA / matrix cores / tensor cores |
| `v_dot4_i32_i8` (dp4a — 4 int8 muls per cycle per lane) | No bf16 native ops |
| `v_pk_fma_f16` (2 fp16 FMAs per cycle per lane) | No FP8 |
| 64-lane wavefronts (Wave64 only) | No async copy from gmem to LDS |
| 256 VGPRs per SIMD, 4 SIMDs per CU | No 64-lane shuffles (DPP tops out at 16-lane) |
| 64 KB LDS per CU | No `s_memrealtime` for fine-grained timing in kernels |
| DPP cross-lane perm ops (`row_ror`, `row_shl`, `quad_perm`) | |
| `ds_swizzle_b32` for in-LDS lane permutations | |
| `v_perm_b32` for 4-byte gather across lanes | |

Five practical consequences shape every kernel:

**1. dp4a is the heavy lifter.** Every matmul-shaped operation in the
engine — weight matvec, KV-attention dot product, MoE expert GEMM,
spec-decode verify GEMM — uses `__builtin_amdgcn_sdot4` (which lowers to
`v_dot4_i32_i8`). One instruction = 4 int8 multiplies + 1 int32
accumulate, per lane, per cycle. 64-lane wave64 = 256 multiplies per
clock per wavefront. The alternative paths (fp16 VALU on `v_pk_fma_f16`,
fp32 VALU on `v_mul_f32`) are 4× and 8× slower respectively for the
same compute work.

**2. Wave64 is the default and the only choice.** No `width=32`
opt-out. All cross-lane reductions assume 64 active lanes. This makes
softmax, max-reduce, sum-reduce easier (the wave naturally aligns with
64-byte HBM cache lines), but means we have to be deliberate about
under-utilizing wavefronts on small operations.

**3. Cross-lane shuffles are LDS-touching.** `__shfl_xor(x, off)` on
gfx906 compiles to an LDS roundtrip plus barriers — slow. DPP
intrinsics (`v_add_f32_dpp`, `quad_perm`, `row_ror`) bypass LDS but
only handle xor-1, xor-2, xor-8, xor-16; xor-32 still needs LDS, and
xor-4 needs two masked row-shifts. The `kernels/gfx906_dpp.h` header
exposes a hand-rolled `wave64_reduce_add_f32` / `wave64_reduce_max_f32`
that does the full 6-step xor-tree using the cheapest primitive at
each width. Every reduction in the engine goes through these helpers.

**4. No MFMA means flash-attention is VALU-bound.** The popular fp16
FlashAttention implementations on CDNA/RDNA lean on MFMA for the QKᵀ
and PV matmuls. On gfx906 those matmuls would compile to packed-fp16
VALU sequences — 30-40× slower than MFMA. Our attention path quantizes
Q to int8 per head per call and uses dp4a for Q·Kᵀ instead, then a
scalar fp32 path for the PV accumulation. We accept lower precision on
KV (int8 + per-(slot, head) scale, ~48 dB SNR) for the dp4a-friendly
compute path.

**5. Launch latency dominates batch=1.** At single-stream decode,
roughly half of wall-clock time is HIP kernel dispatch overhead, not
compute or bandwidth. The mitigation is HIP graph capture: serialize
the entire decode forward into a single HIP graph, replay-launch with
one host→device submission. Without this, the engine runs at maybe
35% of its post-capture decode tok/s.

These five facts will keep coming up.

---

## 2. Code organization

Top-level layout:

```
src/
  hip/         — dlopen libamdhip64.so, raw FFI, safe wrappers
  gguf/        — zero-copy GGUF reader (memmap2)
  quant/       — block formats: q4_k, q5_k, q6_k, q8_0, iq4_xs, turbo3
  model/       — per-architecture loaders: gemma4, qwen3_5, gemma4_assistant
  cpu/         — fp32 oracle implementations (golden-test reference only)
  runtime/     — GPU runtime
    kernels.rs   — generic GPU ops + test harness
    qwen35.rs    — Qwen 3.5 / 3.6 runtime (incl. MoE + GDN)
    gemma4.rs    — Gemma 4 runtime (incl. MoE + sliding-window + drafter)
    prefill.rs   — shared MMQ GEMM tile dispatch
    spec_decode.rs — Gemma 4 MTP draft+verify+rollback loop
    kv_superquant.rs — opt-in 2-tier KV (capacity feature)
  serve/       — OpenAI-compatible HTTP server
  sampling.rs  — temp/top-k/top-p/min-p/penalties/mirostat/logprobs
  chat.rs      — Gemma 4 chat template (turn markers, role formatting)
  tokenizer.rs — BPE tokenizer (Gemma 4 + Qwen variants)
  main.rs      — CLI dispatcher (12 subcommands)

kernels/       — 121 HIP C++ source files, embedded at build time via
                 include_str!, compiled at first use to .hsaco blobs

docs/          — design + feature docs (this file, SUPERQUANT, etc.)
MANUAL.md      — user-facing manpage (CLI, env vars, models, perf)
README.md      — public summary + benchmark tables
```

Rough size: 16 kLOC of Rust runtime, 121 HIP C++ kernel files (some
~2 KB, the SuperQuant attention variants up to 14 KB).

### How the layers talk

```
                    GGUF file
                         │  (mmap)
                         ▼
                   model/qwen3_5
                   model/gemma4         ← parse metadata, locate tensors
                         │
                         ▼
                runtime/{qwen35,gemma4}  ← upload weights, alloc state
                         │
                         ▼
                  runtime/kernels        ← compile HIP source → .hsaco
                  runtime/mod            ← KernelCache (~/.cache/reinstinct)
                         │
                         ▼
                     hip/sys             ← FFI bindings (libloading)
                     hip/mod             ← safe wrappers (DeviceBuf, Stream, Graph)
                         │
                         ▼
                  libamdhip64.so         ← dlopened at startup
```

The dependency direction is strict: `hip/` knows nothing about LLMs;
`runtime/` knows nothing about file formats; `model/` knows nothing
about GPU execution; `serve/` knows nothing about kernels.

### No ROCm link-time dependency

`hip/sys.rs` does runtime `dlopen` of `libamdhip64.so.7` (`.6` and `.5`
also tried in fallback order). All HIP calls go through `libloading::Library`
symbol lookups. The binary itself depends only on system glibc and Rust's
runtime. This was a deliberate choice: ROCm versioning is a moving
target, and a build that links rocBLAS against ROCm 6.4 won't run on a
machine with ROCm 7.2 installed. The dlopen route makes one binary work
across ROCm 5.7 / 6.x / 7.x.

`rocBLAS` is also dlopened lazily — only the prefill HGEMM path
imports it.

### Kernel build pipeline

Kernel sources are `&'static str` constants embedded in the binary via
`include_str!`:

```rust
// kernels.rs
const MATVEC_Q4_K_DP4A_SRC: &str = include_str!("../../kernels/matvec_q4_k_dp4a.cpp");
```

On first use, `KernelCache::compile` (runtime/mod.rs:159) shells out to
`hipcc --genco --offload-arch=gfx906 -O3 -std=c++17` and writes the
output `.hsaco` blob under `~/.cache/reinstinct/kernels/{xxh3}.hsaco`.
The cache key hashes the source text, target arch, compile flags, and
`hipcc --version` output — any change in any of those forces a
recompile, but unchanged sources skip the ~1-2 s `hipcc` invocation.

Subsequent runs of the engine load the pre-compiled `.hsaco` via
`Module::load`. Cold-start (no cache) takes about a minute to compile
all 121 kernels; warm-start is <1 s.

The `REINSTINCT_OFFLOAD_ARCH` env var overrides the target arch (e.g.,
for a friend who wants to try it on a Radeon VII at `gfx906` or hack
it for `gfx908` / CDNA1). Cache keys include the arch, so multiple
targets coexist in the cache.

---

## 3. The decode forward pass — batch=1 latency-critical

A single decode step processes one input token, runs it through every
transformer block (full-attention or GDN, dense or MoE FFN), and
produces the vocab-length logits for the next token. Wall-clock budget
on the dominant model (Gemma 4 31B Dense Q4_K_XL): **~36 ms per token**
(27.5 tok/s).

### What runs each step

For Gemma 4 31B (60 layers, 10 full-attention + 50 sliding-window, all
dense FFN):

```
embed token                            ← 1 lookup
for each layer (60×):
  attn_norm + add_residual             ← fused
  Q/K/V projections                    ← one quantize_q8, three matvec_xq8
  split_q_gate, q_norm, q_rope
  k_norm, k_rope
  kv_write_q8                          ← per-(slot,head) int8 quantize-write
  attn_partial_q8 + attn_merge         ← split-K FlashDecoding
  attn_output projection
  ffn_norm + add_residual              ← fused
  ffn_gate, ffn_up, swiglu, ffn_down
  ffn output_scale + add_residual      ← fused
output_norm
output_proj                            ← vocab-wide matvec
```

About 180 kernel launches per step. Without HIP graph capture, each
launch takes ~5-10 µs of CPU→GPU dispatch overhead — at 180 launches
that's ~1.4 ms wasted per step, ~4% of the budget.

### HIP graph capture

The decode forward is captured once per `(state, n_tokens=1)` shape
into a `GraphExec` (runtime/qwen35.rs:3302+, gemma4.rs has similar).
After capture, each subsequent decode step submits one
`hipGraphLaunch` to the queue — host returns immediately, the GPU
executes the full layer chain back-to-back with no inter-kernel
dispatch overhead.

Critical constraint: **HIP graphs cannot contain `hipMalloc`,
`hipFree`, or host→device memcpy that depends on a host-computed
value.** Every scratch buffer is pre-allocated; every position
counter the kernels need (decode position, current K/V cache length)
lives in a device-resident `DeviceBuf<u32>` updated via a tiny
`hipMemcpy` _before_ the graph fires (and that memcpy isn't part of
the captured graph). Look for `d_pos` references in both runtimes.

Anything that breaks graph capture also breaks the perf — there's no
soft fallback. Environment switches `REINSTINCT_NO_GRAPH=1` and
`REINSTINCT_PREFILL_NO_GRAPH=1` exist for debugging.

### The kernel fusion that mattered

Three fusions ship and account for measured wins; others were tried
and reverted (see §13).

1. **`rmsnorm_add` and `rmsnorm_add_scale`** — folds the per-layer
   pre-norm with the residual add, eliminating one launch and one
   full-vector roundtrip through HBM per sublayer. Saves ~5% on
   Gemma 4 (long graph chain).

2. **`split_q_gate`** — Qwen 3.5/3.6's per-head Q-norm uses a learned
   per-head gate. The split kernel takes the fused QKV-projection
   output and splits Q from the gate plane in one pass. Eliminates a
   separate dispatch + scratch buffer.

3. **`quantize_once + matvec_xq8(×N)`** — for any place that runs
   matvecs sharing the same input vector (Q/K/V on one normed
   activation; FFN gate + up on one normed activation), the input is
   quantized to int8 once into a shared scratch buffer (`xq8`), then
   re-used across all matvecs. Cuts ~120 redundant quantize launches
   per decode forward on Gemma 26B-MoE.

### Sampling

Standard temp / top-k / top-p / min-p / repetition-penalty /
frequency-penalty / presence-penalty / mirostat-v2 stack lives in
`sampling.rs`. Logprobs (`logprobs: n` in OpenAI requests) are
extracted from the same softmax that produces the sampled token. All
CPU-side; sampling is fast and the input is a single vocab-length
fp32 vector copied D2H per step.

---

## 4. The prefill forward pass — batch=N throughput-critical

Prefill processes the prompt's N tokens in one batched forward. Wall
clock at N=512 on Gemma 4 31B: **~177 tok/s** (vs 21 tok/s on
llama.cpp's gfx906 build).

### Architectural difference vs decode

Prefill is **weight-bandwidth-bound**, not launch-bound. At N=512 a
typical FFN matvec needs to compute 512 output rows × 5376 hidden ×
21504 ffn = 59 GFLOPs against ~110 MB of weight bytes — at peak HBM
that's a 110 µs floor. Each kernel launch is ~10 µs; with 180 layers
× a few kernels each = ~1.8 ms of dispatch on top of ~80 ms of math.
Worth optimizing, but kernel fusion is a small win compared to making
each kernel hit peak bandwidth.

### MMQ tiled GEMM

The dominant prefill kernel is the 2D-tiled int8 MMQ GEMM
(`prefill.rs`, plus `kernels/mmq_gemm_q4k_repacked.cpp` /
`mmq_gemm_q5k_repacked.cpp` / `mmq_gemm_q6k_repacked.cpp` /
`mmq_gemm_q8_0_repacked.cpp`). Per-shape config: `BM=64, BN=64`
(rows × cols per tile), `BK=4` (sub-block depth per tile), occupancy
2 WG/CU, the sX activation tile padded `[BN][BK+1]` to break a
4-way LDS bank conflict on `xq32` reads (-3% to -10% across MoE
variants without the pad).

The dispatch picks MMQ vs the legacy `dequant → HGEMM` path based on
N: above ~32 prefill tokens, MMQ wins (the BN=64 tile fills with
work). Below that, neither MMQ nor MMVQ wins enough to ship — see
§13 on the reverted small-batch MMVQ attempt.

### Pooled per-call buffers + graph capture

Prefill has many transient scratch buffers (per-N activation
matrices, dequant temporaries, MoE token-gather/scatter staging).
Allocating them per call would forbid HIP graph capture (no
`hipMalloc` allowed mid-graph). The runtime uses `DeviceBufPool`
(`pool_f32`, `pool_u8`, `pool_u16` in qwen35.rs) — first prefill at
a new N runs uncaptured and warms the pool; subsequent prefills at
that same N capture into a per-N `GraphExec`. The `prefill_warm_p`
HashSet tracks which N counts are warmed.

End-result: a serve-mode workload that sees the same prompt sizes
repeatedly (e.g., bulk batch inference over short prompts) pays
the per-N graph-capture cost once, then runs at full HBM bandwidth
on every subsequent call at that N.

### MoE grouped-expert GEMM

The naive MoE forward runs one matvec per (token, selected-expert)
pair — for top-k=4 routing with 60 experts active that's 4 matvecs
per token, each touching the full expert weight matrix once. At N=128
prefill tokens × 4 experts = 512 matvecs.

The grouped path (`REINSTINCT_MOE_GROUPED=1`, default-on):

1. Counting-sort the (token, expert) pairs by expert.
2. Scatter activations into expert-contiguous order.
3. One tiled MMQ GEMM per expert (BN=16 for qwen MoE's Q4_K/Q5_K
   experts, BN=32 for Gemma 26B-MoE's Q6_K/Q8_0 experts — sized so
   the "tokens per expert" average exactly fills the tile, no
   padding waste).
4. Scatter results back to original token order.

Measured: Qwen MoE prefill ~2.1×, Gemma MoE prefill ~2.05× vs the
per-token path. See `project_moe_grouped_gemm` memory.

---

## 5. Quantization & weight repack

The engine supports six on-disk quant types from GGUF:

| Type | bpv | Used for |
|---|---:|---|
| **Q4_K** | 4.5 | dense weights (UD-Q4_K_XL primary) |
| **Q5_K** | 5.5 | dense weights, embedding tables |
| **Q6_K** | 6.5625 | dense weights (UD-Q6_K_XL primary), embedding tables |
| **Q8_0** | 8.5 | shared experts in MoE, MTP head, sometimes critical-precision tensors |
| **IQ4_XS** | ~4.25 | Unsloth-dynamic mix tensors (~4% of UD-Q4_K_XL files) |
| **F16/BF16** | 16 | router gates, norms, small precision-critical layers |

### Why repacking matters

The on-disk K-quant superblock layout (Q4_K is 144 bytes encoding 256
weights: 16 fp16 dmin/d scales, 12-byte 6-bit scale plane, 128-byte
nibble plane, interleaved) is great for compact storage but terrible
for streaming matvec — each thread has to gather scales from one part
of the superblock and quants from another, with non-coalesced
strided reads. Measured: dp4a Q4_K matvec on the on-disk format
sustains ~470 GB/s, vs the 835 GB/s the same-shape clean streaming
kernel can hit.

### The v2 repacked format

At load time, every K-quant matvec weight is repacked once into three
contiguous planes (`src/quant/q4_k.rs::repack_for_matvec`):

```
plane A: [out_dim × n_sub × 16]   nibble plane, sub-block contiguous
plane B: [out_dim × n_sub × 2]    6-bit sc + 6-bit m as two u8/sub
plane C: [out_dim × n_super × 4]  fp16 d + fp16 dmin per 256-weight superblock
```

A lane reads sub-block `lane_id` as one `uint4` (16 bytes contiguous);
consecutive lanes hit consecutive memory; the read is a fully-coalesced
sweep. `dsc = d·sc` and `deff = dmin·m` get computed on the fly. The
v2 (current) format stores the scale plane at native 2 B/sub density
rather than the v1 pre-multiplied fp16 (4 B/sub) — saved ~10% HBM
traffic on the dominant matvec.

Result on a real shape (21504 × 5376, Gemma FFN gate): **740 GB/s** —
89% of the streaming ceiling. See `project_q4k_matvec_wall` memory for
the full history (block=256 → repack v1 → repack v2 → ROWS dispatch).

**Anti-aliasing pad.** The repacked row stride is padded by 1 when the
natural sub-block count is a power of two, otherwise all rows alias to
one HBM channel and bandwidth collapses 3×. See `repacked_n_sub_padded`
in `q4_k.rs`.

### Q8_0 — adaptive ROWS dispatch

Q8_0 on-disk is already a 34-byte flat block (no superblock header,
no interleaving). Repack just rearranges into separate qs and d
planes for cleaner streaming.

Two kernel variants ship and the dispatcher picks per-call:
- `matvec_q8_0_repacked_f32` (ROWS=2, grid=out/2): best at mid out_dim
- `matvec_q8_0_repacked_r1_f32` (ROWS=1, grid=out): wins for
  `out_dim ≥ 4096` — doubles wavefront count, sustains HBM bandwidth
  that ROWS=2 starves at large out_dim

Measured: +13% on qwen 35B-MoE decode. See `matvec_q8_0_repacked.cpp`
header for the analysis.

### What's *not* repacked

- **Q8_0** is already flat-friendly on-disk; tried repacking and it
  added a second stream that cost ~1% (`project_q4k_matvec_wall`).
- **IQ4_XS** stays on-disk but gets the dp4a treatment. Was 12% of
  decode on Qwen 27B in fp32 wave64 fallback; now ~4%.
- **token_embd** (the embedding lookup table) is also the LM-head
  output in tied-embedding models. Reading it from-disk is fine
  because embedding lookup is one row per call, not a sweep.

---

## 6. Attention

### Decode attention: split-K FlashDecoding

The decode case is batch=1, sequence position increasing by 1 each
step. The naive "one block per Q head, walk the KV cache from 0 to
total_len, accumulate" kernel leaves CUs idle at depth (with 32
heads on Gemma 31B, only 32 blocks dispatched — barely fills 60 CUs)
and runs the PV accumulation serially over the whole context.

FlashDecoding-style split-K (`kernels/attn_partial_q8.cpp`) splits the
KV range into N partitions, dispatches grid `(n_heads, n_splits)`,
each block runs a stable softmax over its slice and writes
`(m_partial, l_partial, o_partial)`. A separate `attn_merge` kernel
combines the partial outputs via the standard FlashAttention merge
formula. Restores occupancy and shortens the serial PV scan ~N×.

Number of splits is `clamp(ceil(max_seq/256), 1, 16)` — chosen to keep
the grid > number of CUs without over-saturating the wave-per-CU
budget.

### int8 KV cache (Gemma) — dp4a-friendly

Gemma 4 uses int8 KV with per-(slot, head) fp32 scale:
- Storage: K, V as `[max_seq, n_kv, head_dim]` int8 + scales as
  `[max_seq, n_kv]` fp32 (per K and per V separately)
- Write path: `kv_write_q8_f32` — 256-thread cooperative amax + quantize
  per (slot, head)
- Read in attention: per-Q-row Q-quantize to int8 (per-head scale),
  then `__builtin_amdgcn_sdot4` for the Q·Kᵀ accumulation, scalar fp32
  for PV (V dequant on the fly)

Measured per-call cost: **19.2 µs** at head_dim=256 average on Gemma 31B
(vs the fp32 KV path's 41.7 µs at the same head_dim — 2× from
bandwidth alone). Quality: 48 dB SNR, output indistinguishable from
fp16 KV.

### fp32 KV cache (Qwen) — kept on purpose

Qwen 3.5/3.6 full-attention layers use fp32 KV. We tried porting them
to the Gemma-style int8 KV path (May 2026) and **reverted it** —
the per-call attention speedup was real (~2×) but Qwen is GDN-heavy
(1 full-attention layer per 4 in the 27B variant) so attention is only
~1.7% of decode time. Halving 1.7% gets ~0.85%, which the per-token
quantize-on-write overhead exactly cancelled. Measured zero wall-clock
benefit at any context length.

Documented in `project_qwen_int8_kv_attempt` memory with do-not-re-attempt
criteria.

### Sliding-window attention (Gemma 4)

Gemma 4 has a 5:1 SWA:full pattern (the 31B has 50 SWA + 10 full
attention layers in its 60-layer stack). SWA layers attend over the
trailing `window` tokens of the cache; the same `attn_partial_q8`
kernel takes a `window: u32` parameter — when nonzero, the kernel
limits the iterated range. No separate SWA kernel.

Gemma 4 also uses **KV-sharing** across consecutive SWA layers — every
group of layers shares one physical KV cache buffer. Reduces VRAM
significantly on E4B and is wired in `gemma4.rs` via the
`GpuBlockState::Full(kv)` vs `Shared(layer_idx)` enum variants.

### Prefill attention — batched FlashAttention

`attn_prefill_flash_f32` (Qwen) / Gemma's prefill attention path use
per-Q-row workgroups walking the full causal-mask range. Same
split-by-row structure as decode but batched over P prompt tokens
in one grid dimension.

### What's *not* there

- **No FlashAttention-2 with MFMA.** gfx906 has no MFMA. We use
  dp4a-friendly attention instead. The fp16 VALU FlashAttention path
  that ships in the `ai-infos/flash-attention-gfx906` Triton fork is
  meaningfully slower per the published autotune tile (`BLOCK_N=16,
  waves_per_eu=1` — register-heavy small-tile bookkeeping).
- **No batched decode (multi-sequence concurrent forward).** This is
  a single-stream engine. Adding a batched decode path is the one
  serving-style optimization remaining; it's not on the kernel list
  but flagged as the single architectural choice that could give a
  5×+ aggregate-throughput win for serve workloads.

---

## 7. MoE — grouped-expert GEMM + the per-token decode path

Gemma 26B-A4B and Qwen 3.5/3.6 35B-A3B are MoE models. Top-k routing
(2 or 4 experts per token), per-token sigmoid-gated shared expert,
60-128 routed experts per layer.

### Decode (batch=1) path

One token, one (or two, or four) selected experts per layer.
Per-(token, expert) matvecs. Wins from kernel fusion are big here
because there are many small launches per layer.

`step_moe_ffn` in qwen35.rs:2292 / gemma4.rs equivalent:
1. Router matvec → per-expert scores
2. Top-k selection (CPU-side or small GPU kernel depending on K)
3. For each selected expert: `moe_matvec_*_repacked_f32` (gate + up + down)
4. Shared expert path (separate matvec, sigmoid gate)
5. Weighted sum into output

The MoE matvec kernels are the same repacked-K-quant code path as
dense matvec; the only difference is that they take a per-token expert
index for picking which row block to use.

### Prefill grouped GEMM (covered in §4)

Token-by-expert sort, scatter activations, one tiled GEMM per expert,
scatter back. Cuts MoE prefill ~2×.

---

## 8. GDN — Qwen 3.5/3.6 hybrid recurrent layers

Qwen 3.5 and 3.6 have a non-standard layer pattern: most layers are
not full attention but **Gated DeltaNet** (GDN) — a linear-attention
variant that maintains a per-head recurrent state matrix
`S[head][value_dim][key_dim]` updated by gated outer-product rules.

The architecture interleave is `L, L, L, F` (3 GDN + 1 full attention,
repeating) for Qwen 27B with `full_attention_interval=4` from the GGUF
metadata. The 27B has 64 layers = 48 GDN + 16 full attention.

### Why GDN matters for the engine

It's the layer kind that *doesn't* fit any standard transformer-block
template. The runtime has to maintain per-block recurrent state across
the entire decode (not just KV cache). Two kernels:

- `gdn_recurrent_step_fused.cpp` — single-token decode update. Per-head
  state is `[head_dim, head_dim]`, lives LDS-resident during the
  kernel (~256 KB at head_dim=256 — barely fits gfx906's 64 KB/CU
  budget, so we use LDS staging tricks).
- `gdn_recurrent_step_fused_batched.cpp` — prefill batched update.

### The optimization that mattered

The GDN recurrent step is sensitive to LDS bank conflicts on the
per-head state slice. Padding the state stride by +1 broke a 64-way
bank conflict — measured: 3.84× speedup on Qwen 27B GDN layers (8.57
ms → 2.23 ms in the layer cost). See `project_qwen35_prefill_kernels`
memory.

The state-shuffle kernel uses `v_perm_b32` for the 4-byte gather
across lanes. gfx906-specific instruction; the kernel won't run as-is
on non-Vega.

### Why this matters for tuning attention

Per the rocprof analysis (§docs/SUPERQUANT.md not applicable; see the
`project_qwen_int8_kv_attempt` memory): GDN dominates qwen decode at
~4.5% of decode time, and attention is only ~1.7%. **The hybrid
architecture caps the upside of any attention-side optimization on
Qwen** — half the decode time is matvecs, the other half is GDN +
norms + RoPE + KV write + softmax. There's no single fat target left.

---

## 9. Speculative decoding (Gemma 4 only)

Gemma 4 ships a "MTP" head (Multi-Token Prediction) as a separate
small drafter network. Reinstinct loads it as `Gemma4Assistant`
(model/gemma4_assistant.rs) — same architecture as the target's first
few layers, sized to draft K=2/3/4 tokens per round.

### The loop

`spec_decode.rs::spec_decode_generate`:

```
loop until n_generated >= max_tokens:
  snapshot state                       ← cheap, just len + GDN-state copy
  drafter.forward_token × K            ← propose K candidate tokens
  target.forward_tokens_verify(K+1)    ← target's prediction for each
                                         drafted token + the post-K token
  for i in 0..K:
    if target_argmax[i] == drafted[i]:
      accept
    else:
      reject, restore state to (pre-verify-pos + i)
      replacement = target_argmax[i]
      break
  if all K accepted: bonus token = target_argmax[K]
```

Per-round structure: one drafter forward × K + one target batched
verify. The verify path uses `attn_prefill_flash_f32` (Gemma-style
batched attention) so a K-token verify is one launch through the
target's full forward, not K sequential calls.

### Measured wins (Gemma 31B Q4_K_XL, K=3)

| Prompt class | Plain tok/s | MTP tok/s | accept rate | Δ |
|---|---:|---:|---:|---:|
| Factual ("Capital of France?") | 27.5 | 32.8 | 89% | +19% |
| Structured ("List 5 primes") | 27.5 | 31.8 | 85% | +16% |
| Procedural ("How to make tea") | 27.5 | 25.7 | 63% | −7% |
| Creative ("Write a haiku") | 27.5 | 23.6 | 55% | −14% |

MTP is per-request opt-in via the serve endpoint (`use_speculative: true`
in OpenAI request body) — default-on when a drafter is loaded.

### Qwen MTP — shelved

Qwen 3.6 ships an in-GGUF MTP head. Wired fully (`qwen-mtp-gen`
CLI command) but measured 0.58× best vs plain decode on MI50. The
GDN-heavy architecture's `K+1`-deep sequential recurrence per verify
round costs more than the speculation saves. See `project_qwen_mtp`
memory.

---

## 10. Tiered KV — SuperQuant (opt-in capacity feature)

When the int8 KV cache won't fit (`max_seq` × 16 layers × 4 KV
heads × 256 head_dim × ~1 byte = real numbers on long context),
SuperQuant offers a 2-tier cache:

| Tier | Format | bpv | SNR | Holds |
|---|---|---:|---:|---|
| Warm | int8 + per-(slot, head) f32 scale | ~8 | ~48 dB | recent context |
| Cold | turbo3 (RHT + Lloyd-Max 3-bit codebook) | 3.5 | ~14.6 dB | older context |

Writes go to Warm; when Warm fills, the oldest entry slides to Cold
via the `kv_promote_q8_to_turbo3` GPU kernel.

**This is a VRAM/capacity feature, not a perf feature.** Decode is
29–35% slower than int8 on every realistic config. Documented in
detail at `docs/SUPERQUANT.md`.

Enabled per-process by `REINSTINCT_KV_SUPERQUANT=1`, Gemma 4 only
today.

---

## 11. Runtime infrastructure

### KernelCache (`runtime/mod.rs`)

Single-file responsibility: turn HIP C++ source strings into `.hsaco`
blobs. Filesystem-backed cache under `~/.cache/reinstinct/kernels/`.
Cache key: `xxh3(source + arch + flags + hipcc_version)`. Compiled
modules are cloneable (Clone derive) so multiple runtime types
(`GpuQwen35`, `GpuGemma4`) can share one cache without re-compiling.

### DeviceBuf + Stream + Graph (`hip/mod.rs`)

Safe RAII wrappers around HIP runtime calls. `DeviceBuf<T>` is the
strongly-typed device allocation. `Stream` is the work queue. `Graph`
and `GraphExec` are the capture/replay primitives. Everything goes
through one default Stream per `GpuQwen35` / `GpuGemma4` instance —
no kernel-level parallelism within a model's forward pass (we have
plenty of intra-kernel parallelism; multiple streams just add
scheduling complexity for negligible benefit on single-batch decode).

### Per-call buffer pools (`DeviceBufPool`)

Some kernel chains (especially MoE prefill, the verify path) need
many transient buffers. Allocating them via `hipMalloc` per call
breaks HIP graph capture. The pool owns a stack of typed `DeviceBuf`s
keyed by element count — `take(n)` returns an existing buffer if one
exists at that size or allocates and returns it; the buffer goes back
to the pool when the owning Rust scope ends. First call at a new
size allocates; subsequent calls at the same size are zero-allocation.

This is what makes "first prefill at N=512 captures the graph,
subsequent prefills at N=512 replay it" work.

---

## 12. Serve mode

`src/serve/mod.rs` — OpenAI-compatible HTTP server. Three ports:
big LLM (~30B dense), small LLM (~4-8B for fallback), embedder
(slot reserved, 503 today).

```
client → TCP listener (per port) → accept thread → mpsc::Sender<Job>
                                                       │
                                          one worker thread owns GPU
                                                       │
                                               GpuQwen35 / GpuGemma4
```

Single GPU = single worker thread. Per-request panic is caught with
`catch_unwind` and reported as HTTP 500 + logged with request_id.
Worker doesn't crash on a single bad request.

Features: SSE streaming, logprobs, prefix KV cache (LRU N-slot per
model, restored via `state.restore(snapshot)` + `state.truncate(common_prefix_len)`),
per-request spec-decode opt-in, sampling parameter pass-through.

**No auth, no TLS, no graceful SIGTERM**, no backpressure beyond the
60s socket timeout. Documented in MANUAL.md as "deploy behind a
reverse proxy." Same trade-off as most "embedded LLM server"
implementations.

---

## 13. Things we tried that didn't work

Documented in memory so we don't re-attempt:

- **LDS-staged matvec** (single + double buffered) — barrier-serialized
  the load/compute; HBM→reg→LDS→reg roundtrip cost more than the
  contiguous-read saving. Repacking the on-disk weight layout is what
  actually mattered.
- **32-rows-per-wavefront scatter, 4-lane chunk mapping** — both
  regressed 30-50% on the matvec.
- **Small-batch prefill MMVQ** — proposed an MMVQ kernel for `pp32`
  (small batches). Re-reading the weight P× through L2 cost more than
  MMQ's BN=64 tile-waste; regressed pp32 2×. The real fix would be
  split-K MMQ.
- **Kernel fusion of quantize-into-rmsnorm** — looked like an 11% win
  per a noop-kernel microbench but measured 0% on the real workload.
  HIP graphs pipeline each kernel's *dispatch* behind the previous
  kernel's *execution*; the per-dispatch cost is hidden for real
  (non-noop) kernels. Lesson: don't trust noop-dispatch microbenchmarks.
- **`__builtin_nontemporal_load` on weight streams** — regressed −2.7
  to −12.4% across all benchmarked models. gfx906's HW L2 prefetcher
  does real work on our sequential per-lane access pattern; `slc:1`
  (nontemporal) disables it.
- **Qwen full-attention KV port to int8** — built end-to-end, measured
  zero wall-clock improvement (qwen's GDN-heavy architecture caps
  attention's share of decode at ~1.7%). Reverted in single session.
- **Q4_K ROWS=1/4 matvec variants** — built, benched across 13 real
  shapes; +7% on the smallest (kv-projection-sized) matvec but
  invisible wall-clock. Kernels kept as research artifacts; dispatcher
  unchanged.
- **Per-head-dim FlashAttention specialization** (the upstream
  llama.cpp PR #22880 approach) — confirmed our workload doesn't have
  the variation that specialization targets.

The pattern that recurs: kernel-level wins on specific shapes don't
propagate to wall-clock when the shape is a small fraction of total
decode time. **Always multiply through by fraction-of-decode before
investing in a port.**

---

## 14. Testing strategy

Three tiers:

1. **`cargo test --lib`** — 180+ unit tests in `src/runtime/kernels.rs::tests`
   and per-runtime test modules. GPU-required, must be run with
   `--test-threads=1` to serialize the HSA signal pool. Test categories:
   - Per-kernel oracle tests (random inputs, GPU vs CPU fp32 reference,
     `rel_l2` tolerance bound per quant type)
   - Shape sweeps (matvec across 10+ shapes, attention across 4-5
     total-lens)
   - Bit-identical tests where the GPU kernel claims to match the CPU
     encode (turbo3 quantize round-trip, weight repack)

2. **Integration tests** (`tests/*.rs`) — load a real GGUF fixture,
   exercise a layer or end-to-end forward against the CPU oracle.
   The oracle is `src/cpu/`'s fp32 reference. Requires
   `REINSTINCT_GGUF_FIXTURE=path/to/qwen-3.5-0.8B-Q4_K_XL.gguf`
   (or the default `~/models/qwen-3.5-0.8B/...` location).

3. **Golden tests** (`tests/qwen35_golden.rs`,
   `tests/golden/`) — fixed prompt + temperature=0 + N tokens →
   asserted-equal output token IDs. Catches any regression that
   silently shifts logits enough to change argmax.

**Tolerances are quantization-aware.** Tests use `set_dp4a(false)`
when comparing against the fp32 oracle to avoid the systematic
int8 quantization noise drifting the assert; for tests that compare
the production dp4a path against the oracle, tolerances are wider
(per `feedback_gpu_oracle_tests` memory: `5e-3 rel_l2` typical).

**Two known float-divergence tests** are currently skipped:
`full_attention_block_matches_cpu_for_real_block` and
`full_attention_step_matches_cpu_for_real_block`. Both detect a real
~25% divergence at step 1 of dense full-attention layers on Qwen.
Hasn't been root-caused; pre-dates current work.

---

## 15. Where to look next

Open avenues for further work (none are needed for the current shipping
quality, but mapped here for the next maintainer):

1. **Batched decode for serving workloads** — process N concurrent
   sequences per forward, share weight reads across them. Big
   aggregate-throughput win (2-5× vs single-stream). Requires KV-cache-
   per-sequence, per-sequence position vectors, per-sequence sampling.
   Architectural lift, not a kernel; estimated weeks.

2. **`tracing` migration** — ~92 ad-hoc `eprintln!` sites would benefit
   from structured logging (log levels, JSON output for shippers,
   request_id correlation). Deferred from the pre-release pass per
   `project_tracing_followup` memory.

3. **Qwen full-attention divergence root-cause** — the two skipped
   tests above. Not a regression but real; understanding it might
   yield a small quality or perf win.

4. **Split-K MMQ GEMM** — would let us beat the current MMQ at small
   batch sizes (pp32 etc.) without the L2-broadcast trap that killed
   the MMVQ attempt. Untried.

5. **MTP for Gemma 4 26B-A4B MoE** — currently regressed because the
   verify path falls back to K sequential `forward_token` calls (a
   batched-MoE verify kernel was prototyped and is ~3.5× slower on
   MI50 due to per-(token, slot) expert weight reads). Could be
   revisited with a different verify design.

Things explicitly *not* to revisit (each with measured rejection in
memory):

- LDS-staged matvec, nontemporal loads, per-head-dim FA tuning,
  small-batch prefill MMVQ, kernel fusion of quantize, Qwen int8 KV
  port (without a long-context attention-share motivation), Q4_K
  ROWS-variant dispatch.

---

## Reading order for the curious

If you're working through the code for the first time, the path that
yields the most context per minute:

1. This document (you're here)
2. `MANUAL.md` — user-facing CLI, env vars, model list
3. `kernels/gfx906_dpp.h` — the gfx906-specific primitives
4. `kernels/matvec_q4k_repacked.cpp` — the dominant decode kernel
5. `kernels/attn_partial_q8.cpp` — the dominant attention kernel
6. `src/runtime/gemma4.rs::forward_token` — end-to-end decode for the
   simpler architecture (no GDN)
7. `src/runtime/qwen35.rs::forward_token` — same but with GDN layers
8. `src/runtime/prefill.rs` — shared MMQ tile dispatch
9. `docs/SUPERQUANT.md` — the one opt-in feature with non-trivial design
10. The `memory/` directory — every shipped feature has a project
    memory; every abandoned attempt has a do-not-re-attempt memory.

The single best file for understanding the engine's center of gravity
is `kernels/attn_partial_q8.cpp` — ~150 lines that combine dp4a,
wave64 DPP reductions, split-K dispatch, per-head Q quantization, and
the int8 KV format in one place. Read it twice.
