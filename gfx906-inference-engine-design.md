# Custom HIP Inference Engine for AMD MI50/MI60 (gfx906)
## Design and Feasibility Reference

**Target hardware:** AMD Instinct MI50 32GB / MI60 32GB (gfx906, Vega 20, 7nm)
**Target models:** Gemma 4 family, Qwen 3.x family (dense + MoE)
**Target quants:** Unsloth Dynamic GGUF — UD-Q4_K_XL and UD-Q6_K_XL exclusively
**Scope:** Single-card inference with speculative decoding. Raw token-in/logits-out — no chat templates, no tokenizer.

---

## 1. Hardware Platform

### gfx906 Specifications

| Parameter | MI50 32GB | MI60 32GB |
|---|---|---|
| Architecture | GCN 5.1 (Vega 20) | GCN 5.1 (Vega 20) |
| Process | 7nm | 7nm |
| Compute Units | 60 | 64 |
| Stream Processors | 3840 | 4096 |
| Wavefront | **Wave64 only** | **Wave64 only** |
| FP16 peak | 26.5 TFLOPS | 28.3 TFLOPS |
| FP32 peak | 13.3 TFLOPS | 14.1 TFLOPS |
| FP64 peak | 6.6 TFLOPS (½ rate) | 7.1 TFLOPS (½ rate, full rate) |
| BF16 | **Not supported** | **Not supported** |
| Matrix cores / Tensor cores | **None** | **None** |
| HBM2 | 32 GB, 4096-bit bus | 32 GB, 4096-bit bus |
| HBM2 bandwidth | ~1 TB/s | ~1 TB/s |
| L2 cache | 4 MB | 4 MB |
| LDS per CU | **64 KB** | **64 KB** |
| VGPRs per SIMD | 256 (4 SIMDs/CU) | 256 (4 SIMDs/CU) |
| Max wavefronts/SIMD | 10 (at ≤24 VGPRs), 8 (at 32), 5 (at 48) | Same |
| PCIe | Gen 4 x16 | Gen 4 x16 |
| Infinity Fabric Link | Dual (248 GB/s aggregate) | Dual |

MI60 is the same die with all 64 CUs enabled and full-rate FP64. For this project the ISA is identical, kernel binaries are identical, and both report as `gfx906:sramecc+:xnack-`. The runtime sees 4 extra CUs on MI60 — free performance, zero extra work.

### Key Hardware Constraints for LLM Inference

**Launch latency dominates batch-1 decode.** Community benchmarks (gfx906 llama.cpp turbo fork, vllm-gfx906-mobydick) consistently measure ~10% effective HBM2 bandwidth utilization at batch=1. Both converge to a ~56-57 tok/s ceiling on 30B-class MoE models — strongly indicating the ceiling is CPU→GPU dispatch overhead (AQL packet submission), not memory bandwidth or compute. This is the single most important engineering fact: **the MI50 at batch=1 is launch-bound, not bandwidth-bound.**

**No tensor cores / matrix cores.** All FP16 matmul throughput comes from packed VALU instructions (`v_pk_fma_f16` — 2 FLOPs per instruction per lane, 64 lanes per wavefront). Practical HGEMM via rocBLAS reaches ~20-24 TFLOPS on large square matrices (vs 26.5 TFLOPS theoretical peak), degrading significantly on the tall-skinny shapes that dominate LLM inference.

**64 KB LDS is small.** FlashAttention-2 tile sizes must be reduced vs modern CDNA targets. Community kernels use `num_warps=4`, `num_stages=1`, and smaller `BLOCK_M`/`BLOCK_N`. Forcing `num_stages=1` is also a stability requirement on gfx906.

**No BF16.** All BF16 weights in GGUF files must be cast to FP16 at load time. FP16 attention math has overflow risk on long sequences — keep softmax accumulation in FP32 even at the cost of registers.

**DPP cross-lane operations** are the fastest reduction primitive on GCN5. Softmax reductions, Q8 quantize, and warp-cooperative dot products should use `ds_swizzle`/`v_permlane`/DPP rather than LDS-based reductions.

### ROCm Support Status

AMD entered gfx906 into maintenance mode with ROCm 5.7 (Q3 2023). End-of-maintenance was Q2 2024. **The last officially fully-supported ROCm release is 5.7.**

Starting with **ROCm 6.4.0, AMD stopped shipping pre-compiled Tensile library files for gfx906** in the official packages. This breaks rocBLAS GEMM operations unless you build from source or copy libraries from Arch Linux (see Section 6).

The community continues to build and run ROCm 7.x on gfx906 successfully. The recommended path:

- **ROCm 7.1+ runtime** (install via amdgpu-install, target gfx906)
- **rocBLAS built from source** with `-DGPU_TARGETS=gfx906:xnack-`
- **HIP graphs** require ROCm 7.1+ on gfx906 for stability
- **MIOpen, rocSPARSE, AITER** do not officially support gfx906 — bypass or carry patches
- **hipBLASLt** is primarily gfx90a+ / CDNA2+ — do not depend on it for gfx906

This situation strongly argues for the hipfire-style approach: **own your kernels, depend only on `libamdhip64.so` at runtime, and treat rocBLAS as an optional accelerator for prefill GEMM.**

---

## 2. Target Models and Memory Budget

### Model Fit Analysis (32 GB HBM2)

| Model | Type | Params (total / active) | Q4 weight size | Q6 weight size | Fits 32GB? | Notes |
|---|---|---|---|---|---|---|
| Qwen3-8B | Dense | 8B / 8B | ~4.5 GB | ~6.5 GB | **Yes** — ample room | Primary dense target |
| Qwen3-30B-A3B | MoE | 30B / 3.3B | ~17 GB | ~24.5 GB | **Yes** at Q4, tight at Q6 | Primary MoE target |
| Gemma 4 E2B | Dense | ~2B / 2B | ~1.2 GB | ~1.7 GB | **Yes** | Drafter candidate |
| Gemma 4 E4B | Dense | ~4B / 4B | ~2.3 GB | ~3.3 GB | **Yes** | Drafter candidate |
| Gemma 4 26B-A4B | MoE | 26B / ~4B | ~14.5 GB | ~21 GB | **Yes** | Second MoE target |
| Gemma 4 31B | Dense | 31B / 31B | ~17 GB | ~25 GB | **Yes** at Q4, tight at Q6 | Large dense target |
| Qwen3.5-122B-A10B | Hybrid MoE | 122B / 10B | ~62 GB | ~90 GB | **No** | Out of scope — does not fit |

### Memory Budget Breakdown (Example: Qwen3-30B-A3B at Q4_K_XL)

| Component | Size | Notes |
|---|---|---|
| Model weights (Q4 dominant) | ~17 GB | Loaded once at startup |
| KV cache (Q8_0, 8k context) | ~1-2 GB | Scales with context length |
| Draft model (Qwen3-0.6B Q8) | ~0.7 GB | For speculative decoding |
| Draft model KV cache | ~0.1 GB | Small |
| Prefill scratch buffer | ~0.5-1 GB | FP16 dequant tile for rocBLAS |
| Runtime overhead | ~0.5 GB | HIP allocator, graph capture, misc |
| **Total** | **~20-21 GB** | **~11 GB headroom** |

At Q6_K_XL the weight budget increases to ~24.5 GB, leaving ~5 GB headroom — still workable but KV cache must be quantized for longer contexts.

### Architecture-Specific Details That Drive Kernel Design

**Gemma 4 (Google, April 2026):**
- QK-Norm on Q and K after projection (RMSNorm fused into rotary path)
- 5:1 sliding-window-to-global attention pattern (50 sliding layers, 10 global); sliding window = 1024
- Global layers use unified K/V and p-RoPE (proportional RoPE)
- Soft-capped final logits
- 26B-A4B MoE: 128 experts per MoE layer, 8 routed + 1 shared expert
- 262k vocab, tied embeddings

**Qwen 3 (April 2025):**
- QK-Norm, standard RoPE, RMSNorm pre-norm, SwiGLU
- No QKV bias (removed from Qwen2)
- Qwen3-8B: 36 layers, hidden 4096, GQA 32Q/8KV, head_dim 128
- Qwen3-30B-A3B: 48 layers, hidden 2048, GQA 32Q/4KV, 128 experts, 8 active, **no shared expert**, expert intermediate 768
- 32k native context, 128k via YaRN

### Required Kernel Variants

The engine needs these distinct kernel paths based on the model architectures:

1. RMSNorm + pre-norm + QKV projection (fused FlashNorm-style)
2. QK-Norm (second normalization on Q and K — both families use it)
3. RoPE (standard for Qwen3; p-RoPE for Gemma 4 global layers) fused with KV cache write
4. Sliding-window attention with Q8 KV cache (Gemma 4)
5. Full attention with unified K/V (Gemma 4 global layers)
6. Standard GQA attention (Qwen3)
7. SwiGLU FFN with fused gate × up activation
8. MoE: TopK routing + softmax + scatter → grouped expert FFN → weighted gather
9. Shared expert fast path (Gemma 4 MoE — always runs, no routing)
10. Logit soft-cap (Gemma 4)
11. Embedding lookup (tied for Gemma 4; untied for Qwen3-8B+)

---

## 3. Quantization Format: Unsloth Dynamic GGUF

### What UD-Q4_K_XL and UD-Q6_K_XL Actually Are

Unsloth Dynamic GGUFs are **not new on-disk block formats**. They are standard GGUF files where each tensor's quantization type is chosen by Unsloth's per-model imatrix-driven recipe rather than llama.cpp's fixed `_S/_M/_L` mixing rules.

- **Q4_K_XL**: Most tensors use Q4_K. Sensitive tensors upcast to Q5_K, Q6_K, Q8_0, or BF16. Some less-sensitive tensors may downcast below Q4_K.
- **Q6_K_XL**: Base type Q6_K. Sensitive tensors (embeddings, output projection, parts of attention) upcast to Q8_0 or BF16.

There is no custom container, no proprietary metadata, and no Unsloth-specific block layout. Files are fully readable by any standard GGUF parser. The per-tensor type is stored in the standard GGUF tensor-info table (`type` field).

### Block Formats to Implement

The engine needs dequant kernels for exactly these types (covering >99% of tensors in UD-Q4_K_XL and UD-Q6_K_XL files):

#### Q4_K — 4.5 bpw, 144 bytes per 256-weight super-block

```c
typedef struct {
    ggml_fp16_t d;          // FP16 super-block scale
    ggml_fp16_t dmin;       // FP16 super-block min scale
    uint8_t scales[12];     // 8 sub-block scales + 8 mins, 6-bit each, packed
    uint8_t qs[128];        // 256 nibbles packed 2-per-byte
} block_q4_K;               // Asymmetric: w = d·sc·q − dmin·m
```

- 8 sub-blocks of 32 weights
- 6-bit scale/min encoding uses ggml's bespoke `get_scale_min_k4` bit-packing (the one place naive readers break)
- Nibbles packed in pairs: bytes `qs[0..15]` hold sub-blocks 0 (low) and 1 (high), etc.
- ~24-32 VGPRs for the GEMV dequant kernel

#### Q5_K — 5.5 bpw, 176 bytes per 256-weight super-block

```c
typedef struct {
    ggml_fp16_t d;          // FP16 super-scale
    ggml_fp16_t dmin;       // FP16 super-min
    uint8_t scales[12];     // Same 6-bit layout as Q4_K
    uint8_t qh[32];         // High (5th) bit per weight
    uint8_t qs[128];        // Low 4 bits per weight
} block_q5_K;               // Asymmetric: w = d·sc·(q5) − dmin·m
```

- Same sub-block/scale layout as Q4_K — share decode logic with Q4_K (template on high-bit source)
- 5-bit code: `(qs_nibble) | ((qh_bit) << 4)`

#### Q6_K — 6.5625 bpw, 210 bytes per 256-weight super-block

```c
typedef struct {
    uint8_t ql[128];        // Lower 4 bits of each 6-bit quant
    uint8_t qh[64];         // Upper 2 bits, four 2-bit pairs per byte
    int8_t  scales[16];     // 16 signed int8 sub-block scales
    ggml_fp16_t d;          // FP16 super-scale
} block_q6_K;               // Symmetric: w = d · scales[i/16] · (q − 32)
```

- 16 sub-blocks of 16 weights (different from Q4_K/Q5_K's 8×32)
- Dual source arrays (ql + qh) — heavier ALU, ~32-48 VGPRs
- Needs its own kernel — do not try to share with Q4_K
- 210-byte block is not power-of-two aligned; cache-line-aware vectorized loads matter

#### Q8_0 — 8.5 bpw, 34 bytes per 32-weight block

```c
typedef struct {
    ggml_fp16_t d;          // FP16 scale
    int8_t qs[32];          // 32 signed 8-bit quants
} block_q8_0;               // Symmetric: w = d·q
```

- Used heavily for upcast-sensitive tensors in UD files
- 34-byte block is awkward alignment; benefits from software gather to 32-byte-aligned scratch
- Also used for KV cache quantization

#### BF16 — 16 bpw, 2 bytes per weight

- Appears for norms, embeddings, output tensors in many UD-XL files
- gfx906 has no native BF16 — must convert to FP16 at load time or use FP32 FMA
- No dequant kernel needed; just type conversion

#### Q8_K — activation/intermediate only (not on disk)

```c
typedef struct {
    float   d;              // FP32 super-scale
    int8_t  qs[256];        // 256 signed int8 quants
    int16_t bsums[16];      // Precomputed sub-block sums
} block_q8_K;               // 292 bytes per 256 weights
```

- Produced at runtime by the activation quantization pass
- Consumed by `vec_dot_q*_K_q8_K` fused dequant+dot-product kernels
- The `bsums` array lets the dot-product strip out the `dmin·m·Σq` cross-term without re-summing

### Kernel Dispatch Model

The GGUF tensor-info table stores a `type` field (uint32, ggml_type enum) per tensor:

```
GGML_TYPE_Q4_K = 12    GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q5_K = 13    GGML_TYPE_BF16 = 30
GGML_TYPE_Q6_K = 14
```

At load time, read each tensor's type and dispatch to the appropriate kernel. No runtime branching inside kernels by quant type — each type gets its own specialized kernel binary (`.hsaco`). The "Q4_K_XL" / "Q6_K_XL" filename is a marketing label, not a binary format indicator.

### Total Kernel Count

Two kernel families per quant type (fused GEMV for decode + bulk dequant-to-FP16 for prefill):

| Type | Decode GEMV | Prefill dequant | Notes |
|---|---|---|---|
| Q4_K | 1 | 1 | Share scale decode with Q5_K |
| Q5_K | 1 | 1 | Template variant of Q4_K |
| Q6_K | 1 | 1 | Separate kernel, higher VGPR |
| Q8_0 | 1 | 1 | For upcast tensors |
| BF16 | 0 | 1 (type convert) | FP16 passthrough |

Plus: attention kernels (sliding-window, full, GQA), MoE dispatch, softmax, RMSNorm, RoPE, SwiGLU, embedding, logit soft-cap. Total: **~25-30 distinct HIP kernels**.

---

## 4. Engine Architecture

### Runtime Design (hipfire-inspired, Wave64-redesigned)

**Library loading:**
- `dlopen("libamdhip64.so")` at startup — no link-time ROCm dependency
- HIP kernels stored as embedded C++ source strings, compiled lazily via `hipcc --genco` to `.hsaco`, cached on disk with source-hash invalidation
- Optional `dlopen("librocblas.so")` for prefill HGEMM acceleration
- Single binary runs against ROCm 5.7, 6.x, or 7.x runtimes

**GGUF loading:**
- Standard GGUF header + tensor-info parser
- Read `type` field per tensor, validate against supported set (Q4_K, Q5_K, Q6_K, Q8_0, BF16)
- Compute tensor footprint: `block_size_bytes × (n_elements / elements_per_block)`
- `hipMalloc` + `hipMemcpy` weights to device in one contiguous allocation per layer
- BF16 tensors: convert to FP16 during host→device copy

### Decode Path (batch=1, latency-critical)

**Goal: minimize kernel launches per token.**

The decode path is captured as a **HIP graph** (`hipGraphCreate` / `hipGraphLaunch`) at warmup. One graph encodes the entire "decode one token" operation across all layers:

Per layer:
1. Fused RMSNorm + QKV projection (FlashNorm: fold γ into linear weights)
2. Fused QK-Norm + RoPE + KV cache write
3. FlashAttention (sliding-window or full, depending on layer) with Q8 KV cache
4. Fused RMSNorm + routing (MoE) or RMSNorm + up_proj (dense)
5. Fused SiLU·Mul (dense) or fused MoE expert dispatch (MoE)
6. Down projection with residual add

Final: logit head with soft-cap (Gemma 4)

All weight-matrix operations use **fused dequant GEMV kernels** — weights are never materialized as FP16 during decode. The dequant is fused into the dot product.

**Target: captured graph, single `hipGraphLaunch` per token.**

### Prefill Path (batch=N, throughput-critical)

**Goal: maximize GEMM throughput via rocBLAS.**

The prefill path is **not** graph-captured. It uses chunked prefill (chunk_size = 128-512 tokens) with:

1. Custom HIP kernel: bulk dequant Q4_K/Q6_K blocks → FP16 scratch buffer, tiled to stay L2-resident (4 MB L2 ÷ 2 bytes/FP16 = ~2M elements per tile)
2. `rocblas_gemm_ex` with `compute_type=f32_r` (HPA — FP16 inputs, FP32 accumulation) for the matmul
3. For MoE expert GEMMs: `rocblas_gemm_strided_batched_ex` with `batch_count=8` to process all active experts in a single launch

**rocBLAS HGEMM shapes for LLM prefill (Qwen3-8B example, chunk=512):**

| Operation | transA | transB | M | N | K |
|---|---|---|---|---|---|
| QKV projection | T | N | 12288 | 512 | 4096 |
| Attention output | T | N | 4096 | 512 | 4096 |
| FFN gate+up | T | N | 22016 | 512 | 4096 |
| FFN down | T | N | 4096 | 512 | 11008 |

**MoE expert shapes (Qwen3-30B-A3B, chunk=512, per-expert or batched):**

| Operation | transA | transB | M | N | K | batch_count |
|---|---|---|---|---|---|---|
| Expert gate+up | T | N | 1536 | 512 | 2048 | 8 |
| Expert down | T | N | 2048 | 512 | 768 | 8 |

### Speculative Decoding

**Strategy:** Sequence-style verification (not tree — tree verification loses to sequence on quantized models due to dequant overhead scaling with parallel positions).

**Dense targets (Qwen3-8B):**
- ngram-mod drafting as zero-cost baseline (+47% throughput measured on community gfx906 fork)
- Co-resident small drafter: Qwen3-0.6B at Q8 (~0.7 GB) or Gemma 4 E2B
- γ=3-5 draft tokens per verification step

**MoE targets (Qwen3-30B-A3B, Gemma 4 26B-A4B):**
- At Q4, 32GB MI50 has ~11 GB headroom — room for a small drafter alongside the MoE model
- EAGLE-style trained heads as upgrade path (single projection layer, <300 MB) if drafter quality insufficient
- MoE-aware verification: route verifier tokens to the union of experts the drafter chose; skip verification of tokens routed to cold experts (reduces weight-read volume)

---

## 5. Fusion Strategy (Priority-Ordered by gfx906 Impact)

Because gfx906 is launch-bound at batch=1, fusion ROI is dominated by **launch elimination**, not micro-kernel optimization:

### Tier 1: Highest Impact

1. **HIP graph capture of entire decode step.** Even after all kernel fusions, you still have ~5-7 launches per layer × ~48 layers ≈ 240-336 launches per token. Capturing as a graph and submitting via `hipGraphLaunch` amortizes AQL packet overhead. Measured +8-10% generation speed on community fork.

2. **FlashAttention v2 (single-kernel QK^T → online softmax → PV).** Eliminates the most launches in the attention block. On gfx906: shrink tiles to fit 64 KB LDS, use Wave64, use DPP-warp-reduction softmax. Community reference: iacopPBK Q8 FlashAttention tile kernel.

3. **Fused MoE expert dispatch.** End state: 2 kernels per MoE layer instead of 8+ (one for routing + first grouped GEMM with activation in epilogue, one for second grouped GEMM with weighted accumulation). Use persistent grouped-GEMM with CK-style scheduling — single kernel that loops over expert IDs internally.

### Tier 2: Significant Impact

4. **RMSNorm + QKV projection (FlashNorm).** Fold RMSNorm γ into column scales of subsequent linear; defer RMS divide to after matmul. Removes one pass over residual stream per layer.

5. **RoPE + KV-cache write.** Rotate Q,K and write K,V to paged cache in one launch. Saves two launches per layer.

6. **GEMM epilogue fusion.** SiLU activation, bias add, residual add, soft-cap, and output quantization (when next op consumes Q8) all done in the matmul epilogue without HBM roundtrips.

### Tier 3: Polish

7. **Router fusion (MoE).** TopK + softmax + bias + expert-token scatter in one launch. Small compute but at 48 MoE layers × every decode step, it adds up.

8. **SiLU + gate-multiply + (optionally) quantize** between the two expert matmuls. Saves three launches and two HBM round-trips per MoE layer.

### Where NOT to Fuse

Prefill wants throughput, not launch-amortization. Use rocBLAS/hipBLASLt batched HGEMM for QKV/FFN projections during prefill. Reserve hand-fused kernels for decode.

---

## 6. rocBLAS / Tensile on gfx906

### The Problem

ROCm 6.4.0+ stopped shipping pre-compiled Tensile library files for gfx906 in official packages. This breaks `rocblas_hgemm` / `rocblas_gemm_ex` at runtime with:
```
rocBLAS error: Cannot read TensileLibrary.dat: No such file or directory for GPU arch : gfx906
```

Additionally, rocSOLVER in ROCm 7.x does not include pre-compiled Strsm-family kernels for gfx906 (irrelevant for inference but worth noting).

### Solutions (Choose One)

**Option A — Build rocBLAS from source (recommended):**
```bash
git clone https://github.com/ROCm/rocBLAS.git
cd rocBLAS
git checkout release/rocm-rel-7.1
# Edit CMakeLists.txt line ~115 to only build gfx906
./install.sh -a gfx906:xnack-
```
Or with raw cmake:
```bash
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DGPU_TARGETS=gfx906 \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm-gfx906 ..
```
Takes 30-60 min. Produces a complete Tensile library with tuned HGEMM assembly kernels for gfx906.

**Option B — Copy Arch Linux Tensile files:**
Download the Arch Linux rocBLAS package, extract all files containing `gfx906` in the filename (~156 files), copy to `/opt/rocm/lib/rocblas/library/`. Confirmed working by multiple community members on Ubuntu 24.04 + ROCm 6.4/7.x. Risk: ROCm version mismatch can cause `hipErrorInvalidDeviceFunction`.

**Option C — Pin to ROCm 6.3.x** (last version with official gfx906 Tensile libraries). Not recommended for new development.

### Tensile Kernel Configurations for gfx906 HGEMM

The Tensile library for gfx906 contains pre-tuned assembly and source GEMM kernels. The HGEMM variants are identified by `HH` (FP16→FP16) and `HBH` (FP16→FP32 accumulate→FP16 output) in the filename convention:

- `TensileLibrary_Type_HH_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx906.hsaco` — HGEMM TN (transA=T, transB=N) — **this is your primary prefill kernel**
- `TensileLibrary_Type_HH_Contraction_l_Alik_Bjlk_Cijk_Dijk_gfx906.hsaco` — HGEMM NN

Tuned kernel parameter ranges observed in gfx906 Tensile solutions:

| Parameter | Range | Notes |
|---|---|---|
| MacroTile (MT) | 16×16 to 128×128 | 64×64 and 128×128 for large GEMM; 16×16 / 32×32 for small |
| DepthU (unroll) | 8, 16, 32 | Deeper unrolling needed without MFMA to hide FMA latency |
| WorkGroup (WG) | 8×8×4, 16×16×1, 8×16×1 | Third dim is LocalSplitU (>1 tiles K reduction across warps) |
| ThreadTile (TT) | 2×2, 4×4, 8×8, 4×8 | Per-thread output tile |
| VectorWidth (VW) | 2 or 4 | VW4 = half4 loads, full 128-bit memory transactions |
| KernelLanguage | Assembly (ISA906) or HIP Source (ISA000) | Assembly is the fast path |
| PrefetchGlobalRead | Enabled on high-perf kernels | Software pipelining global→LDS overlap with compute |

**gfx906 HGEMM does NOT use MFMA or WMMA.** All math is via `v_pk_fma_f16` (packed FP16 FMA). Practical throughput: ~20-24 TFLOPS on large square matrices, ~8-15 TFLOPS on the tall-skinny shapes typical of LLM prefill.

### Benchmarking Your Exact Shapes

Use `rocblas-bench` to profile every GEMM shape:
```bash
# QKV projection shape (Qwen3-8B, chunk=512)
rocblas-bench -f gemm_ex \
  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
  --compute_type f32_r \
  --transposeA T --transposeB N \
  -m 12288 -n 512 -k 4096 --alpha 1 --beta 0

# MoE expert batched GEMM (Qwen3-30B-A3B, 8 experts)
rocblas-bench -f gemm_strided_batched_ex \
  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
  --compute_type f32_r \
  --transposeA T --transposeB N \
  -m 1536 -n 512 -k 2048 \
  --stride_a $((1536*2048)) --stride_b $((2048*512)) \
  --stride_c $((1536*512)) --stride_d $((1536*512)) \
  --batch_count 8 --alpha 1 --beta 0
```

If specific shapes underperform, run a Tensile benchmarking sweep on those exact shapes:
```bash
# Clone Tensile, configure for gfx906
cd Tensile
mkdir build && cd build
../Tensile/bin/Tensile ../Tensile/Configs/your_custom_hgemm.yaml ./
```
The output `3_LibraryLogic/` YAML files contain the winning kernels. These can be integrated into a custom rocBLAS build.

### Integration Strategy

```
Engine startup:
  1. dlopen("libamdhip64.so") — required
  2. dlopen("librocblas.so") — optional
  3. If rocBLAS available:
       rocblas_create_handle(&handle)
       Verify Tensile library loads for gfx906
       Set prefill_backend = ROCBLAS
     Else:
       Set prefill_backend = CUSTOM_HIP
  4. Load GGUF, allocate device memory, compile/cache kernels
  5. Warm up HIP graph capture for decode path
```

---

## 7. hipfire: What to Port vs What to Redesign

### What hipfire Does Well (Copy These Ideas)

- **Runtime `dlopen` of `libamdhip64.so`** with no link-time ROCm dependency
- **Kernel source embedding + on-disk `.hsaco` cache** with source-hash invalidation
- **Minimizing VGPR usage in dequant/GEMV kernels** — hipfire's HFQ4 format uses 18 VGPRs vs ~39 for llama.cpp Q4_K. Lower VGPR → higher wavefront occupancy → better latency hiding. This principle is THE key insight for gfx906.
- **Q8_0 quantized KV cache as default** (not optional)
- **Batched RoPE, batched causal attention, batched KV writes for prefill**

### What Must Be Redesigned for gfx906

- **Wave32 → Wave64.** Every kernel using `__shfl_*`, DPP, or cross-lane operations must be re-tuned: lane masks, reduction trees, MMVQ tiling, LDS bank conflict layouts all change.
- **Prefill GEMM.** hipfire's prefill is 0.57-0.69× llama.cpp because it doesn't use rocBLAS. The engine needs rocBLAS HGEMM for prefill throughput.
- **Quantization format support.** hipfire only supports its custom HFQ4 format. The engine needs Q4_K, Q5_K, Q6_K, Q8_0, and BF16 to consume Unsloth Dynamic GGUFs.
- **MoE support.** hipfire has none. The engine needs expert routing, grouped GEMM, shared expert path.
- **Speculative decoding.** hipfire has none. The engine needs draft model management, verification loop, token acceptance logic.
- **Gemma 4 / Qwen 3 architecture features.** hipfire doesn't support QK-Norm, sliding-window attention, p-RoPE, soft-cap logits, or any MoE-specific ops.

### Build vs Port Recommendation

**Recommendation: build from scratch, inspired by hipfire.**

The amount of code that would survive a port (dlopen scaffolding, hsaco cache, runtime loader) is maybe 500-1000 lines. Everything else — all kernels, all model-specific logic, all quantization support — must be written new. The Wave32→Wave64 rewrite alone touches every kernel. Starting fresh with hipfire's architectural principles (dlopen, embedded kernels, VGPR-conscious design) but targeting gfx906 from the ground up will be faster and cleaner than fighting a codebase designed around RDNA wave semantics.

---

## 8. Performance Targets

Based on community benchmarks on identical hardware (gfx906, ROCm 7.x, llama.cpp turbo fork + vllm-gfx906):

### Decode (batch=1, generation)

| Model | Community baseline | Target with custom engine | Key lever |
|---|---|---|---|
| Qwen3-8B Q4 | ~80-90 tok/s | **100-130 tok/s** | HIP graph + VGPR-optimized GEMV |
| Qwen3-8B Q4 + spec decode | ~120-130 tok/s | **150-180 tok/s** | +ngram-mod or 0.6B drafter |
| Qwen3-30B-A3B Q4 | ~50-57 tok/s | **65-80 tok/s** | Fused MoE dispatch + HIP graph |
| Gemma 4 26B-A4B Q4 | ~50-57 tok/s (est.) | **65-80 tok/s** | Same |

The ~56 tok/s community ceiling is the number to beat. Getting to 80+ on MoE requires aggressive launch elision — this is where the custom engine's value proposition lives.

### Prefill (batch=N, prompt processing)

| Model | Metric | Target | Notes |
|---|---|---|---|
| Qwen3-8B Q4 | tok/s @ chunk=512 | **800-1200** | rocBLAS HGEMM limited path |
| Qwen3-30B-A3B Q4 | tok/s @ chunk=512 | **400-700** | Batched expert GEMM |

Prefill is compute-bound at chunk≥128; the limit is rocBLAS HGEMM throughput (~20-24 TFLOPS effective on the relevant matrix shapes).

---

## 9. Build and Toolchain

### Compiler Flags

```bash
# Kernel compilation
hipcc --genco -O3 --offload-arch=gfx906 \
  -mwavefrontsize64 -fPIC \
  -o kernel.hsaco kernel.hip

# Known issue: HIP compiler miscompiles butterfly/FWHT code at -O3
# Use -O1 for those specific kernels

# Engine compilation (host-side Rust or C++)
# Link only against libdl, libpthread — no ROCm link-time deps
```

### ROCm Environment

```bash
# Recommended: ROCm 7.1+
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0  # Single GPU

# rocBLAS with gfx906 Tensile (built from source)
export LD_LIBRARY_PATH=/opt/rocm-gfx906/lib:$LD_LIBRARY_PATH

# HIP graph support
export GGML_HIP_GRAPHS=ON  # If integrating with ggml ecosystem
```

### Project Structure (Suggested)

```
engine/
├── src/
│   ├── main.rs (or main.cpp)        # Entry point, CLI
│   ├── gguf.rs                       # GGUF parser + tensor loader
│   ├── runtime.rs                    # HIP dlopen, kernel cache, graph mgmt
│   ├── rocblas.rs                    # Optional rocBLAS integration
│   ├── model/
│   │   ├── qwen3.rs                  # Qwen3 model graph (dense + MoE)
│   │   ├── gemma4.rs                 # Gemma 4 model graph (dense + MoE)
│   │   └── speculative.rs            # Draft model + verification loop
│   └── kernels/
│       ├── dequant_q4k.hip           # Q4_K fused GEMV + bulk dequant
│       ├── dequant_q5k.hip           # Q5_K (template variant of Q4_K)
│       ├── dequant_q6k.hip           # Q6_K fused GEMV + bulk dequant
│       ├── dequant_q8_0.hip          # Q8_0
│       ├── attention_sliding.hip     # FlashAttention sliding-window
│       ├── attention_full.hip        # FlashAttention full (Gemma 4 global)
│       ├── attention_gqa.hip         # GQA attention (Qwen3)
│       ├── moe_dispatch.hip          # Fused routing + grouped GEMM
│       ├── rmsnorm_fused.hip         # FlashNorm + projection
│       ├── rope_kv_write.hip         # Fused RoPE + KV cache write
│       ├── swiglu.hip                # SwiGLU activation
│       ├── softcap.hip               # Logit soft-cap (Gemma 4)
│       └── embedding.hip             # Token embedding lookup
├── tools/
│   ├── bench_rocblas.sh              # rocblas-bench scripts for target shapes
│   └── profile_decode.sh             # hipprof / rocprof scripts
└── README.md
```

---

## 10. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| ROCm drops gfx906 runtime support entirely | High | dlopen architecture means engine only needs libamdhip64.so; can pin ROCm version |
| rocBLAS Tensile gfx906 configs degrade over time | Medium | Build from source, pin branch; or replace with custom prefill HGEMM long-term |
| HIP compiler miscompilation at -O3 | Medium | Known issue; per-kernel optimization level control |
| 64 KB LDS too small for FlashAttention on long contexts | Medium | Reduce tile sizes, accept lower occupancy; Q8 KV cache reduces memory pressure |
| FP16 overflow in attention softmax at long context | Low | FP32 accumulation in softmax; minor register cost |
| Unsloth changes UD recipe to include unsupported types (IQ4_NL, MXFP4_MOE) | Low | Add support as needed; current Q4_K+Q5_K+Q6_K+Q8_0+BF16 covers >99% of files |
| Wave64 kernel development is harder than Wave32 | Low (schedule) | Leverage community gfx906 kernel work from llama-turbo fork |
| Model architecture changes break assumptions | Low | Modular model graph; new architectures = new model file, same kernel library |

---

## 11. Recommended Development Sequence

**Phase 1 — Foundation (weeks 1-3)**
- GGUF parser + tensor loader (standard format, well-documented)
- HIP dlopen runtime + kernel cache + .hsaco compilation
- Q4_K and Q6_K fused GEMV kernels (Wave64, VGPR-optimized)
- Basic dense forward pass: RMSNorm → QKV → attention → FFN → logits
- Validate correctness on Qwen3-8B Q4_K_XL

**Phase 2 — Performance (weeks 4-6)**
- rocBLAS integration for prefill (build from source, benchmark shapes)
- HIP graph capture for decode path
- FlashAttention with Q8 KV cache
- FlashNorm (fused RMSNorm + projection)
- RoPE + KV write fusion
- Profile and tune: target >80 tok/s decode on Qwen3-8B

**Phase 3 — MoE (weeks 7-9)**
- MoE routing kernel (TopK + softmax + scatter)
- Fused grouped-GEMM expert dispatch
- Shared expert path (Gemma 4)
- Batched expert GEMM via rocBLAS for prefill
- Validate on Qwen3-30B-A3B and Gemma 4 26B-A4B

**Phase 4 — Speculative Decoding (weeks 10-11)**
- ngram-mod drafter (near-zero implementation cost)
- Co-resident draft model support
- Sequence-style verification loop
- Acceptance/rejection token management

**Phase 5 — Polish (weeks 12+)**
- Gemma 4 architectural features (sliding-window, p-RoPE, soft-cap, QK-Norm)
- Q5_K support (template variant of Q4_K — low effort)
- Tensile benchmarking sweep for custom problem-size-specific GEMM solutions
- MI60 validation
- 16GB MI50 support (reduced context, tighter memory budget) if desired
