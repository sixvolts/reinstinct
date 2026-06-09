# Navi21 (RDNA2 / gfx1030) Support — Design Doc

Status: **proposal** (no code yet). Written 2026-06-09 against commit `7511836`.

## Goal

Run reinstinct on Navi21-class GPUs — RX 6800 / 6800 XT / 6900 XT (16 GB)
and Radeon Pro W6800 (32 GB) — with correct output first, competitive
throughput second. The engine is currently hard-targeted at gfx906
(MI50, GCN Vega 20, wave64). Navi21 is gfx1030: RDNA2, wave32-native
(wave64-capable), GDDR6 instead of HBM2, plus a 128 MB Infinity Cache.

## Hardware delta that matters

| | MI50 (gfx906) | RX 6800 | RX 6900 XT | W6800 Pro |
|---|---:|---:|---:|---:|
| CUs | 60 | 60 | 80 | 60 |
| Wave size | 64 only | 32 native / 64 mode | 32 / 64 | 32 / 64 |
| VRAM | 32 GB HBM2 | 16 GB GDDR6 | 16 GB GDDR6 | 32 GB GDDR6 |
| VRAM bandwidth | ~1024 GB/s | 512 GB/s | 512 GB/s | 512 GB/s |
| Infinity Cache | — | 128 MB | 128 MB | 128 MB |
| fp32 / fp16 TFLOPS | 13.3 / 26.5 | ~16 / 32 | ~23 / 46 | ~18 / 36 |
| LDS | 64 KB per CU | 128 KB per WGP (64 KB/workgroup) | same | same |

Two consequences frame the whole port:

1. **Decode will be roughly half of MI50.** Decode is weight-bandwidth
   bound (we stream the full weight set per token; the repacked Q4_K
   matvec sustains ~740 GB/s on MI50). A 17 GB model cannot live in the
   128 MB Infinity Cache, so weight streaming runs at GDDR6 speed:
   512 GB/s ceiling → expect ~12-14 tok/s on 27/31B dense models vs
   27-28 on MI50, and proportionally on the rest. The IC does help KV
   cache, activations, router/expert metadata — secondary traffic, not
   the main stream.
2. **Prefill should be at parity or better.** Prefill is compute-bound
   (MMQ int8 GEMM). A 6900 XT has ~1.7× MI50's dot-product throughput;
   even the 60-CU parts clock higher than MI50. The MMQ tiles will need
   re-tuning but the ceiling is higher, not lower.

### VRAM fit (real file sizes from ~/models)

| Model | GGUF size | 16 GB Navi21 | 32 GB W6800 |
|---|---:|:---:|:---:|
| Qwen 3.5 0.8B / 4B | 0.5 / 2.8 GB | yes | yes |
| Gemma 4 E2B / E4B | 3.0 / 4.8 GB | yes | yes |
| Qwen 3.5/3.6 27B | 17 GB | no | yes |
| Gemma 4 31B dense | 18 GB | no | yes |
| Qwen 3.6 35B-A3B MoE | 21 GB | no | yes |
| Gemma 4 26B-A4B MoE | 22 GB | no | yes |

A 16 GB card is an E4B/4B-class machine. The W6800 32 GB covers the
full current model lineup. (No GPU-split support in the engine, and
none planned for this port.)

## What carries over unchanged

The good news, verified against llama.cpp's own arch gating
(`ggml/src/ggml-cuda/common.cuh:690-728` treats RDNA2 and gfx906
identically here):

- **`__builtin_amdgcn_sdot4` (v_dot4_i32_i8) exists on gfx1030.** All
  ~40 dp4a kernels — every `matvec_*_dp4a`, `mmq_gemm_*`, the Q8 KV
  attention — compile and run as-is. This is the single biggest
  portability risk already retired.
- **`__builtin_amdgcn_perm` (v_perm_b32)** is retained on RDNA2
  (`matvec_iq4_xs_dp4a.cpp:75`, GDN state permute).
- **fp16 storage / `__half2float` conversions** — used everywhere for
  scales; RDNA2 fp16 support is a superset.
- **rocBLAS HGEMM** — stock ROCm rocBLAS ships gfx1030 Tensile kernels;
  the dlopen path in `src/hip/rocblas.rs` needs nothing.
- **HIP graphs, streams, pooled buffers** — HIP API level, arch-neutral.
- **All host-side logic** — repacking, GGUF loading, spec-decode,
  serve, SuperQuant tiering.

## What breaks, by category

Inventory from a full kernel audit (June 2026):

### 1. Arch target plumbing (trivial)

- `src/runtime/mod.rs:28` — `DEFAULT_ARCH = "gfx906"`; hipcc is invoked
  with `--offload-arch=$arch` and the kernel cache already keys on arch.
  `REINSTINCT_OFFLOAD_ARCH=gfx1030` already gets most of the way.
- Gap: no runtime detection. Add `hipDeviceProp_t::gcnArchName` →
  derive arch at startup, keep the env var as override. One small
  change in `src/hip/mod.rs` + `KernelCache::new`.

### 2. Wave64 assumptions (the core of the port)

The entire kernel suite assumes `warpSize == 64`:

- `__shfl_xor(x, 32/16/8/4/2/1)` reduction cascades (~8 kernels).
- "256 threads = 4 waves" grid math (~20 kernels).
- ROWS=2-per-wavefront row mapping in the repacked matvecs.
- `__ballot` with 64-bit masks.

**Decision: compile everything `-mwavefrontsize64` for gfx1030 in
phase 1.** RDNA2 executes wave64 natively (it dual-issues the two
halves); HIP supports the flag and `warpSize` becomes 64. This makes
the wave-size assumptions *correct by construction* and turns the port
into a DPP problem instead of a rewrite-every-reduction problem.
Wave32 conversion of hot kernels is a phase-3 tuning item, not a
correctness item.

### 3. `gfx906_dpp.h` (must be shimmed)

The 109-line DPP reduction header is GCN-specific:

- `row_shr` / `row_shl` / quad_perm survive on RDNA2 (DPP16), but
  **`row_bcast15` / `row_bcast31` and the wave-level shifts were
  removed in RDNA**, and our `row_ror:8` usage needs verification
  against the RDNA2 ISA manual.
- `ds_swizzle_b32` still exists on RDNA2 (offset-mode), so the xor16
  step likely survives — verify.
- The final `__shfl_xor(x, 32)` cross-half step works in wave64 mode.

**Plan:** make the header arch-conditional. `#if defined(__gfx906__)`
keeps today's code; the `#else` branch implements every `xorN` as plain
`__shfl_xor`. That fallback is *correct everywhere* and costs little:
these reductions sit in bandwidth-bound kernels where a few extra VALU
ops per wave are noise. Re-introducing RDNA2 DPP8/DPP16 variants is
phase-3 polish.

### 4. LDS and occupancy (re-tune, not redesign)

- Static/dynamic LDS sizes all fit the 64 KB-per-workgroup limit RDNA2
  shares with GCN. The one kernel that reasons explicitly about "64 KB
  per CU" (`gdn_recurrent_step_fused_batched_lds128.cpp:11`) gets the
  same workgroup budget; the *sharing* differs (two CUs per WGP pull
  from 128 KB), which changes co-residency, not correctness.
- The LDS bank-conflict padding (stride+1, 64-bank assumptions) stays
  valid: RDNA2 LDS is 32 banks × 2 (wave64 mode behaves like GCN for
  our strides). Verify with rocprof once running.
- `__launch_bounds__` values (`(256,2)`, `(128,4)`, `(64,2)` across ~8
  kernels) encode MI50 VGPR/wave budgets. RDNA2 has a different
  register file (1024 VGPRs per SIMD32, allocated in larger granules)
  — leave the bounds in place for phase 1 (they're hints, not
  correctness), re-tune in phase 3 with rocprof occupancy data.

### 5. Performance constants

- `N_ROWS_MAX=4` batched-matvec cap, MMQ `BM=64/BN=64/BK=4` tiles,
  `n_splits = ceil(seq/256), cap 16` in split-K attention, block=256
  everywhere — all tuned on 60-CU/wave64/HBM. They will *run* on
  gfx1030; they will not be optimal. Phase 3 sweeps, same methodology
  as the original MI50 tuning sessions (3-pass averages, tok/s).

## Phased plan

**Phase 0 — plumbing (half a day).**
Runtime arch detection from `gcnArchName`; `-mwavefrontsize64` added to
hipcc flags when arch is gfx10xx; CI-style check that all 120 kernels
compile for gfx1030. No hardware needed.

**Phase 1 — correctness on hardware (1-2 days with the card in hand).**
gfx906_dpp.h portable fallback; run the full GPU oracle test suite
(consistency tests with `set_dp4a(false)` baselines + top-K asserts,
per the established methodology); chase any kernel that miscompares.
Expected trouble spots: ds_swizzle semantics, anything with inline asm
(`v_add_f32_dpp` in reductions), the GDN lds128 kernel.
Exit criteria: E4B and Qwen 4B produce token-identical output to MI50
greedy decode; all tests green.

**Phase 2 — benchmark + ship as supported (1 day).**
Full bench sweep (tok/s, prefill + decode) on the target card; README
table gains a Navi21 column; document the 16 GB model-fit constraints.
Wave64-mode numbers are the baseline we publish.

**Phase 3 — RDNA2 tuning (open-ended, data-driven).**
In expected-ROI order:
1. MMQ tile/launch_bounds re-tune for prefill (highest ceiling delta).
2. Wave32 rebuild of the repacked matvec family — doubles wavefronts
   in flight on the bandwidth-critical path; RDNA2 may extract more of
   the 512 GB/s with more, smaller waves.
3. DPP8/DPP16 reduction variants to replace the shfl fallback.
4. Split-K / N_ROWS / block-size sweeps.
Each item gets the standard treatment: measure, keep if ≥2%, document.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Wave64-mode perf penalty on RDNA2 (dual-issue halves) | Medium — could eat part of the already-halved BW budget | Phase 3 wave32 path for hot kernels; penalty mostly hits VALU-bound kernels, ours are BW-bound |
| `ds_swizzle` / inline-asm DPP behaving differently | Low — caught by oracle tests | shfl fallback shim covers every case |
| rocBLAS gfx1030 HGEMM slower than Tensile-tuned gfx906 | Low — HGEMM only backs the non-MMQ prefill path now | MMQ path is our own code |
| 16 GB cards can't run the flagship models | Certain | Document clearly; W6800 32 GB is the real target for big models |
| ROCm version support for gfx1030 drifting | Low | gfx1030 is still in ROCm support matrix (unlike gfx906, ironically) |

## Why bother (given the bandwidth halving)

- gfx906 is EOL in ROCm; gfx1030 is still supported. This port is the
  engine's path to *currently supported* silicon.
- W6800 32 GB cards are appearing used at MI50-class prices, with
  display outputs, lower idle power, and no cooling hacks.
- The port forces the codebase to grow a second arch cleanly (runtime
  detection, conditional DPP header, per-arch tuning tables) — which is
  90% of the work for any *future* arch (gfx1100/RDNA3 has the same
  shape of differences plus WMMA).
