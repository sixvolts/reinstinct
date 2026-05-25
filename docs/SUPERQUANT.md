# SuperQuant — tiered KV cache (opt-in)

**Status:** design lock + foundation kernels (Phase 1 + 2a)
shipped; live integration (Phase 2b/3) is follow-up work.

## What

A three-tier KV cache that matches cache precision to attention
importance:

| Tier | Format | SNR | bpv | Holds |
|---|---|---:|---:|---|
| Hot | fp16 | ∞ | 16 | Current turn (writes go here) |
| Warm | int8 (sym, per-head scale) | ~48 dB | 8 | Previous 2 turns |
| Cold | turbo3 (RHT + Lloyd-Max 3-bit) | ~14.6 dB | 3.5 | Older context |

Default tier sizes (opt-in via `REINSTINCT_KV_SUPERQUANT=1` or
`--kv-superquant`):
- Hot:  last 2K tokens (configurable `REINSTINCT_KV_HOT_TOKENS`)
- Warm: 8K tokens (configurable `REINSTINCT_KV_WARM_TOKENS`)
- Cold: everything older

For a 32K-token conversation:
- Pure fp16 KV: 1.0× capacity (baseline)
- Current int8 KV: 2.0× capacity
- **SuperQuant: ~3.5× capacity** with worst-case 14.6 dB SNR
  applying ONLY to positions older than ~10K

## Why this matches chat attention

Attention softmax on chat workloads:
- ~70% of mass goes to the current turn
- ~20% to the previous 2 turns
- ~10% spread across older context (mostly system prompt + retrieval)

Tier sizing maps to that distribution. Most attention computation
runs at full precision; only the long tail uses turbo3, where its
14.6 dB SNR is dominated by per-token softmax weights <1%.

## Tier-demotion trigger (hybrid)

- **Chat path** (`/v1/chat/completions`): the chat template
  renderer calls `cache.mark_turn_boundary()` after each
  assistant turn. Demotion happens then: Hot tokens older than
  the most-recent turn slide to Warm; Warm tokens older than
  N-back turns slide to Cold.
- **Raw-completion path** (`/v1/completions`): no turn
  boundaries → fall back to a position-based sliding window.

## Architecture

```text
src/runtime/kv_superquant.rs    SuperQuantKvCache (3 buffers + tier
                                tracking) — orchestrates demotion
src/quant/turbo3.rs             RHT + 3-bit codebook + encode/decode (DONE)
src/runtime/kv_turbo3.rs        TurboKvCache + write kernel        (DONE)
kernels/turbo3_quantize.cpp     GPU encode (DONE)
kernels/turbo3_dequantize.cpp   GPU decode (DONE)
kernels/kv_write_turbo3.cpp     decode-step write to turbo3 slot   (DONE)

# === Phase 2b TODO ===
kernels/kv_promote_fp16_to_q8.cpp    sym int8 quant of a slot range
kernels/kv_promote_q8_to_turbo3.cpp  int8 dequant → RHT-encode pipeline
kernels/attn_decode_superquant.cpp   3-tier attention (sees all 3 buffers,
                                     dequants on-the-fly per tier)

# === Phase 3 TODO ===
src/runtime/kv_superquant.rs (Rust glue)
  - tier_for_position(pos) -> &mut tier
  - mark_turn_boundary(turn_id) -> demote according to policy
  - reset(), truncate(n)
  - snapshot()/restore() for spec-decode rollback

src/serve/mod.rs                    plumb `--kv-superquant` to state ctor
src/main.rs                         CLI flag + env var
MANUAL.md                           docs + recommended config table
```

## Attention kernel (Phase 2b — the hard part)

**Option A — naive 3-pass:**
For each of {Hot, Warm, Cold}: run the existing-style attention
kernel reading from that tier's storage, output partial logits.
Combine via a final stable-softmax merge step.

LOC: ~400 (mostly per-tier dequant variants of our existing
split-K attention).

**Option B — rotated-space single-pass (preferred):**
Pre-rotate Q with the RHT once per head; score Hot+Warm in their
native space (un-rotated); score Cold in RHT space (no per-entry
iRHT needed since Q·K is rotation-invariant). Combine logits in
a single softmax. After softmax, multiply by V from each tier
(V needs the same treatment — Cold stores V in RHT space).
Single iRHT at the end on the per-head accumulator.

LOC: ~300, faster, but trickier to debug. Likely the right end
state once the naive version is verified.

## Spec-decode interaction

Spec-decode rollback (`state.truncate(n)`) needs per-tier
truncation. If `n` falls in the Hot range: simple truncation. If
it falls in the Warm or Cold range: more involved — we'd need to
re-promote the affected tier suffix back to fp16. For now the
proposal is to **disable SuperQuant when spec-decode is on**, and
re-enable later if the snapshot/restore cost is acceptable.

## Tests needed

1. fp16→int8→turbo3 demotion pipeline: round-trip SNR per tier
2. Attention output equivalence: SuperQuant attention vs pure
   fp16 attention on the same K/V — check that attention output
   diff is within tolerance bounded by the tier mix
3. Long-context decode correctness on a real model (gemma-31B at
   16K context, compare token-by-token vs int8 KV baseline)
4. Capacity test: confirm we can fit context that wouldn't fit
   with int8 KV

## Defaults summary

```text
REINSTINCT_KV_SUPERQUANT=1     # opt-in master switch
REINSTINCT_KV_HOT_TOKENS=2048  # fp16 sliding window
REINSTINCT_KV_WARM_TOKENS=8192 # int8 mid-tier
                               # cold = anything older
```

Disabled (default) → existing int8 KV cache, no behavior change.

## What ships today

- `crate::quant::turbo3`: CPU reference + RHT + Lloyd-Max codebook
- `kernels/turbo3_quantize.cpp` + `dequantize.cpp`: GPU encode/decode
- `kernels/kv_write_turbo3.cpp`: decode-step single-token write
- `crate::runtime::kv_turbo3::TurboKvCache`: Rust cache struct
- 11 GPU + CPU oracle tests, all passing
- Precision validated: 14.6 dB SNR on round-trip (matches theory)
- Storage cost validated: 2.0× compression vs int8 at head_dim=256

The foundation is **complete and tested**. Live attention
integration is the remaining work.
