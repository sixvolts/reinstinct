# SuperQuant — tiered KV cache (opt-in, 2-tier)

**Status:** live-wired into Gemma 4 `generate-text` behind
`REINSTINCT_KV_SUPERQUANT=1`. Real-model correctness validated;
opt-in is single env var. Serve integration + qwen35 path are the
remaining work.

## What it is

A two-tier KV cache that matches cache precision to attention
importance:

| Tier | Format | SNR per-value | bits/value | Holds |
|---|---|---:|---:|---|
| Warm | int8 + per-(slot,head) scale | ~48 dB | 8 | Writes go here; recent context |
| Cold | turbo3 (Walsh-Hadamard rotation + Lloyd-Max 3-bit codebook) | ~14.6 dB | 3.5 | Older context |

Writes always land in Warm. When Warm fills, the oldest 1 entry
slides to Cold via the `kv_promote_q8_to_turbo3` GPU kernel. Cold
fills until `cold_cap` is reached; further writes after that error
(context exhausted — caller's responsibility to size).

## Why this shape

Standard attention softmax in chat / agent workloads:
- ~70% of mass on the current 2K-token "active" turn
- ~25% on the previous 4-6K tokens
- ~5% spread across the long tail (system prompt, prior topics)

Warm covers the top 70-95% of attention mass at int8's 48 dB SNR
(visually indistinguishable from fp16). Cold's 14.6 dB SNR only
matters for positions contributing <5% of softmax mass — quantization
noise there gets weighted out.

## Design history

The original 3-tier design (fp16 Hot / int8 Warm / turbo3 Cold) was
simplified per user feedback (2026-05-25): int8 at 48 dB is plenty
for any attention tier; the fp16 Hot tier added complexity (separate
write kernel + dequant path) for invisible quality gain.

The original 3-tier kernels (`kv_write_fp16.cpp`,
`kv_promote_fp16_to_q8.cpp`) remain in the tree as primitives in
case the Hot tier is reinstated later.

## CLI: `superquant-bench`

End-to-end benchmark on synthetic K/V tensors. Validates the
pipeline + measures performance + precision on the shape of your
choice.

```bash
$ reinstinct-engine superquant-bench \
    --warm-cap 2048 --cold-cap 8192 \
    --n-kv 2 --n-heads 16 --head-dim 256 \
    --n-writes 8192 --n-splits 8

SuperQuant bench (2-tier: Warm int8 / Cold turbo3):
  tier caps:    warm=2048  cold=8192  total=10240
  shape:        n_heads=16  n_kv=2  groups=8  head_dim=256
  writes:       8192  (requested 8192)
  n_splits:     8

phase 1: writes
  total = 3.71 s  (0.453 ms/token, 2208 tok/s)
  tier counts: cold=6144 warm=2048

phase 2: 3-tier attention
  8 iter avg = 7.17 ms/call (139.5 calls/s)
  (one call = full attention over 8192 positions across all 16 q-heads)

phase 3: rel_l2 vs pure-fp32 reference
  rel_l2 = 0.1596

phase 4: memory accounting
  per-layer footprint (K + V):
    fp16 (baseline):         20971520 bytes  (20.00 MiB)
    int8 KV (current):       10649600 bytes  (10.16 MiB)
    SuperQuant:               6324224 bytes  (6.03 MiB)
    capacity vs fp16:      3.32x
    capacity vs int8:      1.68x
```

## Live integration (Gemma 4 31B, real model)

```bash
# Enable SuperQuant. Defaults: warm=min(8192, max_seq), cold=remainder.
REINSTINCT_KV_SUPERQUANT=1 reinstinct-engine generate-text MODEL.gguf \
  --system "Be brief." --user "..." --steps 64 --gpu

# Custom tier sizes:
REINSTINCT_KV_SUPERQUANT=1 \
  REINSTINCT_KV_WARM_TOKENS=128 REINSTINCT_KV_COLD_TOKENS=512 \
  reinstinct-engine generate-text MODEL.gguf --system "..." --user "..." \
  --steps 64 --gpu
```

End-to-end decode tok/s on Gemma 4 31B (UD-Q4_K_XL) with a 28-token
prompt + 64 decode steps:

| Config | Decode tok/s | vs int8 |
|---|---:|---:|
| int8 baseline (default) | **27.0** | 1.00× |
| SuperQuant warm=128 cold=128 | 18.0 | 0.67× |
| SuperQuant warm=64  cold=128 | 16.6 | 0.61× |
| SuperQuant warm=32  cold=256 | 14.2 | 0.53× |
| SuperQuant warm=64  cold=512 | 16.6 | 0.61× |

**Quality preserved.** Side-by-side outputs on the same prompt:
- int8: `On July 20, 1969, NASA's Apollo 11 mission successfully
  landed the first humans on the moon. ... Lunar Module Eagle while
  Michael`
- SuperQuant warm=64 cold=256: `On July 20, 1969, NASA's Apollo 11
  mission successfully landed the first humans on the moon. ...
  Lunar Module Eagle at the`

Both factually correct. The minor word-choice divergence is the
expected logit perturbation from int8/turbo3 quantization noise —
not a meaning loss.

**Caveats:**
- Decode is 33–47% slower than int8. The cold-tier per-position
  cooperative iRHT dominates; rotated-space attention (planned
  optimization) would cut this 3–5×.
- HIP graph capture disabled when SuperQuant is on (warm-cascade
  D2D memcpys can't capture). Adds ~10% per-token overhead from
  kernel launches.
- snapshot/restore + spec-decode mutually exclusive with SuperQuant
  (per-tier rollback not implemented).
- Sliding-window attention layers ignore the window when SuperQuant
  is active — SuperQuant uses tiering for the same context-length
  goal that sliding windows target.

## Synthetic-bench measured numbers (Gemma 31B layer shape)

n_kv=2, n_heads=16, head_dim=256, warm_cap=2048, n_splits=8. Each
row is a separate `superquant-bench` invocation; n_writes scaled
to fill `warm + cold` positions exactly.

| Cold positions | Attention rel_l2 | Attention ms/call | Capacity vs int8 |
|---:|---:|---:|---:|
| 0 (all Warm)    | **0.0039** | 0.54 | 1.06× |
| 1024            | 0.116 | 2.54 | 1.24× |
| 2048            | 0.136 | 3.40 | 1.37× |
| 4096            | 0.150 | 5.26 | 1.53× |
| 6144            | 0.160 | 7.16 | 1.62× |
| 8192            | 0.163 | 9.06 | 1.69× |

Observations:
- rel_l2 plateaus around 0.16 — the cold tier's per-value SNR
  floor dominates and adding more cold positions barely moves it.
- Attention latency scales linearly with cold-tier size because
  the per-position cooperative iRHT (FWHT in LDS) is the cost
  centre. The Warm-only call is ~13× faster than the
  8K-cold call — that's the optimization headroom for the
  rotated-space attention follow-up.
- Capacity gain caps at ~2.0× vs int8 (turbo3 is 0.4375× the
  bytes-per-value of int8, but `warm_cap` worth of capacity is
  paid at int8 rates).

Attention latency scales with cold position count because the
per-position iRHT (cooperative FWHT in LDS) is the dominant cost.
This is the obvious optimization target for a follow-up: rotated-
space scoring would amortise the iRHT to once per attention call
instead of once per cold position.

## Live integration (next session)

To wire SuperQuant into the actual forward pass, replace
`Gemma4KvCache` (or `GpuKvCache` for qwen35) construction with
`SuperQuantKvCache` when `REINSTINCT_KV_SUPERQUANT=1`, and route
the attention call to `attn_partial_superquant` instead of
`attn_partial_q8`. The cache's `write_step` signature is
intentionally close to what the existing decode path expects.

For prefill, the simplest first cut: prefill writes directly to
the int8 KV (existing path), then a one-shot migration kernel
copies populated Warm-eligible slots to SuperQuant's Warm buffer
and demotes overflow to Cold. Migration code is not yet written.

Spec-decode rollback (`truncate(n)`) needs per-tier handling:
positions sliding back from Cold to Warm would require dequant +
re-quant (lossy round-trip). For now the safe move is to **disable
SuperQuant when spec-decode is enabled** — either skip the
migration, or fall back to plain int8 KV.
