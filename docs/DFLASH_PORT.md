# DFlash port — block-diffusion drafting for Gemma 4

Spec and plan for running `gemma4-31b-it-dflash-Q8_0.gguf` as a drafter.
Written against the paper (arXiv 2602.06036), the reference implementation
(`github.com/z-lab/dflash`, `dflash/model.py`), and the upstream config
(`huggingface.co/z-lab/gemma-4-31B-it-DFlash/config.json`). Everything
below is taken from those three sources, not inferred from the GGUF.

## Why it is different from the MTP drafter we already run

The `gemma4-assistant` drafter is autoregressive: it proposes one token,
feeds it back, proposes the next. K tokens cost K sequential drafter
forwards, and one bad token poisons the rest of the chain.

DFlash proposes an **entire 16-token block in a single forward pass**. The
block's first position holds a token we already know (the anchor); the
other 15 are `MASK`, and the drafter denoises them all at once. There is
no chain to accumulate error along, and drafting cost stops scaling with
K — which is what lets it use 5 layers at the target's full width where
the assistant had to stay tiny.

It also conditions far more strongly on the target. EAGLE-style drafters
fuse target hidden states into the draft model's *input*, where the signal
dilutes with depth. DFlash injects them into the **Key and Value
projections of every draft layer**, so conditioning stays strong no matter
how deep the drafter is.

Measured baseline to beat: the assistant head tops out at **α = 0.63**
(K=1 ceiling, chat-templated, 8 prompt classes — see `scripts/` bench),
which makes plain MTP net-negative and caps adaptive-K at +1.7%.

## The checkpoint

| | |
|---|---|
| arch | `dflash`, but the draft body is **Qwen3-style** (`model_type: qwen3`) |
| blocks | 5 |
| hidden | 5376 (= target hidden) |
| FFN | 10752, **SwiGLU / SiLU** (`hidden_act: silu`) |
| attention | GQA 64 Q heads / 8 KV heads, `head_dim` 128 |
| norms | pre-norm only — `attn_norm`, `ffn_norm`. **No Gemma sandwich norms** |
| head norms | `attn_q_norm`, `attn_k_norm`, both `[128]` (per-head-dim RMSNorm) |
| layer types | blocks 0-3 `sliding_attention` (window 2048), block 4 `full_attention` |
| RoPE | theta 1e6, applied over the full `head_dim` |
| rms eps | 1e-6 |
| block size | 16 |
| mask token | 4 |
| logit softcap | 30.0 |
| tied | shares the **target's** `token_embd` and LM head — the GGUF has neither |

55 block tensors (11 × 5) plus `fc.weight`, `enc.output_norm.weight`,
`output_norm.weight`. Note what is *absent*: no `attention_conv`, no
`mlp_conv`, no candidate-selector codebooks. Those belong to `DFlash2`;
this is the base `DFlashDraftModel`, which is the simpler of the two.

### Target layers — an off-by-one worth getting right

The GGUF says `dflash.target_layers = [2, 13, 24, 36, 47, 58]`. The
upstream config says `target_layer_ids = [1, 12, 23, 35, 46, 57]`.

They differ by exactly one because `extract_context_feature` indexes
HF's `hidden_states` list with `offset = 1`, and `hidden_states[0]` is the
embedding output. So the GGUF stores the *hidden-state index*, and what we
actually need is:

> the output of blocks **1, 12, 23, 35, 46, 57** (0-indexed), after that
> block's residual, before the next block.

Reading the GGUF value as a block index would take the outputs of blocks
2,13,24,36,47,58 — one block late everywhere. That would not crash; it
would just quietly cost acceptance. Worth an A/B once the tap works: the
right reading should be clearly better.

## Forward pass

### 1. Context features (once per target forward)

Concatenate the six tapped hidden states along the feature axis, then
project and normalize:

```
Ht = RMSNorm_enc( Wc · [ H(1) ; H(12) ; H(23) ; H(35) ; H(46) ; H(57) ] )
```

* `Wc` = `fc.weight`, `[32256 → 5376]` (32256 = 6 × 5376)
* `RMSNorm_enc` = `enc.output_norm.weight`

`Ht` has one row per target position. It is computed for the prefill pass,
and again after every verify pass for the positions that verify produced.

### 2. Draft block forward

Input is the 16 block positions: `[anchor, MASK × 15]`, embedded from the
**target's** `token_embd` — **raw and unscaled**. The config has no
`input_embedding_scale`, so the reference's default of 1.0 applies. This is
*not* the target's own embedding path, which multiplies by `sqrt(hidden)`;
do not reuse that.

Per layer `i`, with draft hidden `Hd` (16 rows) and context `Ht`:

```
Qi = WiQ · Hd                        # queries from draft tokens only
Ki = [ WiK · Ht ; WiK · Hd ]         # context keys FIRST, then draft keys
Vi = [ WiV · Ht ; WiV · Hd ]
```

Then, in this order (matching `Qwen3DFlashAttention.forward`):

1. `q_norm` on Q (per head_dim), `k_norm` on the **concatenated** K —
   context rows included.
2. RoPE on Q and on the **concatenated** K — context rows included.
   Positions: context rows take their real target positions; the 16 block
   rows take `start .. start+15`. In the reference this falls out of
   `cos[..., -q_len:, :]` for Q against the full `cos` for K.
3. Attention. Blocks 0-3: causal with a 2048 window. Block 4: **no mask at
   all** — fully bidirectional across context *and* block. This follows
   from `is_causal = (layer_type == "sliding_attention")`, so the full
   layer is non-causal and its `sliding_window` is `None`.

Context rows bypass Q, the output projection, the residual update and the
FFN entirely — they are only ever extra KV entries.

Rest of the block is ordinary Qwen3: `h += attn(attn_norm(h))`,
`h += swiglu(ffn_norm(h))`.

### 3. Logits

`output_norm` → the **target's** LM head → `tanh(logits/30)*30` softcap.
The hidden at block position `j` predicts the token **at** position `j`
(denoising), not `j+1`. Only positions 1..15 are used; position 0 is the
anchor we already had.

### 4. KV cache

The draft cache is persistent and holds only the **context** KV — the
block's own KV is appended during the forward and then discarded
(`_crop_to(cache, start)` in the reference). Each round appends the newly
produced context rows. So the draft cache length always equals the number
of accepted tokens so far.

## Draft / verify loop

```
prefill target → first token; tap 6 layers → Ht for all prompt positions
loop:
  block = [ output[start] , MASK × 15 ]
  draft forward → fill block[1..15]
  target verifies all 16 positions in one causal pass (hidden states on)
  accept = longest prefix where draft[j] == target_argmax[j-1]
  bonus  = target_argmax[accept]            # always one free token
  produced = accept + 1                     # 1..16 tokens per round
  start += produced
  Ht = fresh context features from the verify pass, first `produced` rows
```

Per round: **one** target forward, **one** draft forward, 1-16 tokens.

## Implementation plan

Staged so each step is separately testable against the CPU oracle.

**1. Target-layer tap** — `gemma4.rs` must expose the outputs of six
blocks from both the prefill and the verify forward, concatenated into
`[P, 32256]`. This is the foundation and is useful on its own: it is also
what a `dump-traces` upgrade would need to produce EAGLE-3-grade training
data, which the current 34 GB corpus cannot (it stores one post-norm hidden
per step, not six mid-stack ones).

**2. `src/model/dflash.rs`** — GGUF parse + shape validation, mirroring
`model/gemma4_assistant.rs`.

**3. `src/runtime/dflash.rs`** — the draft forward. Reuses the existing
repacked matvec / MMQ machinery for every projection. The genuinely new
piece is attention over `[ctx | block]` with a 16-row query tile, in two
variants (causal+window, and unmasked). Everything else the engine
already has.

**4. Loop + CLI** — a `dflash-gen` subcommand mirroring `mtp-gen`, then
the same 8-class α benchmark for a like-for-like comparison against the
assistant's 0.63.

### Risks

* The block-4 bidirectional attention has no analogue in the engine today;
  every existing attention kernel is causal or sliding-causal.
* The `[ctx | block]` KV layout is not the engine's KV cache layout — the
  context rows are a separate, persistent, draft-owned cache.
* The target-layer off-by-one above is silent if wrong. A/B it.
* Whether the tap can be folded into the existing HIP graph capture
  without breaking it is unknown; worst case the tap costs a graph break.
