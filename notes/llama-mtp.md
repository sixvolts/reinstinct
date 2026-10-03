# MTP in the llama fork (Qwen3.8-Flash-Next, 4x MI50)

Source: `common/speculative.cpp` (`common_speculative_impl_draft_mtp`) and `tools/server/server-context.cpp` on branch `qwen4exp-mtp`. The draft head is a port of upstream PR 28243 (Qwen NextN), with the draft borrowing the target's token embeddings and LM head.

## Serving policy (ledger L1)

- **Depth:** `n_max = 2`. n_max 3 measured the same within noise, and every extra draft token costs verify compute when several slots are active.
- **No confidence cutoff:** `p_min = 0`, `n_min = 0`, greedy top-1 draft tokens.
- **Adaptive on/off:** drafting is allowed only while at most N slots are generating (`--spec-draft-max-slots 2`, or env `LLAMA_SPEC_MAX_GEN`). Above that, everyone decodes without drafts. The drafter keeps ingesting every target batch while drafting is off, so it is warm the moment load drops. Reason: at 4 concurrent users MTP aggregate was ~64 tok/s versus ~75 without speculation; verify batches are real compute, not overhead.
- **Placement:** draft model on GPU0 (the GPU with spare VRAM), target layer-split across all four.
- **Measured (Swift 1.5 Flash-Next, prod flags):** single stream 49 -> 52-66 tok/s with MTP depending on content (prose 53, code 62-66, long-context 61-64, copy-edit 67-72). Acceptance (accepted / drafted) ~0.76 greedy, ~0.53 at temperature 1.0. ngram-mod on top of MTP adds 0-1 tok/s.

## Draft loop for a single trained head (ledger L2)

1. **Catch-up on every target batch.** Every target decode, prompt ubatches and verify batches alike, is replayed through the MTP layer before the next draft. Input row k pairs token `x[p_k]` with the target's hidden `h[p_k - 1]`: the target's nextn hidden rows are shifted right by one, and the first row of each sequence takes a carried-over `pending_h` from the previous batch. So the MTP layer's own KV cache is always built from **target** hidden states for the whole accepted context, prompt included. The target must emit hidden rows for every position, not just the logits rows.
2. **Draft step 1:** input `(id_last, pending_h)` at `pos0`, where `pending_h` is the target hidden of the last accepted position.
3. **Draft step i > 1:** input `(drafted token, the MTP block's own output hidden from step i-1)` at `pos0 + i`. The draft region's KV grows during drafting and is overwritten by the next catch-up with target-derived entries.
4. **accept(n):** `pending_h = verify_h[n]`, the target's hidden row for the last accepted token (row 0 is the sampled token, row n the n-th accepted draft).

## Checklist for low acceptance at depth (reinstinct: ~35% at K=3)

Things to check:

- **Draft KV not rebuilt from target hiddens.** If accepted positions keep the KV the draft computed from its own hiddens, or prompt positions are skipped, drafts degrade quickly with depth.
- **Off-by-one pairing.** The input at position p is the embedding of `x[p]` together with the target hidden at `p - 1` (the hidden that produced `x[p]`); its output predicts `x[p+1]`. A one-position shift still yields plausible step-1 drafts, which hides the bug.
- **Wrong hidden tensor (Qwen 3.8 hyper-connection models).** In the fork's qwen4exp graph (Flash-Next) the head's hidden input is the full hyper-connection residual after the last trunk layer, `n_embd x hc` values per token, taken *before* the final `hc_head` mix (which doubles as the output norm). Not the collapsed `n_embd` hidden. Inside the head: per-stream RMS norm of that residual times `hnorm` (length `n_embd*hc`); RMS norm of the token embedding times `enorm`, repeated across streams; concat **[embedding, hidden]** per stream; `eh_proj` applied per stream, giving an `n_embd x hc` residual that enters the head's own HC attention mix. Pooling streams before the projection discards the residual. The head's own output residual (same shape) is what chained steps feed back. If Qwen 3.8-27B shares this architecture, check this first.
- **Chained steps feeding the target hidden again** instead of the head's own output hidden.
- **Sampling.** Compare greedy-to-greedy. Our acceptance drops from ~0.76 to ~0.53 at temperature 1.0 at K=2.

## Gemma 4 assistant MTP

Upstream's path (present in the fork, not measured on furnace): the assistant shares the target's KV cache, runs all heads in one graph, and uses the same position for every draft token, following the transformers reference implementation.
