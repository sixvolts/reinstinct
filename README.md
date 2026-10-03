# crossport

Shared working branch between the two gfx906 (MI50) inference projects owned by sixvolts:

| Project | What | Where it runs | Code |
|---|---|---|---|
| **llama fork** ("Furnace" session) | llama.cpp fork tuned for MI50; production server for Qwen3.8-Flash-Next (177B MoE) on 4x MI50 | furnace (4x MI50) | github.com/sixvolts/llamacpp-gfx906-furnace |
| **reinstinct** ("Reinstinct" session) | greenfield Rust + HIP engine; diverges where llama overhead or infrastructure limits us | podcast (2x MI50) | github.com/sixvolts/reinstinct (`main`, work on `review-fixes`) |

The two should stay close. Reinstinct experiments; whatever wins and fits llama's structure gets ported back, and llama-side work reinstinct can use is ported forward. No duplicated effort: check the ledger before starting anything.

This branch holds coordination material only, never either project's source tree.

## Layout

- `LEDGER.md` - every cross-port item: id, direction, source, owner, status, evidence. Single source of truth.
- `notes/` - write-ups one side needs from the other (strategies, kernel designs, pitfalls).
- `baselines/` - benchmark tables. Same model + same card + same settings, or it does not go in a comparison.
- `patches/` - diffs or kernel sources being handed over, under `llama-to-reinstinct/` or `reinstinct-to-llama/`.

## Protocol

1. Furnace coordinates (owner's decision). Either side may add items; Furnace keeps the ledger consistent.
2. Claim an item by setting `owner` and `status: in-progress` in a commit before working on it.
3. Status values: `proposed`, `in-progress`, `ported`, `measured`, `rejected` (with the reason), `n/a`.
4. Evidence means numbers: kernel time before/after, end-to-end tok/s, correctness check used. Link the commit in the destination repo.
5. Small commits, rebase before push (`git pull --rebase`), never force-push this branch.
6. Session messages are for pings ("pushed X, please look"); anything that should outlive the conversation goes here.
