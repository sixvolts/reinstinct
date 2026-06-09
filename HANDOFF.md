# Handoff — reinstinct (2026-06-09)

Snapshot for moving the project to another machine. Picks up after commit `1afa1a9`.

## What just shipped

Three commits worth of work, most of it concentrated in the last ~10 days:

| Commit | What |
|---|---|
| `1afa1a9` | **dump-traces CLI** — EAGLE drafter training data dumper. New `reinstinct-engine dump-traces` subcommand: walks a JSONL of prompts, chat-templates each, prefills + greedy-decodes Gemma 4 31B for N steps, writes `(prev_tok, label_tok, hidden_b fp16[H])` traces to a binary file. Resume-safe (header counter rewritten per prompt). |
| `dba4f31` | **adaptive-K MTP** — rolling-window α tracker that flips MTP off when sustained acceptance drops below threshold. Defaults: serve ON (`MIN_ALPHA=0.55`, `WINDOW=8`), mtp-gen CLI OFF (measurement parity). Also lands per-block FFN width support (Gemma 4 E2B has a 35-entry `feed_forward_length` array) and the `align-check` CLI for K=1 ceiling sweeps between any two Gemma 4 models. |
| `5be6288` | **GPU power-tuning systemd unit** — `scripts/reinstinct-gpu-tune.{sh,service}` applies 300 W + mclk=1125 + sclk=1825 + perflevel=high at boot. Survives reboot; pp_table edits via `upp` are otherwise volatile. |

Earlier in the same session: README perf table refresh, OWUI integration polish, ARCHITECTURE.md, Q4_K matvec bench, SuperQuant docs reframe.

## The MTP / spec-decode arc — current verdict

Net: **adaptive-K is +1.7% over plain decode** on a 24-prompt diverse benchmark. That's the ceiling without a better drafter. Detailed numbers:

- Plain MTP (assistant.gguf, K=4): −2.3% net vs base decode across diverse workloads. Code/JSON/factual gain ~15%, but creative/longform run α≈0.4 and lose ~25% because verify cost is paid on every rejected draft.
- Adaptive-K (above): salvages the disasters (creative +25%, longform +35% over plain MTP) at the cost of minor false-trip regressions on borderline-α math/reasoning prompts.
- E2B-as-drafter explored and **ruled out**: K=1 ceiling α = 0.628 vs assistant's 0.621 — basically identical. A bigger general-purpose drafter is not the lever.

The only path past +1.7% net is a **custom-trained EAGLE drafter** — and that's what the trace-dump infrastructure exists for.

## Where the next move is — EAGLE drafter retrain on MI100

User is swapping in an MI100 (CDNA1 / gfx908, MFMA matrix cores) on the new machine for training. Plan:

1. **Trace dump** — DONE on MI50. `gemma4_31b_eagle_train.bin` exists (see below). 50,000 prompts × up to 64 decode steps each, Gemma 4 31B target argmax + post-norm hidden states. ~34 GB.
2. **Train an EAGLE head** matching the existing `gemma-4-31B-it-assistant.gguf` shape (4 transformer layers, takes target hidden state + previous token → predicts next token). PyTorch training loop, cross-entropy vs target argmax. Estimated 1 GPU-day on MI100.
3. **Quantize to Q8**, drop GGUF in `~/models/gemma4-mtp/`, swap in via `mtp-gen --drafter` or the serve env knob. **No engine changes needed** — drafter must match the existing assistant.gguf tensor layout exactly so the runtime's MTP path picks it up unchanged.
4. **Re-run the eval harness**:
   - `reinstinct-engine align-check <target> <drafter> --prompt ...` — quick K=1 α ceiling check
   - `/tmp/mtp_alpha/*.sh` — full 24-prompt α benchmark (script may need recreating; see "Inventory" below)
   - EAGLE-2/3 papers report α ≥ 0.85 for well-trained heads; assistant currently caps at 0.61. Target α ≥ 0.78 to make MTP net-positive without adaptive.

## Files / state that must travel

Everything in git travels via `git push` (see "Push status" below — you'll need to do this manually). Things NOT in git:

### Critical (the EAGLE training run depends on these)

| Path | Size | Notes |
|---|---|---|
| `~/datasets/traces/gemma4_31b_eagle_train.bin` | **34 GB** | The trace dataset. Rsync this. Format documented in `src/main.rs` above `dump_traces_cli` (32 B header + per-prompt 16 B + per-step `prev u32, label u32, fp16[5376]`). |
| `~/datasets/traces/run_dump.sh` | 1 KB | Restart-loop wrapper. Only needed if you re-run dump on the new machine. |
| `~/datasets/ultrachat_200k/prompts_50k.jsonl` | small | Input prompts for the dump (already consumed; only needed if regenerating). |
| `~/models/gemma4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf` | ~20 GB | The target model. Required for evals after training. |
| `~/models/gemma4-mtp/gemma-4-31B-it-assistant.gguf` | ~700 MB | Existing baseline drafter. Required to compare against the new head. |

### Useful but recoverable

| Path | Notes |
|---|---|
| `~/models/gemma4-2b/gemma-3n-E2B-it-UD-Q4_K_XL.gguf` | Only needed if revisiting the E2B-as-drafter direction (already shown to not help). |
| `/tmp/mtp_alpha/` | Ad-hoc α benchmark scripts. Lost on reboot already; reconstruct from memory entries `mtp-alpha-ceiling` + `mtp-adaptive-k` if needed. |
| `~/.local/bin/amdgpu_top` | Optional ASIC info tool. Easy to reinstall. |

### Claude Code session memory (optional — see below)

`~/.claude/projects/-home-sixvolts-reinstinct/` (~104 MB) — contains the JSONL transcripts of every Claude Code session in this repo AND the auto-memory store under `memory/`. If you want a new Claude Code session on the new machine to inherit the full conversation history + memory:

```bash
# On the new machine, after cloning the repo at the same /home/sixvolts/reinstinct path:
rsync -av sixvolts@oldhost:/home/sixvolts/.claude/projects/-home-sixvolts-reinstinct/ \
                          /home/sixvolts/.claude/projects/-home-sixvolts-reinstinct/
```

The directory name is the absolute repo path with `/` → `-`. If the repo lives at a different path on the new machine, rename the directory accordingly or symlink. Then `claude --resume` in the repo will list past sessions.

Memory entries that encode hard-won decisions worth keeping (rough priority order):

- `project_mtp_adaptive_k.md` — adaptive-K design + A/B numbers
- `project_mtp_alpha_ceiling.md` — why plain MTP is net-negative
- `project_drafter_training_roadmap.md` — the actual roadmap this doc summarizes
- `reference_gemma4_mtp.md` — engine-side MTP wiring (PRE vs POST norm gotcha)
- `feedback_git_identity.md` — `-c user.name=... -c user.email=...` per commit, never write to git config
- `feedback_bench_units.md` — tok/s never ms/tok
- `feedback_alias_safe_kernels.md` — step kernels are called input==output aliased
- `feedback_shared_moe_kernels.md` — `kernels/moe_*.cpp` shared between gemma4.rs and qwen35.rs
- `feedback_gpu_oracle_tests.md` — `set_dp4a(false)` + quantisation-realistic tolerances + top-K
- `feedback_it_model_testing.md` — chat-template `-it` models, raw prompts give fake "garbage"

If you'd rather start fresh on the new machine, this `HANDOFF.md` plus the in-repo `docs/` should be enough to bootstrap context without the JSONL transcripts.

## Push status

`git push origin main` failed on this machine — no GitHub credentials (no `gh` CLI, no SSH key, no credential helper). The new commit `1afa1a9` is local-only. Either:

```bash
# Option A — push from this machine before moving
gh auth login                         # if installing gh is easier than ssh
# or
ssh-keygen -t ed25519 -C 'rigel@sixvolts.org'  # then add the pubkey to github
git remote set-url origin git@github.com:sixvolts/reinstinct.git
git push origin main

# Option B — push from the new machine after cloning local objects
# (e.g. transfer the .git via rsync, then push from the new box)
```

## How to resume on the new machine

```bash
# 1. Clone + fetch the local-only commit
git clone https://github.com/sixvolts/reinstinct.git ~/reinstinct
cd ~/reinstinct
# if 1afa1a9 isn't on origin yet, also rsync the working tree / .git from the old box

# 2. Build (needs ROCm 6.x with gfx906 OR gfx908 target — adjust hipcc target in build.rs if MI100-only)
cargo build --release

# 3. Bring over the trace dataset + models (see "Files that must travel" above)
rsync -av sixvolts@oldhost:/home/sixvolts/datasets/traces/ /home/sixvolts/datasets/traces/
rsync -av sixvolts@oldhost:/home/sixvolts/models/ /home/sixvolts/models/

# 4. (Optional) bring over Claude Code session history + memory — see that section above
```

For the EAGLE training step itself: there's no PyTorch training script in-repo yet. The minimum is a tiny loop that mmaps `gemma4_31b_eagle_train.bin`, iterates per-prompt records, and runs cross-entropy of a small 4-layer transformer (mirror of `gemma-4-31B-it-assistant.gguf` shape) against the recorded `label_tok` at each step, conditioned on `hidden_b`. Quantize the trained PyTorch weights back into the GGUF layout the engine already loads.

## Open / parked work (from memory; not blocking the drafter retrain)

- `project_tracing_followup.md` — switch 92 `eprintln!` sites to `tracing` for production log control.
- `project_superquant_status.md` — Phase 2b/3 of tiered KV cache (Phase 1+2a shipped).
- `project_qwen_int8_kv_attempt.md` — abandoned; documented for "don't try this again."
- `project_qwen_mtp.md` — shelved; GDN-heavy arch defeats speculative batching on MI50.

None of these need attention before resuming on the new box.
