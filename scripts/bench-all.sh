#!/usr/bin/env bash
# Benchmark all locally-available targets: decode tok/s, prefill ms/tok,
# and (where a drafter exists) mtp-gen tok/s across a small K sweep.
# Emits a markdown summary to stdout.
#
# Usage:
#   scripts/bench-all.sh                  # full run
#   scripts/bench-all.sh quick            # skip prefill + K-sweep
#   MODELS_DIR=/other/path scripts/bench-all.sh
#
# Required env: GGUFs live under $MODELS_DIR (default ~/models).

set -uo pipefail
shopt -s nullglob

MODE="${1:-full}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
BIN="${BIN:-$(dirname "$0")/../target/release/reinstinct-engine}"

if [[ ! -x "$BIN" ]]; then
    echo "engine binary not found at $BIN — run 'cargo build --release' first" >&2
    exit 1
fi

# Brief cool-down between models. Without this the late-sequence models
# (positions 8+) get systematically depressed numbers (e.g. qwen-3.6-27B
# at script-position 9 measured 16.7 tok/s vs 22.6 standalone — 30%
# throttle). For most accurate numbers also pin clocks high once before
# the run: `sudo rocm-smi --setperflevel high`.
COOLDOWN_S="${COOLDOWN_S:-10}"

# Decode bench: greedy generate N tokens, parse the "decode  = X tok/s" line.
# Returns "<rate>" or "ERR" on failure / model missing.
bench_decode() {
    local gguf="$1" prompt="$2" steps="$3"
    if [[ ! -f "$gguf" ]]; then echo "ABSENT"; return; fi
    local out
    out=$("$BIN" generate-text "$gguf" --prompt "$prompt" -n "$steps" --gpu 2>&1)
    local rate
    rate=$(grep -oE 'decode\s*=\s*[0-9.]+ ms/token \(([0-9.]+) tok/s\)' <<<"$out" \
           | grep -oE '\([0-9.]+ ' | tr -d '( ' | head -1)
    if [[ -z "$rate" ]]; then echo "ERR"; else echo "$rate"; fi
}

# Prefill bench: REINSTINCT_PREFILL prints "batched prefill = X ms (P tokens, Y ms/token)".
bench_prefill_msptok() {
    local gguf="$1" prompt="$2"
    if [[ ! -f "$gguf" ]]; then echo "ABSENT"; return; fi
    local out
    out=$(REINSTINCT_PREFILL=1 "$BIN" generate-text "$gguf" --prompt "$prompt" \
          -n 1 --gpu 2>&1)
    grep -oE '[0-9.]+ ms/token' <<<"$out" | head -1 | awk '{print $1}'
}

# mtp-gen end-to-end on a fixed prompt at one K. Returns "<tok/s>:<accept%>".
bench_mtp() {
    local target="$1" drafter="$2" k="$3" prompt="$4" steps="$5"
    if [[ ! -f "$target" || ! -f "$drafter" ]]; then echo "ABSENT:-"; return; fi
    local out tok_s accept
    out=$("$BIN" mtp-gen "$target" "$drafter" \
          --system "" --prompt "$prompt" --k "$k" --steps "$steps" 2>&1)
    tok_s=$(grep -oE '= [0-9.]+ tok/s' <<<"$out" | head -1 | awk '{print $2}')
    accept=$(grep -oE 'draft accept rate: [0-9]+ / [0-9]+ = [0-9]+%' <<<"$out" \
             | awk -F'= ' '{print $2}')
    if [[ -z "$tok_s" ]]; then echo "ERR:-"; else echo "${tok_s}:${accept}"; fi
}

# --- Model inventory: (label, gguf-path) pairs ---
TARGETS=(
    "gemma-31B            $MODELS_DIR/gemma4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf"
    "gemma-26B-A4B-MoE    $MODELS_DIR/gemma4-26B/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf"
    "gemma-E4B-PLE        $MODELS_DIR/gemma4-E4B/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
    "qwen-3.5-0.8B        $MODELS_DIR/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
    "qwen-3.5-4B          $MODELS_DIR/qwen-3.5-4B/Qwen3.5-4B-UD-Q4_K_XL.gguf"
    "qwen-3.5-4B-MTP      $MODELS_DIR/qwen-3.5-4B-MTP/Qwen3.5-4B-UD-Q4_K_XL.gguf"
    "qwen-3.5-27B         $MODELS_DIR/qwen-3.5-27B/Qwen3.5-27B-UD-Q4_K_XL.gguf"
    "qwen-3.5-35B-A3B-MoE $MODELS_DIR/qwen-3.5-35B/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    "qwen-3.6-27B         $MODELS_DIR/qwen-3.6-27B/Qwen3.6-27B-UD-Q4_K_XL.gguf"
    "qwen-3.6-27B-MTP     $MODELS_DIR/qwen-3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf"
    "qwen-3.6-35B-A3B-MoE $MODELS_DIR/qwen-3.6-35B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
)

PREFILL_PROMPT="The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. How vexingly quick daft zebras jump. Sphinx of black quartz, judge my vow. Pack my box with five dozen liquor jugs. Two driven jocks help fax my big quiz. The job requires extra pluck and zeal from every young wage earner. We promptly judged antique ivory buckles for the next prize."
DECODE_PROMPT="Hello world"
DECODE_STEPS=16

echo "# reinstinct bench-all  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo
echo "Hardware: $(rocm-smi --showproductname 2>/dev/null | grep -E 'Card series|Card model' | head -1 | sed 's/.*: //' || echo 'unknown')"
echo

# --- Decode + prefill table ---
if [[ "$MODE" == "quick" ]]; then
    printf "| %-22s | %10s |\n" "Model" "Decode tok/s"
    printf "| %-22s | %10s |\n" "----------------------" "----------:"
    first=1
    for entry in "${TARGETS[@]}"; do
        label="${entry%% *}"
        gguf="${entry##* }"
        if [[ $first -eq 0 ]]; then sleep "$COOLDOWN_S"; fi
        d=$(bench_decode "$gguf" "$DECODE_PROMPT" "$DECODE_STEPS")
        printf "| %-22s | %10s |\n" "$label" "$d"
        first=0
    done
else
    printf "| %-22s | %10s | %14s |\n" "Model" "Decode tok/s" "Prefill ms/tok"
    printf "| %-22s | %10s | %14s |\n" "----------------------" "----------:" "-------------:"
    first=1
    for entry in "${TARGETS[@]}"; do
        label="${entry%% *}"
        gguf="${entry##* }"
        if [[ $first -eq 0 ]]; then sleep "$COOLDOWN_S"; fi
        d=$(bench_decode "$gguf" "$DECODE_PROMPT" "$DECODE_STEPS")
        p=$(bench_prefill_msptok "$gguf" "$PREFILL_PROMPT")
        printf "| %-22s | %10s | %14s |\n" "$label" "$d" "${p:-ERR}"
        first=0
    done
fi

echo
echo "## mtp-gen on Gemma 31B (the one model where MTP wins on MI50)"
echo
GEMMA31="$MODELS_DIR/gemma4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf"
GEMMA31_DRAFTER="$MODELS_DIR/gemma4-mtp/gemma-4-31B-it-assistant.Q8_0.gguf"
if [[ "$MODE" != "quick" && -f "$GEMMA31" && -f "$GEMMA31_DRAFTER" ]]; then
    printf "| %-30s | %3s | %10s | %10s |\n" "Prompt" "K" "tok/s" "Accept"
    printf "| %-30s | %3s | %10s | %10s |\n" "------------------------------" "--:" "--------:" "-------:"
    for prompt in "What is 17 times 23?" \
                  "List 5 prime numbers" \
                  "Capital of France?" \
                  "Explain how to make tea" \
                  "Write a haiku about programming"; do
        for k in 1 2 3 4; do
            r=$(bench_mtp "$GEMMA31" "$GEMMA31_DRAFTER" "$k" "$prompt" 48)
            tok_s="${r%:*}"
            accept="${r#*:}"
            printf "| %-30s | %3s | %10s | %10s |\n" \
                   "$(echo "$prompt" | cut -c1-30)" "$k" "$tok_s" "$accept"
        done
    done
else
    echo "(skipped — quick mode, or files missing)"
fi

echo
echo "Done."
