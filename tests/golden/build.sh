#!/usr/bin/env bash
# Build the dump_logits helper against an existing llama.cpp build.
#
# Defaults:
#   LLAMA_CPP_DIR = /home/sixvolts/llama.cpp
#   LIB_DIR       = $LLAMA_CPP_DIR/build/bin
# Override either via env if your llama.cpp lives elsewhere.

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/sixvolts/llama.cpp}"
LIB_DIR="${LIB_DIR:-$LLAMA_CPP_DIR/build/bin}"

cd "$(dirname "$0")"

build_one() {
    g++ -std=c++17 -O2 \
        -I"$LLAMA_CPP_DIR/include" \
        -I"$LLAMA_CPP_DIR/ggml/include" \
        -L"$LIB_DIR" \
        -Wl,-rpath,"$LIB_DIR" \
        "$1.cpp" \
        -lllama -lggml -lggml-base -lggml-cpu \
        -o "$1"
    echo "built $(pwd)/$1"
}

# dump_logits: golden-logit reference. llama_bench: throughput harness.
# Both link libllama.so directly — the llama-cli / llama-bench frontends
# segfault on this build, but the core llama_decode path is sound.
build_one dump_logits
build_one llama_bench
