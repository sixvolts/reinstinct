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

g++ -std=c++17 -O2 \
    -I"$LLAMA_CPP_DIR/include" \
    -I"$LLAMA_CPP_DIR/ggml/include" \
    -L"$LIB_DIR" \
    -Wl,-rpath,"$LIB_DIR" \
    dump_logits.cpp \
    -lllama -lggml -lggml-base -lggml-cpu \
    -o dump_logits

echo "built $(pwd)/dump_logits"
