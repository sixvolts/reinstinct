# S1: benchmarking the llama fork on a reinstinct box

Goal: fork vs reinstinct on the same MI50, same GGUF, same settings. Matches reinstinct's method (256 decode tokens, temperature 0, pp512, 5 runs).

## Build (single or multi GPU)

    git clone -b gfx906-perf https://github.com/sixvolts/llamacpp-gfx906-furnace llama-fork
    cd llama-fork
    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target llama-bench llama-server

For more than one GPU without P2P (PCIe MI50s), add `-DGGML_CUDA_NO_PEER_COPY=ON`.

## Environment

    export GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1
    export HIP_VISIBLE_DEVICES=0          # single-card numbers

## Plain decode and prefill

    ./build/bin/llama-bench -m MODEL.gguf -ngl 999 -fa 1 -p 512 -n 256 -r 5

Report `pp512` and `tg256` (mean and stdev as printed). Run reinstinct's bench back to back on the same card with the same cooling duty cycle.

## Speculative decode with an MTP head

    ./build/bin/llama-server -m MODEL.gguf -ngl 999 -fa on --port 18080 \
        --spec-type draft-mtp --model-draft MTP.gguf --gpu-layers-draft 999 --spec-draft-n-max 2

Then send a fixed prompt with `"temperature": 0, "n_predict": 256` to `/completion` and read `timings.predicted_per_second` plus `timings.draft_n` / `timings.draft_n_accepted`. Use the same prompts on both engines; acceptance varies a lot with content (prose < code < copy-edit).

Acceptance is reported as accepted / drafted tokens with prefix acceptance. Don't compare it against an all-or-nothing rate.

## Where results go

`baselines/S1-<model>.md`: model file, card, engine commit for both, the commands, the numbers.
