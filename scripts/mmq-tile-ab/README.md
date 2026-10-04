# Standalone MMQ tile A/B (Q4_0 planes)

`ab.hip` times, on one GPU and identical buffers, reinstinct's `mmq_gemm_q4_0_repacked_f32`
(`#include`d from `patches/reinstinct-to-llama/R11/mmq_gemm_q4_0_repacked.cpp`, copy it next to
`ab.hip` as `ri_q4_0.cpp`) against three versions of the fork's `mmq_gemm_nib_repacked<Q4_0, TN=4>`:
the original 36 B `block_q8_1` staging, the 40 B aligned staging (R13b part a), and 40 B staging
plus 144 B padded LDS weight rows (R13b as shipped). Prints us/op, TMAC/s and the nmse between
reinstinct's output and the fork's.

    cp ../../patches/reinstinct-to-llama/R11/mmq_gemm_q4_0_repacked.cpp ri_q4_0.cpp
    hipcc --offload-arch=gfx906 -O3 -std=c++17 ab.hip -o ab
    HIP_VISIBLE_DEVICES=0 ./ab 512     # P (token count) argument, default 512
    ./ab 2048

Shapes: 5376x21504, 21504x5376, 5376x4096 (Gemma 31B QAT), 5120x17408 (Qwen 3.8-27B).
Furnace, MI50 GPU 0, P=512: reinstinct 11.21 / 10.44 / 9.92 / 10.82 TMAC/s; fork 36 B 9.04 / 8.15 /
8.09 / 8.80; 40 B 10.01 / 9.37 / 9.01 / 9.76; 40 B + padded 11.25 / 10.54 / 10.07 / 10.77.
The fork kernel copy inside `ab.hip` is a verbatim transcription of the tile; keep it in sync when
the tile changes (it is a measurement tool, not the source of truth).
