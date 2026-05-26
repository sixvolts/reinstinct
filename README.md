# reinstinct

A custom HIP inference engine that brings AMD Instinct MI50/MI60 GPUs back from the dead for local AI inference. Outperforms llama.cpp on the same hardware by 20-40%, runs models up to 31B dense on a single $500 card, and delivers throughput competitive with hardware costing significantly more. Reinstinct is built/tuned specifically for two model families: Gemma-4 and Qwen-3.x. Other models might work, need some patches, etc. The goal was to make a few good models work on this hardware well, not account for every model or use case. 

## Why does this exist?

GPUs are expensive. HBM is even harder to get. An NVIDIA RTX 3090 runs $800-1200 used. An M4 Max MacBook Pro starts at $3500. A single H100 rents for $2-3/hr.

Meanwhile, AMD Instinct MI50s are $400-500 on eBay. They have **32 GB of HBM2** and **1 TB/s of memory bandwidth** — the same bandwidth class as an RTX 4090, with 33% more VRAM than a 3090. The reason they are cheap is that AMD declared them end-of-life in 2023 and stopped shipping optimized software. Stock inference frameworks leave 70-90% of the cards bandwidth on the table due to kernel launch overhead and unoptimized dispatch.

reinstinct is a from-scratch inference engine written in Rust + HIP that fixes that. Custom Wave64 kernels, repacked quantization formats, HIP graph capture, fused dequant+matmul, Q8 FlashAttention — all tuned specifically for the gfx906 architecture. No ROCm link-time dependency, no reliance on AMDs deprecated library support. Just libamdhip64.so and raw .hsaco kernel binaries.

## Performance

Single MI50 32 GB, 300W TDP. All models are Unsloth Dynamic GGUF at Q4_K_XL or Q6_K_XL. 
### Decode throughput (tok/s)

| Model | Params | reinstinct | llama.cpp | Delta |
|---|---|---|---|---|
| Qwen 3.5 0.8B | 0.8B | **275.9** | 192.0 | **+44%** |
| Qwen 3.5 4B | 4.2B | **111.2** | 75.9 | **+47%** |
| Gemma 4 E4B | 7.5B | **96.6** | 81.2 | **+19%** |
| Qwen 3.5 35B-A3B MoE | 3.3B active | **101.3** | 78.3 | **+29%** |
| Qwen 3.6 35B-A3B MoE | 3.3B active | **93.5** | 77.1 | **+21%** |
| Gemma 4 26B-A4B MoE | 4B active | **86.5** | 85.5 | **+1%** |
| Gemma 4 31B Dense | 30.7B | **27.5** | 21.0 | **+31%** |
| Qwen 3.5 27B (GDN hybrid) | 26.9B | **28.1** | 23.4 | **+20%** |
| Qwen 3.6 27B-MTP | 26.9B | **28.4** | 23.2 | **+22%** |
| Qwen 3.6 27B | 26.9B | **28.5** | 23.2 | **+23%** |

reinstinct wins **10 of 10** tested configurations.

### How does this compare to other hardware?

| Hardware | Price (used) | VRAM | Qwen 3.5 35B MoE tok/s | Gemma 31B Dense tok/s |
|---|---|---|---|---|
| **MI50 + reinstinct** | **~$500** | 32 GB HBM2 | **101.3** | **27.5** |
| RTX 3090 + llama.cpp | $800-1200 | 24 GB GDDR6X | ~136 | ~21* |
| M4 Max + llama.cpp | $3500+ | 36 GB unified | ~44 | ~20 |
| M4 Max + MLX | $3500+ | 36 GB unified | ~92 | N/A |

*3090 cannot comfortably fit Gemma 31B Q4 (17.5 GB weights + KV exceeds 24 GB at reasonable context lengths).

The MI50 is the price/performance king for local inference on models up to 31B. It is the only ~$500 card with 32 GB of HBM and 1 TB/s bandwidth.

### MTP speculative decoding (Gemma 4 31B, K=3)

| Prompt type | tok/s | Accept rate | vs baseline |
|---|---|---|---|
| Factual ("Capital of France?") | **32.8** | 89% | +19% |
| Structured ("List 5 primes") | **31.8** | 85% | +16% |
| Procedural ("How to make tea") | 25.7 | 63% | -7% |
| Creative ("Write a haiku") | 23.6 | 55% | -14% |

MTP wins on factual/structured prompts, but costs performance on creative workloads. The API endpoint allows per-request MTP toggle so you can decide if you need it for the use cases where it performs well. Area of future improvement for sure. 

## Features

- **Dense + MoE model support**: Gemma 4 (E4B, 26B MoE, 31B), Qwen 3.5 (0.8B-35B), Qwen 3.6 (27B, 35B MoE)
- **Unsloth Dynamic GGUF**: Native support for UD-Q4_K_XL and UD-Q6_K_XL
- **Repacked v2 quantization**: Custom weight layout with denser scale planes for better HBM utilization
- **Q8 KV cache**: INT8 key/value cache with dp4a FlashAttention
- **MTP speculative decoding**: Multi-Token Prediction with per-request control
- **OpenAI-compatible serve endpoint**: /v1/chat/completions with streaming, logprobs, prefix cache
- **HIP graph capture**: Entire decode step as a single GPU submission
- **Fused kernels**: RMSNorm+projection, RoPE+KV write, SwiGLU, dequant+GEMV, attention
- **Wave64-native**: All kernels designed for GCN5.1 64-lane wavefronts with DPP reductions
- **Zero ROCm link dependency**: Runtime dlopen, embedded kernel sources compiled and cached
- **Sliding window attention**: Gemma 4 5:1 sliding/global ratio
- **Gated-DeltaNet**: Qwen 3.5/3.6 hybrid GDN+attention with fused recurrent kernels

## Supported hardware

| GPU | Arch | VRAM | Status |
|---|---|---|---|
| AMD Instinct MI50 | gfx906/Vega20 | 60 CUs | 16 or 32 GB HBM2 | Primary target |
| AMD Instinct MI60 | gfx906/Vega20 | 64 CUs | 32 GB HBM2 | Same ISA, 4 extra CUs |
| AMD Radeon VII | gfx906/Vega20 | 60 CUs | 16 GB HBM2 | Same die, should work (untested) |


## Quick start

```bash
git clone https://github.com/sixvolts/reinstinct.git
cd reinstinct
cargo build --release

# Interactive generation
./target/release/reinstinct-engine generate-text model.gguf \
    --prompt "Hello, world" -n 256 --temperature 0.7 --gpu

# OpenAI-compatible server
./target/release/reinstinct-engine serve --model model.gguf --port 8080

# Benchmark
scripts/bench-all.sh
```

See [MANUAL.md](MANUAL.md) for full documentation.

## Preparing your MI50

These are datacenter pulls. A little prep work goes a long way.

### Replace the Thermal Interface Material (TIM)

Most of these cards originally shipped with a dry graphite pad designed to last the life of the card. You CAN leave the graphite pad, but if you are putting this somewhere where airflow is not perfect, I strongly recommend the upgrade. For sustained workloads on this kind of hardware, a Phase-change Pad is what I would recommend. Thermal Grizzly Phasesheet works great, is inexpensive and is available on Amazon. A single package is all you need for one card. $15-20 depending on the day. PTM7950 works well too, but lots of fake stuff is floating around. 

1. Remove the heatsink shroud - screws along the top/bottom sides of the card. 
2. Scrape off the graphite pad with something soft - like a plastic card. 
3. Clean the die and heastink with Isopropyl alcohol, wipe clean with a lint-free cloth or paper towel. 
4. Apply quality Phase-change pad to the die. 
5. Reassemble. You'll want to run a "burn in", like a benchmark, for a while to help the Phase change material work its way into the the two surfaces. 

Expected improvement: 5-15C drop in junction temperature, preventing thermal throttling during sustained inference.

### Power and clocks

```bash
# Pin clocks for best performance
sudo rocm-smi --setperflevel high

# Set power limit (300W full power, or 250W for quieter operation)
sudo rocm-smi --setpoweroverdrive 300
```

At 250W you lose about 5% throughput but gain significantly better thermals. At 300W the card wants serious airflow.

### Cooling

MI50s are designed for 2U server chassis with high-CFM fans. For use in regular PC or on a bench, 3D print one of the fan adapters listed below and use a high-cfm and pressure fan. A quiet 80mm fan like a noctua will work, but if you are running more than intermittent loads, you'll probably throttle. 

If junction temp exceeds 85-90C during sustained decode (watch with rocm-smi), repaste and improve airflow first.

Fan Shrouds: 
easiest, just add 80mm Fan - https://www.printables.com/model/1479089-amd-mi50-mi100-m210-gpu-80mm-fan-cooling-attachmen
https://www.thingiverse.com/thing:7153218
https://www.thingiverse.com/thing:7314821

Fans:
Best performance: ARCTIC P8 Max
Silent but slower: Noctua NF-A8

## Architecture

reinstinct is built around a few key insights about the MI50:

**Bandwidth-bound, not compute-bound.** At 1 TB/s HBM2 bandwidth, the theoretical decode ceiling for a 4.5 GB model (Q4) is ~222 tok/s. Stock llama.cpp achieves ~10% of this due to kernel launch overhead. HIP graph capture + kernel fusion closes most of that gap.

**Wave64 is an advantage.** 64-lane wavefronts reduce instruction fetch pressure and naturally align with 64-byte cache lines. reinstinct kernels are designed from the ground up for Wave64 with DPP cross-lane reductions.

**Custom quantization layouts matter.** The v2 repacked format converts ragged cache-line-crossing access patterns into fully coalesced sequential reads, yielding 10-15% higher effective HBM bandwidth.

**Q8 attention with dp4a.** INT8 KV cache with v_dot4_i32_i8 dot products halves attention bandwidth and increases throughput vs FP16 attention with relatively little precision loss. 

## Acknowledgments

- The gfx906 community: iacopPBK, arte-fact, nalanzeyu, Kaden-Schutt (hipfire)
- Unsloth for the Dynamic GGUF quantization format and their awesome quants. 
- The llama.cpp project for the GGUF format specification and the foundational work. 
