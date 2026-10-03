# B1 diagnostics: fork draft-mtp slow on Qwen 3.8-27B (podcast GPU 1, fork 905021dba)

## Root cause candidate: every 2-16 token batch is slow on dense K-quant models, MTP or not

`llama-bench -ngl 999 -fa 1 -p 1,2,3,4,8,16,32,64,128 -n 0 -r 2` (repack env on):

| batch | Qwen 3.8-27B Q4_K_XL | Gemma 4 31B Q4_K_XL |
|---|---:|---:|
| pp1 | 24.4 tok/s (41 ms) | 25.1 tok/s (40 ms) |
| pp2 | 8.3 tok/s (240 ms) | 5.5 tok/s (362 ms) |
| pp3 | 12.2 (246 ms) | 8.3 (363 ms) |
| pp4 | 16.3 (246 ms) | 11.0 (364 ms) |
| pp8 | 31.6 (253 ms) | 21.9 (366 ms) |
| pp16 | 52.3 (306 ms) | 42.8 (374 ms) |
| pp32 | 100.8 (318 ms) | |
| pp64 | 178.6 (358 ms) | |
| pp128 | 212.1 (603 ms) | |

A batch of 2 costs ~6-9x a batch of 1 and is flat to ~16-64 tokens: n>1 leaves the decode matvec and lands on a path with a ~240 ms (Qwen) / ~360 ms (Gemma, also not GDN) fixed cost - looks like the repacked Q4_K/Q5_K/Q6_K dense weights falling to a dequant + generic GEMM (or similar) for 1 < n < MMQ threshold. Furnace production is dense Q8_0 + Q4_K/Q5_1 experts, which would not show it. The no-spec server also shows it: an 11-token prompt took 518 ms (47 ms/token).

MTP is just the most visible victim: the depth-2 verify is a 3-token batch (~250 ms) every round.

## Requested items

- (a) no "graph splits / offload / layers to GPU / CPU" lines at verbosity 3; the MTP draft context is created against the target model ("creating MTP draft context against the target model").
- (b) "graph reallocation": 0 occurrences. "graphs reused = 22" for 25 rounds.
- (c) catch-up per round: 3 tokens, decode call 0.2-0.5 ms, done 9.3-9.7 ms (first: 9 tokens, 14.6 ms). Rounds are ~275 ms apart.
- (d) GGML_SCHED_TIME=1 printed no "sched split" lines in this build/verbosity.
- (e) n_predict 64, palindrome prompt: default 9.5 tok/s (39/47 accepted); LLAMA_SPEC_NO_DEFER=1 9.6 tok/s (39/47); --spec-draft-n-max 1 7.4 tok/s (30/32).

## Profiling note

podcast has no rocprof/rocprofv2/rocprofv3 installed (distro ROCm 7.1 packages), so kernel traces need a profiler install on podcast first (owner's call) or a kernel-family timer inside the engines.
