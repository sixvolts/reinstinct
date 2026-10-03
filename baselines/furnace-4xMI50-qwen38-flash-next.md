# llama fork baseline: Qwen3.8-Flash-Next on 4x MI50 (furnace)

Hardware: 4x MI50 32 GB, PCIe Gen3, no P2P (cross-GPU copies host-staged), dual Xeon.
Model: Swift1.5-Qwen3.8-Flash-Next UD-Q4_K_XL (177B total MoE, 111 GB), MTP head Q8_0 on GPU0.
Build: fork `qwen4exp-mtp` af566d8a1 (production until 2026-10-03), env `GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1`.
Server config: 6 slots x 64K over a 256K unified KV pool, `-b 16384 -ub 1024`, `-fa on`, layer split 0.20/0.267/0.267/0.266, MTP depth 2, drafting only while <= 2 slots generate.

| metric | value |
|---|---|
| prefill, 8K prompt (server) | 1251 tok/s |
| prefill, 32K prompt (server) | 1561 tok/s |
| prefill, 32K prompt with ~210K resident in other slots | 1547 tok/s |
| decode, llama-bench tg64, no speculation | 49.2 tok/s |
| decode, single stream with MTP (server) | 52-66 tok/s by content |
| decode, 6 concurrent users, aggregate | ~98 tok/s |
| verify shapes, llama-bench pp3 / pp6 | ~90 / ~122 tok/s |

Physics context: decode moves 6.57 GB per token; at the 775 GB/s practical HBM rate the floor is ~118 tok/s single stream, so we sit at 36-42% of it. Gaps are documented in the fork session's profile report.

Not comparable to reinstinct's single-card numbers: different model and a 4-card PCIe split. Like-for-like work is ledger item S1.
