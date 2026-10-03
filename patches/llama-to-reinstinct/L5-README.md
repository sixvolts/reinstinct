# L5: MoE expert dedup for few-token gate/up

Fork commit 99e97eafc, `ggml/src/ggml-cuda/repack-gcn.cu` only. Scheme from Rune Prod.

With 2-6 tokens (spec verify, small batches), several tokens often route to the same expert. Without dedup, each (token, slot) pair re-reads that expert's weights. With dedup:

- One wave loads all routed expert ids at once (`moe_dedup_route<NTMAX>`).
- `__ballot` plus `__builtin_amdgcn_readlane` find, for every expert, the first (token, slot) pair that routes to it. That pair owns the expert.
- Only owners run. Each owner reads the expert's weights once and accumulates every token routed to it.

Kernel: `mul_mat_vec_q4k_repacked_glu16_dedup<5, 4|6>` (Q4_K gate/up GLU, K=2560). Results are bit-identical with dedup on vs off: KLD at ub3 and ub6 is equal, confirmed again on the production build 2026-10-03.

Measured on Flash-Next, 4x MI50:
- glu16 91.4 -> 80.8 us per call; few-token MoE time -9%.
- pp6 121.9 -> 126.9 tok/s.
- Server with MTP: +0.9% at 1 user, +1.7% at 2 users.

A dedup of the down projection (Q5_1, K=640) was slower, 96 vs 71 us, so it isn't included.

The gain is smaller than Rune's +8% because the fork's repacked layout already stores decoded scales. Rune's old kernel was bound by re-decoding scales per token. Expect more where the matvec is decode-bound.

Disable at runtime: `GGML_CUDA_NO_MOE_DEDUP=1`.
