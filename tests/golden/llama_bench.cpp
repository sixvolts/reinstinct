// Standalone llama.cpp timing harness — links libllama.so directly
// (the llama-cli / llama-bench frontends segfault on this build, but
// the core llama_decode path works, as dump_logits proves).
//
// Measures, on the GPU backend, for a fixed token prompt:
//   - prefill: one batched llama_decode of N prompt tokens
//   - decode:  K sequential single-token llama_decode calls (greedy)
//
// Usage:  llama_bench <model.gguf> <n_prompt> <n_decode> [n_gpu_layers=99]
//
// Output: prefill + decode tokens/sec, directly comparable to
// `reinstinct-engine gpu-bench` / `generate-text`.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <model.gguf> <n_prompt> <n_decode> [n_gpu_layers=99]\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    int n_prompt = atoi(argv[2]);
    int n_decode = atoi(argv[3]);
    int n_gpu_layers = (argc >= 5) ? atoi(argv[4]) : 99;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    int32_t n_ctx = n_prompt + n_decode + 16;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }

    // Deterministic synthetic prompt: tokens 1..n_prompt (mid-vocab,
    // valid ids). Content doesn't matter for a throughput measurement.
    std::vector<llama_token> prompt(n_prompt);
    for (int i = 0; i < n_prompt; ++i) prompt[i] = (llama_token)(1 + i);

    // --- Warmup: one decode so kernels are JIT'd / caches are warm. ---
    {
        llama_token w = 1;
        llama_batch b = llama_batch_get_one(&w, 1);
        llama_decode(ctx, b);
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    // --- Prefill: one batched llama_decode of all n_prompt tokens. ---
    auto t0 = clk::now();
    {
        llama_batch b = llama_batch_get_one(prompt.data(), n_prompt);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "prefill llama_decode failed\n"); return 1;
        }
    }
    auto t1 = clk::now();
    double prefill_s = secs(t0, t1);

    // --- Decode: K greedy single-token steps. ---
    auto pick_argmax = [&](float * lg) {
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        return (llama_token)best;
    };
    auto t2 = clk::now();
    llama_token tok = pick_argmax(llama_get_logits(ctx));
    for (int s = 0; s < n_decode; ++s) {
        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "decode llama_decode failed at step %d\n", s); return 1;
        }
        tok = pick_argmax(llama_get_logits(ctx));
    }
    auto t3 = clk::now();
    double decode_s = secs(t2, t3);

    printf("llama.cpp  model=%s  ngl=%d\n", model_path, n_gpu_layers);
    printf("  prefill: %d tokens in %.3f s  -> %.1f tok/s  (%.3f ms/token)\n",
           n_prompt, prefill_s, n_prompt / prefill_s, 1000.0 * prefill_s / n_prompt);
    printf("  decode:  %d tokens in %.3f s  -> %.1f tok/s  (%.3f ms/token)\n",
           n_decode, decode_s, n_decode / decode_s, 1000.0 * decode_s / n_decode);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
