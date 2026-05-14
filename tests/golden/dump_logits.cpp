// Tiny standalone tool: load a GGUF, decode a single token, print logits.
//
// Usage:  dump_logits <model.gguf> <token_id> [k]
// Output to stdout: JSON object {input, vocab, top: [{idx, logit}, ...]}
//                    + a CSV "idx,logit\n" line per all logits to stderr (optional, see below)
//
// Built ad-hoc (see tests/golden/build.sh); links against
// /home/sixvolts/llama.cpp/build/bin/libllama.so.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <model.gguf> <token_id> [top_k=64]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    int token_id_in = atoi(argv[2]);
    int top_k = (argc >= 4) ? atoi(argv[3]) : 64;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    // CPU-only to match the Rust oracle (no GPU offload).
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (token_id_in < 0 || token_id_in >= n_vocab) {
        fprintf(stderr, "token_id %d out of range [0, %d)\n", token_id_in, n_vocab);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16;
    cp.n_batch = 16;
    cp.n_ubatch = 16;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }
    llama_set_n_threads(ctx, 1, 1);

    llama_token tok = (llama_token)token_id_in;
    llama_batch batch = llama_batch_get_one(&tok, 1);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { fprintf(stderr, "llama_decode failed: %d\n", rc); return 1; }

    float * logits = llama_get_logits(ctx);
    if (!logits) { fprintf(stderr, "llama_get_logits returned null\n"); return 1; }

    // Build top-K by scanning all logits.
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });

    // Compute summary stats over all logits.
    float mn = logits[0], mx = logits[0];
    double sum = 0.0, sum_sq = 0.0;
    int nan = 0;
    for (int i = 0; i < n_vocab; ++i) {
        float v = logits[i];
        if (!std::isfinite(v)) { nan++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sum_sq += (double)v * (double)v;
    }
    double mean = sum / n_vocab;
    double std = std::sqrt(sum_sq / n_vocab - mean * mean);

    // Print JSON to stdout.
    printf("{\n");
    printf("  \"model\": \"%s\",\n", model_path);
    printf("  \"input_token\": %d,\n", token_id_in);
    printf("  \"vocab_size\": %d,\n", n_vocab);
    printf("  \"top_k\": %d,\n", top_k);
    printf("  \"stats\": {\n");
    printf("    \"min\": %.6f,\n", mn);
    printf("    \"max\": %.6f,\n", mx);
    printf("    \"mean\": %.6f,\n", mean);
    printf("    \"std\": %.6f,\n", std);
    printf("    \"nonfinite\": %d\n", nan);
    printf("  },\n");
    printf("  \"top\": [\n");
    for (int i = 0; i < top_k; ++i) {
        int t = idx[i];
        printf("    {\"idx\": %d, \"logit\": %.6f}%s\n",
               t, logits[t], (i + 1 < top_k) ? "," : "");
    }
    printf("  ]\n");
    printf("}\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
