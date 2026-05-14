// Tiny standalone tool: load a GGUF, decode 1+ tokens, print logits at last position.
//
// Usage:  dump_logits <model.gguf> <token_csv> [k]
//   token_csv is either a single id ("248046") or comma-separated ("1,2,3,4").
//
// Output (stdout): JSON object with input_tokens, vocab_size, top-K, summary stats.
//
// Built ad-hoc (see tests/golden/build.sh); links against
// /home/sixvolts/llama.cpp/build/bin/libllama.so.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int32_t> parse_token_csv(const char * s) {
    std::vector<int32_t> out;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out.push_back((int32_t)v);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <model.gguf> <token_csv> [top_k=64]\n", argv[0]);
        fprintf(stderr, "  token_csv: single id or comma-separated, e.g. \"248046\" or \"1,2,3\"\n");
        return 1;
    }
    const char * model_path = argv[1];
    std::vector<int32_t> tokens = parse_token_csv(argv[2]);
    if (tokens.empty()) {
        fprintf(stderr, "no tokens parsed from %s\n", argv[2]);
        return 1;
    }
    int top_k = (argc >= 4) ? atoi(argv[3]) : 64;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    // CPU-only to match the Rust oracle (no GPU offload).
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    for (int32_t t : tokens) {
        if (t < 0 || t >= n_vocab) {
            fprintf(stderr, "token_id %d out of range [0, %d)\n", t, n_vocab);
            return 1;
        }
    }

    int32_t n_tok = (int32_t)tokens.size();
    int32_t n_ctx = std::max(n_tok + 8, 32);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }
    llama_set_n_threads(ctx, 1, 1);

    // Decode tokens one at a time so the per-step KV/GDN cache evolves
    // exactly the same way our Rust forward_token loop does. (Batched decode
    // would also work but exercises the chunked GDN path which has subtly
    // different fp behavior than the autoregressive one.)
    int rc = 0;
    for (int32_t i = 0; i < n_tok; ++i) {
        llama_token tok = (llama_token)tokens[i];
        llama_batch batch = llama_batch_get_one(&tok, 1);
        rc = llama_decode(ctx, batch);
        if (rc != 0) { fprintf(stderr, "llama_decode failed at step %d: %d\n", i, rc); return 1; }
    }

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
    printf("  \"input_tokens\": [");
    for (int32_t i = 0; i < n_tok; ++i) {
        printf("%d%s", tokens[i], (i + 1 < n_tok) ? ", " : "");
    }
    printf("],\n");
    printf("  \"input_token\": %d,\n", tokens.back()); // backwards-compat: last token
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
