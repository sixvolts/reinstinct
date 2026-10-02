// Mean next-token NLL with llama.cpp — the reference for our
// perplexity diagnostics. Reads token sequences from stdin, one per line
// (comma-separated ids, BOS included), evaluates each from an empty
// context, and prints the token count and mean NLL over all of them.
//
// Usage:  nll <model.gguf> [n_gpu_layers] < sequences.csv
// Built ad-hoc (see build.sh); links libllama.so.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [ngl] < seqs\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 2 ? atoi(argv[2]) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<std::vector<llama_token>> seqs;
    size_t longest = 0;
    for (std::string line; std::getline(std::cin, line);) {
        std::vector<llama_token> s;
        std::stringstream ss(line);
        for (std::string t; std::getline(ss, t, ',');) if (!t.empty()) s.push_back(atoi(t.c_str()));
        if (s.size() > 1) { longest = std::max(longest, s.size()); seqs.push_back(s); }
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = longest + 8; cp.n_batch = longest + 8; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);

    double nll = 0; long n = 0;
    for (auto & s : seqs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init(s.size(), 0, 1);
        for (size_t i = 0; i < s.size(); i++) {
            b.token[i] = s[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = i + 1 < s.size();
        }
        b.n_tokens = s.size();
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (size_t i = 0; i + 1 < s.size(); i++) {
            const float * l = llama_get_logits_ith(ctx, i);
            double mx = l[0];
            for (int v = 1; v < n_vocab; v++) mx = std::max(mx, (double)l[v]);
            double se = 0;
            for (int v = 0; v < n_vocab; v++) se += std::exp(l[v] - mx);
            nll += mx + std::log(se) - l[s[i + 1]]; n++;
        }
        llama_batch_free(b);
    }
    printf("%ld tokens, mean NLL %.4f, ppl %.3f\n", n, nll / n, std::exp(nll / n));
    llama_free(ctx); llama_model_free(model);
    return 0;
}
