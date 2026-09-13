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
#include "ggml.h"
#include "ggml-backend.h"
#include <map>

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


// Per-layer residual capture: every "l_out-<il>" tensor (the layer's
// output after the residual adds) of the most recent decode step.
static std::map<int, std::vector<float>> g_layers;
static bool cb_eval(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const char * name = ggml_get_name(t);
    static const char * names_env = getenv("DUMP_NAMES");   // "-0": every tensor of layer 0
    const bool want_names = names_env && strstr(name, names_env) != nullptr;
    const bool want = strncmp(name, "l_out-", 6) == 0 || want_names;
    if (ask) return want;
    if (!want) return true;
    if (want_names && strncmp(name, "l_out-", 6) != 0) {
        if (t->type == GGML_TYPE_F32) {
            const int64_t n = ggml_nelements(t);
            std::vector<float> v(n); ggml_backend_tensor_get(t, v.data(), 0, n * sizeof(float));
            const int64_t ne0 = t->ne[0];
            double ss = 0; for (int64_t i = n - ne0; i < n; i++) ss += (double)v[i] * v[i];
            fprintf(stderr, "T %-24s ne=[%lld,%lld,%lld] last-row |x|=%.5g first=%.5g %.5g %.5g\n",
                    name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    std::sqrt(ss), v[n-ne0], v[n-ne0+1], v[n-ne0+2]);
            // DUMP_VECS=1: the whole (contiguous) tensor — one token per
            // decode step, so it is that token's — as one "op <name>: ..." line.
            if (getenv("DUMP_VECS") && ggml_is_contiguous(t)) {
                printf("op %s:", name);
                for (int64_t i = 0; i < n; i++) printf(" %.6e", v[i]);
                printf("\n");
            }
        }
        return true;
    }
    const int il = atoi(name + 6);
    const int64_t n = ggml_nelements(t);
    std::vector<float> v(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, v.data(), 0, n * sizeof(float));
    } else {
        return true;
    }
    // keep the last row (the token just decoded) — for a 1-token batch it is the whole tensor
    const int64_t ne0 = t->ne[0];
    std::vector<float> last(v.end() - ne0, v.end());
    g_layers[il] = last;
    return true;
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
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }
    llama_set_n_threads(ctx, 8, 8);

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

    // Per-layer vectors of the last decoded token: one line per layer.
    for (auto & kv : g_layers) {
        printf("layer %d:", kv.first);
        for (float x : kv.second) printf(" %.6g", x);
        printf("\n");
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
