// Tokenize stdin with a GGUF's vocab (vocab only, no weights) and print
// the ids comma-separated — the llama.cpp reference for our tokenizers.
//
// Usage:  tokenize <model.gguf> < text
// Built ad-hoc (see build.sh); links libllama.so.

#include "llama.h"

#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> < text\n", argv[0]); return 1; }
    std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(),
                           (int32_t)toks.size(), /*add_special=*/false, /*parse_special=*/false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    for (int i = 0; i < n; i++) printf(i ? ",%d" : "%d", toks[i]);
    printf("\n");
    llama_model_free(model);
    return 0;
}
