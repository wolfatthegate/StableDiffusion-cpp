// ============================================================================
// clip.cpp - CLIP text encoder forward pass. See clip.h for the architecture.
// ============================================================================
#include "clip.h"

#include <stdexcept>

namespace sd {

namespace {
// Checkpoints saved from CLIPTextModel use "text_model.*"; some conversions
// drop the prefix. Returns whichever one this file uses.
std::string prefix_for(const Weights& w) {
    if (w.has("text_model.embeddings.token_embedding.weight")) return "text_model.";
    if (w.has("embeddings.token_embedding.weight")) return "";
    throw std::runtime_error("CLIP: token embedding not found in text encoder weights");
}
}  // namespace

CLIPTextEncoder::CLIPTextEncoder(const Weights& w, int heads) : heads_(heads) {
    const std::string p = prefix_for(w);
    tok_emb_ = &w.get(p + "embeddings.token_embedding.weight");
    pos_emb_ = &w.get(p + "embeddings.position_embedding.weight");
    // Bind encoder.layers.0, .1, ... until one is missing (12 for ViT-L/14).
    for (int i = 0;; ++i) {
        const std::string lp = p + "encoder.layers." + std::to_string(i);
        if (!w.has(lp + ".layer_norm1.weight")) break;
        Layer L{LayerNorm(w, lp + ".layer_norm1"), LayerNorm(w, lp + ".layer_norm2"),
                Linear(w, lp + ".self_attn.q_proj"), Linear(w, lp + ".self_attn.k_proj"),
                Linear(w, lp + ".self_attn.v_proj"), Linear(w, lp + ".self_attn.out_proj"),
                Linear(w, lp + ".mlp.fc1"), Linear(w, lp + ".mlp.fc2")};
        layers_.push_back(L);
    }
    final_ln_ = LayerNorm(w, p + "final_layer_norm");
}

Tensor CLIPTextEncoder::forward(const std::vector<int32_t>& tokens) const {
    const int64_t S = (int64_t)tokens.size(), D = tok_emb_->dim(1);
    if (S > pos_emb_->dim(0)) throw std::runtime_error("CLIP: sequence longer than position table");
    // Embedding: token vector + learned absolute position vector.
    Tensor x({1, S, D});
    for (int64_t s = 0; s < S; ++s) {
        const int32_t t = tokens[s];
        if (t < 0 || t >= tok_emb_->dim(0)) throw std::runtime_error("CLIP: token id out of range");
        const float* te = tok_emb_->ptr() + (int64_t)t * D;
        const float* pe = pos_emb_->ptr() + s * D;
        float* o = x.ptr() + s * D;
        for (int64_t i = 0; i < D; ++i) o[i] = te[i] + pe[i];
    }
    // Pre-norm transformer layers. The attention is causal: each token sees
    // only itself and earlier tokens, as CLIP was trained.
    for (const Layer& L : layers_) {
        Tensor h = L.ln1(x);
        Tensor a = attention(L.q(h), L.k(h), L.v(h), heads_, /*causal=*/true);
        add_(x, L.o(a));
        h = L.ln2(x);
        h = L.fc1(h);
        quick_gelu_(h);
        add_(x, L.fc2(h));
    }
    return final_ln_(x);
}

}  // namespace sd
