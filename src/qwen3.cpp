// ============================================================================
// qwen3.cpp - See qwen3.h.
// ============================================================================
#include "qwen3.h"

#include <cmath>
#include <stdexcept>

namespace sd {

namespace {
std::string prefix_for(const Weights& w) {
    if (w.has("model.embed_tokens.weight")) return "model.";
    if (w.has("embed_tokens.weight")) return "";
    throw std::runtime_error("Qwen3: embed_tokens.weight not found");
}
}  // namespace

Qwen3Encoder::Qwen3Encoder(const Weights& w, float rope_theta, float eps) : theta_(rope_theta), eps_(eps) {
    const std::string p = prefix_for(w);
    embed_ = &w.get(p + "embed_tokens.weight");
    final_norm_ = &w.get(p + "norm.weight");
    for (int i = 0;; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        if (!w.has(lp + "input_layernorm.weight")) break;
        Layer L{&w.get(lp + "input_layernorm.weight"), &w.get(lp + "post_attention_layernorm.weight"),
                &w.get(lp + "self_attn.q_norm.weight"), &w.get(lp + "self_attn.k_norm.weight"),
                Linear(w, lp + "self_attn.q_proj"), Linear(w, lp + "self_attn.k_proj"),
                Linear(w, lp + "self_attn.v_proj"), Linear(w, lp + "self_attn.o_proj"),
                Linear(w, lp + "mlp.gate_proj"), Linear(w, lp + "mlp.up_proj"), Linear(w, lp + "mlp.down_proj")};
        layers_.push_back(L);
    }
    if (layers_.empty()) throw std::runtime_error("Qwen3: no layers found");
    head_dim_ = (int)layers_[0].q_norm->dim(0);
    heads_ = (int)(layers_[0].q.W->dim(0) / head_dim_);
    kv_heads_ = (int)(layers_[0].k.W->dim(0) / head_dim_);
}

Tensor Qwen3Encoder::forward(const std::vector<int32_t>& ids, int layers, bool final_norm) const {
    const int64_t S = (int64_t)ids.size(), D = embed_->dim(1), d = head_dim_;
    if (layers < 0 || layers > (int)layers_.size()) layers = (int)layers_.size();
    Tensor x = embedding(*embed_, ids);
    x.reshape({1, S, D});

    // Rotary tables in the rotate_half layout: angle i is used at i and i + d/2.
    Tensor cs({S, d}), sn({S, d});
    for (int64_t s = 0; s < S; ++s)
        for (int64_t i = 0; i < d / 2; ++i) {
            const double inv_freq = std::pow((double)theta_, -2.0 * (double)i / (double)d);
            const double a = (double)s * inv_freq;
            cs.ptr()[s * d + i] = cs.ptr()[s * d + i + d / 2] = (float)std::cos(a);
            sn.ptr()[s * d + i] = sn.ptr()[s * d + i + d / 2] = (float)std::sin(a);
        }

    for (int li = 0; li < layers; ++li) {
        const Layer& L = layers_[li];
        Tensor h = rms_norm(x, L.ln1, eps_);
        Tensor q = L.q(h), k = L.k(h), v = L.v(h);
        rms_norm_heads_(q, d, L.q_norm, eps_);
        rms_norm_heads_(k, d, L.k_norm, eps_);
        rope_(q, d, cs, sn, false);
        rope_(k, d, cs, sn, false);
        Tensor a = attention(q, k, v, heads_, /*causal=*/true, kv_heads_);
        add_(x, L.o(a));
        h = rms_norm(x, L.ln2, eps_);
        add_(x, L.down(swiglu(L.gate(h), L.up(h))));
    }
    if (final_norm) x = rms_norm(x, final_norm_, eps_);
    return x;
}

}  // namespace sd
