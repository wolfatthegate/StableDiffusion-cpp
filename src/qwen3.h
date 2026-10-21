// ============================================================================
// qwen3.h - Qwen3 decoder-only language model used as the Z-Image text
// encoder (Qwen3-4B: 36 layers, 2560 wide, 32 query / 8 key-value heads of
// 128, SwiGLU 9728, RMSNorm, rotary positions with theta 1e6).
//
// Per layer:
//     h = rms_norm(x);  q, k, v = h Wq, h Wk, h Wv
//     q, k = rope(rms_norm_per_head(q)), rope(rms_norm_per_head(k))
//     x = x + attention(q, k, v, causal, grouped) Wo
//     h = rms_norm(x);  x = x + (silu(h Wgate) * h Wup) Wdown
//
// Z-Image conditions on the penultimate hidden state: the output of layer
// index n_layers - 2, with no final norm. forward() exposes that choice.
// Weights are read as "model.layers.N.*" (or "layers.N.*") and kept in BF16.
// ============================================================================
#pragma once

#include <vector>

#include "modules.h"

namespace sd {

class Qwen3Encoder {
public:
    // rope_theta and eps are not stored in the weights; the defaults are
    // Qwen3-4B's config values.
    explicit Qwen3Encoder(const Weights& w, float rope_theta = 1000000.0f, float eps = 1e-6f);

    // ids: token ids (one sequence). Runs `layers` decoder layers (-1 = all)
    // and applies the final norm only if final_norm. Returns [1, S, hidden].
    Tensor forward(const std::vector<int32_t>& ids, int layers = -1, bool final_norm = true) const;

    int num_layers() const { return (int)layers_.size(); }
    int hidden_size() const { return (int)embed_->dim(1); }

private:
    struct Layer {
        const Tensor *ln1, *ln2, *q_norm, *k_norm;
        Linear q, k, v, o, gate, up, down;
    };
    const Tensor* embed_;
    const Tensor* final_norm_;
    std::vector<Layer> layers_;
    int heads_, kv_heads_, head_dim_;
    float theta_, eps_;
};

}  // namespace sd
