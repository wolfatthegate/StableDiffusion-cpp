// ============================================================================
// clip.h - CLIP ViT-L/14 text transformer, the SD 1.x text encoder.
//
// Architecture (transformers CLIPTextModel):
//     x = token_embedding[ids] + position_embedding[0..76]      [1, 77, 768]
//     repeat 12 times (pre-norm, causal self-attention):
//         x = x + out_proj(causal_attention(q, k, v of layer_norm1(x)))
//         x = x + fc2(quick_gelu(fc1(layer_norm2(x))))          768->3072->768
//     return final_layer_norm(x)
//
// SD 1.x conditions the UNet on this full last hidden state (all 77
// positions), not on the pooled EOS embedding.
// ============================================================================
#pragma once

#include <vector>

#include "modules.h"

namespace sd {

class CLIPTextEncoder {
public:
    // Binds to weights named either "text_model.*" (transformers layout) or
    // with no prefix. The number of layers is discovered from the weights.
    // heads: attention heads per layer (12 for ViT-L/14).
    explicit CLIPTextEncoder(const Weights& w, int heads = 12);
    // tokens: ids from CLIPTokenizer::encode(), normally 77 of them.
    // Returns the last hidden state after the final layer norm: [1, S, 768].
    Tensor forward(const std::vector<int32_t>& tokens) const;
    int num_layers() const { return (int)layers_.size(); }

private:
    // One encoder layer: two layer norms, attention projections, MLP.
    struct Layer {
        LayerNorm ln1, ln2;
        Linear q, k, v, o, fc1, fc2;
    };
    const Tensor* tok_emb_;  // [vocab = 49408, 768]
    const Tensor* pos_emb_;  // [77, 768]
    std::vector<Layer> layers_;
    LayerNorm final_ln_;
    int heads_;
};

}  // namespace sd
