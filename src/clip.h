// CLIP ViT-L/14 text transformer (the SD 1.x text encoder).
#pragma once

#include <vector>

#include "modules.h"

namespace sd {

class CLIPTextEncoder {
public:
    explicit CLIPTextEncoder(const Weights& w, int heads = 12);
    // tokens: 77 ids -> last hidden state after the final layer norm: [1, 77, 768]
    Tensor forward(const std::vector<int32_t>& tokens) const;
    int num_layers() const { return (int)layers_.size(); }

private:
    struct Layer {
        LayerNorm ln1, ln2;
        Linear q, k, v, o, fc1, fc2;
    };
    const Tensor* tok_emb_;
    const Tensor* pos_emb_;
    std::vector<Layer> layers_;
    LayerNorm final_ln_;
    int heads_;
};

}  // namespace sd
