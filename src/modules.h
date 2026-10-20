// Building blocks shared by the UNet and the VAE, wired to weights by diffusers tensor names.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "safetensors.h"
#include "tensor.h"

namespace sd {

struct Linear {
    const Tensor* W = nullptr;
    const Tensor* b = nullptr;
    Linear() = default;
    Linear(const Weights& w, const std::string& prefix);
    Tensor operator()(const Tensor& x) const { return linear(x, *W, b); }
    bool valid() const { return W != nullptr; }
};

struct Conv2d {
    const Tensor* W = nullptr;
    const Tensor* b = nullptr;
    int stride = 1, pad = 0;
    Conv2d() = default;
    Conv2d(const Weights& w, const std::string& prefix, int stride = 1);  // pad = (k - 1) / 2
    Tensor operator()(const Tensor& x) const { return conv2d(x, *W, b, stride, pad); }
    bool valid() const { return W != nullptr; }
};

struct GroupNorm {
    const Tensor* g = nullptr;
    const Tensor* b = nullptr;
    int groups = 32;
    float eps = 1e-5f;
    GroupNorm() = default;
    GroupNorm(const Weights& w, const std::string& prefix, float eps, int groups = 32);
    Tensor operator()(const Tensor& x) const { return group_norm(x, groups, *g, *b, eps); }
};

struct LayerNorm {
    const Tensor* g = nullptr;
    const Tensor* b = nullptr;
    float eps = 1e-5f;
    LayerNorm() = default;
    LayerNorm(const Weights& w, const std::string& prefix, float eps = 1e-5f);
    Tensor operator()(const Tensor& x) const { return layer_norm(x, *g, *b, eps); }
};

// ResnetBlock2D: norm-silu-conv, (+ time embedding), norm-silu-conv, + shortcut(x)
struct ResnetBlock {
    GroupNorm norm1, norm2;
    Conv2d conv1, conv2;
    Linear temb_proj;    // optional (UNet only)
    Conv2d shortcut;     // optional 1x1 when channel counts differ
    ResnetBlock(const Weights& w, const std::string& prefix, float eps);
    // temb: already SiLU-activated [N, temb_dim], or nullptr
    Tensor forward(const Tensor& x, const Tensor* temb) const;
};

// Multi-head attention with separate q / k / v / out projections (diffusers Attention).
struct CrossAttention {
    Linear to_q, to_k, to_v, to_out;
    int heads = 8;
    CrossAttention(const Weights& w, const std::string& prefix, int heads);
    // x [N,S,C]; context [N,Skv,D] (use x for self-attention)
    Tensor forward(const Tensor& x, const Tensor& context) const;
};

struct FeedForward {  // GEGLU -> Linear
    Linear proj, out;
    FeedForward(const Weights& w, const std::string& prefix);
    Tensor forward(const Tensor& x) const;
};

struct BasicTransformerBlock {
    LayerNorm norm1, norm2, norm3;
    CrossAttention attn1, attn2;
    FeedForward ff;
    BasicTransformerBlock(const Weights& w, const std::string& prefix, int heads);
    Tensor forward(const Tensor& x, const Tensor& context) const;
};

// Transformer2DModel: GroupNorm -> proj_in -> blocks -> proj_out, residual
struct Transformer2D {
    GroupNorm norm;
    Conv2d proj_in_conv, proj_out_conv;
    Linear proj_in_lin, proj_out_lin;
    bool linear_proj = false;
    std::vector<BasicTransformerBlock> blocks;
    Transformer2D(const Weights& w, const std::string& prefix, int heads);
    Tensor forward(const Tensor& x, const Tensor& context) const;
};

// Single-head spatial self-attention used in the VAE mid block.
struct VAEAttention {
    GroupNorm norm;
    Linear q, k, v, out;
    VAEAttention(const Weights& w, const std::string& prefix);
    Tensor forward(const Tensor& x) const;
};

}  // namespace sd
