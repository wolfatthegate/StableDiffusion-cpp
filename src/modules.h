// ============================================================================
// modules.h - Neural network building blocks shared by the UNet and the VAE.
//
// Each struct mirrors a diffusers / PyTorch module and is constructed from a
// Weights object plus a name prefix. For example
//     Linear(w, "mid_block.attentions.0.proj_in")
// binds to "mid_block.attentions.0.proj_in.weight" and ".bias".
//
// Modules hold const pointers into the Weights; they own no memory, are
// cheap to copy, and are immutable after construction, so forward() is const.
// ============================================================================
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "safetensors.h"
#include "tensor.h"

namespace sd {

// nn.Linear. Binds "<prefix>.weight" [out, in] and optional "<prefix>.bias".
struct Linear {
    const Tensor* W = nullptr;
    const Tensor* b = nullptr;
    Linear() = default;  // unbound; valid() is false
    Linear(const Weights& w, const std::string& prefix);
    Tensor operator()(const Tensor& x) const { return linear(x, *W, b); }
    bool valid() const { return W != nullptr; }
};

// nn.Conv2d with "same" padding: pad = (kernel - 1) / 2, so a stride-1 conv
// keeps the spatial size and a stride-2 conv halves it (all SD convs are
// 1x1 or 3x3).
struct Conv2d {
    const Tensor* W = nullptr;  // [Cout, Cin, k, k]
    const Tensor* b = nullptr;  // [Cout] or nullptr
    int stride = 1, pad = 0;
    Conv2d() = default;  // unbound; valid() is false
    Conv2d(const Weights& w, const std::string& prefix, int stride = 1);
    Tensor operator()(const Tensor& x) const { return conv2d(x, *W, b, stride, pad); }
    bool valid() const { return W != nullptr; }
};

// nn.GroupNorm with affine parameters. SD always uses 32 groups; eps is 1e-5
// in UNet resnets and 1e-6 in the VAE and in transformer input norms.
struct GroupNorm {
    const Tensor* g = nullptr;  // gamma [C]
    const Tensor* b = nullptr;  // beta  [C]
    int groups = 32;
    float eps = 1e-5f;
    GroupNorm() = default;
    GroupNorm(const Weights& w, const std::string& prefix, float eps, int groups = 32);
    Tensor operator()(const Tensor& x) const { return group_norm(x, groups, *g, *b, eps); }
};

// nn.LayerNorm over the last dimension.
struct LayerNorm {
    const Tensor* g = nullptr;  // gamma [C]
    const Tensor* b = nullptr;  // beta  [C]
    float eps = 1e-5f;
    LayerNorm() = default;
    LayerNorm(const Weights& w, const std::string& prefix, float eps = 1e-5f);
    Tensor operator()(const Tensor& x) const { return layer_norm(x, *g, *b, eps); }
};

// diffusers ResnetBlock2D:
//
//     h = conv1(silu(norm1(x)))
//     h = h + time_emb_proj(temb)[:, :, None, None]     (UNet only)
//     h = conv2(silu(norm2(h)))
//     return h + shortcut(x)                             (1x1 conv if Cin != Cout)
//
// time_emb_proj and conv_shortcut are bound only if present in the weights.
// (Dropout between norm2 and conv2 is an identity at inference.)
struct ResnetBlock {
    GroupNorm norm1, norm2;
    Conv2d conv1, conv2;
    Linear temb_proj;  // optional (UNet only)
    Conv2d shortcut;   // optional 1x1 when channel counts differ
    ResnetBlock(const Weights& w, const std::string& prefix, float eps);
    // x: [N, Cin, H, W]. temb: the time embedding with SiLU already applied,
    // [N, temb_dim], or nullptr (the VAE has no time conditioning).
    // Returns [N, Cout, H, W].
    Tensor forward(const Tensor& x, const Tensor* temb) const;
};

// diffusers Attention with separate q / k / v / out projections.
// Used both as self-attention (context = x) and cross-attention
// (context = text embeddings). to_q/to_k/to_v have no bias in SD 1.x;
// to_out.0 does. (to_out.1 is dropout and has no weights.)
struct CrossAttention {
    Linear to_q, to_k, to_v, to_out;
    int heads = 8;
    CrossAttention(const Weights& w, const std::string& prefix, int heads);
    // x: [N, S, C] queries.  context: [N, Skv, D] keys/values source.
    // Returns [N, S, C].
    Tensor forward(const Tensor& x, const Tensor& context) const;
};

// diffusers FeedForward with GEGLU: out(geglu(proj(x))).
// proj expands C -> 8C (two halves of 4C), out projects 4C -> C.
struct FeedForward {
    Linear proj, out;
    FeedForward(const Weights& w, const std::string& prefix);
    Tensor forward(const Tensor& x) const;
};

// diffusers BasicTransformerBlock (pre-norm, residual after each sub-layer):
//
//     x = x + attn1(norm1(x))                 self-attention over pixels
//     x = x + attn2(norm2(x), context)        cross-attention to the prompt
//     x = x + ff(norm3(x))                    GEGLU feed-forward
struct BasicTransformerBlock {
    LayerNorm norm1, norm2, norm3;
    CrossAttention attn1, attn2;
    FeedForward ff;
    BasicTransformerBlock(const Weights& w, const std::string& prefix, int heads);
    // x: [N, H*W, C]  context: [N, 77, 768]  ->  [N, H*W, C]
    Tensor forward(const Tensor& x, const Tensor& context) const;
};

// diffusers Transformer2DModel, the "SpatialTransformer" of the UNet:
//
//     h = proj_in(group_norm(x))        reshaped to a token sequence [N, HW, C]
//     h = transformer_blocks(h, context)
//     return proj_out(h) + x            reshaped back to [N, C, H, W]
//
// SD 1.x checkpoints use 1x1 convolutions for proj_in / proj_out; newer
// conversions use Linear layers. The constructor detects which from the
// weight rank, and the two are mathematically equivalent.
struct Transformer2D {
    GroupNorm norm;
    Conv2d proj_in_conv, proj_out_conv;  // used when linear_proj == false
    Linear proj_in_lin, proj_out_lin;    // used when linear_proj == true
    bool linear_proj = false;
    std::vector<BasicTransformerBlock> blocks;  // usually exactly one
    Transformer2D(const Weights& w, const std::string& prefix, int heads);
    // x: [N, C, H, W]  context: [N, 77, 768]  ->  [N, C, H, W]
    Tensor forward(const Tensor& x, const Tensor& context) const;
};

// Single-head spatial self-attention in the VAE mid block:
//     return x + out(attention(q(h), k(h), v(h)))   where h = group_norm(x)
// Supports both the current diffusers names (to_q, to_k, to_v, to_out.0) and
// the older ones (query, key, value, proj_attn).
struct VAEAttention {
    GroupNorm norm;
    Linear q, k, v, out;
    VAEAttention(const Weights& w, const std::string& prefix);
    // x: [N, C, H, W]  ->  [N, C, H, W]
    Tensor forward(const Tensor& x) const;
};

}  // namespace sd
