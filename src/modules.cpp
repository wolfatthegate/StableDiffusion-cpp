// ============================================================================
// modules.cpp - Weight binding and forward passes for the building blocks in
// modules.h. Each forward() follows the diffusers implementation line by line.
// ============================================================================
#include "modules.h"

#include <stdexcept>

namespace sd {

Linear::Linear(const Weights& w, const std::string& prefix)
    : W(&w.get(prefix + ".weight")), b(w.try_get(prefix + ".bias")) {}

Conv2d::Conv2d(const Weights& w, const std::string& prefix, int stride_)
    : W(&w.get(prefix + ".weight")), b(w.try_get(prefix + ".bias")), stride(stride_) {
    if (W->ndim() != 4) throw std::runtime_error(prefix + ": expected a 4-D conv weight, got " + W->shape_str());
    pad = (int)(W->dim(3) - 1) / 2;
}

GroupNorm::GroupNorm(const Weights& w, const std::string& prefix, float eps_, int groups_)
    : g(&w.get(prefix + ".weight")), b(&w.get(prefix + ".bias")), groups(groups_), eps(eps_) {}

LayerNorm::LayerNorm(const Weights& w, const std::string& prefix, float eps_)
    : g(&w.get(prefix + ".weight")), b(&w.get(prefix + ".bias")), eps(eps_) {}

// ---------------------------------------------------------------------------
ResnetBlock::ResnetBlock(const Weights& w, const std::string& p, float eps)
    : norm1(w, p + ".norm1", eps), norm2(w, p + ".norm2", eps), conv1(w, p + ".conv1"), conv2(w, p + ".conv2") {
    if (w.has(p + ".time_emb_proj.weight")) temb_proj = Linear(w, p + ".time_emb_proj");
    if (w.has(p + ".conv_shortcut.weight")) shortcut = Conv2d(w, p + ".conv_shortcut");
}

Tensor ResnetBlock::forward(const Tensor& x, const Tensor* temb) const {
    Tensor h = norm1(x);
    silu_(h);
    h = conv1(h);
    // Time conditioning: project the embedding to C channels and add the
    // same value to every pixel of that channel.
    if (temb_proj.valid() && temb) {
        Tensor t = temb_proj(*temb);  // [N, C]
        const int64_t N = h.dim(0), C = h.dim(1), HW = h.dim(2) * h.dim(3);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c = 0; c < C; ++c) {
                const float add = t.ptr()[n * C + c];
                float* p = h.ptr() + (n * C + c) * HW;
                for (int64_t i = 0; i < HW; ++i) p[i] += add;
            }
    }
    h = norm2(h);
    silu_(h);
    h = conv2(h);
    // Residual connection; the 1x1 conv matches channel counts when needed.
    if (shortcut.valid()) {
        add_(h, shortcut(x));
    } else {
        add_(h, x);
    }
    return h;
}

// ---------------------------------------------------------------------------
CrossAttention::CrossAttention(const Weights& w, const std::string& p, int heads_)
    : to_q(w, p + ".to_q"), to_k(w, p + ".to_k"), to_v(w, p + ".to_v"), to_out(w, p + ".to_out.0"), heads(heads_) {}

Tensor CrossAttention::forward(const Tensor& x, const Tensor& context) const {
    // Queries come from x; keys and values from the context. In SD 1.x the
    // text context has 768 features, so to_k/to_v map 768 -> C.
    Tensor q = to_q(x), k = to_k(context), v = to_v(context);
    return to_out(attention(q, k, v, heads, false));
}

FeedForward::FeedForward(const Weights& w, const std::string& p) : proj(w, p + ".net.0.proj"), out(w, p + ".net.2") {}

Tensor FeedForward::forward(const Tensor& x) const { return out(geglu(proj(x))); }

BasicTransformerBlock::BasicTransformerBlock(const Weights& w, const std::string& p, int heads)
    : norm1(w, p + ".norm1"), norm2(w, p + ".norm2"), norm3(w, p + ".norm3"),
      attn1(w, p + ".attn1", heads), attn2(w, p + ".attn2", heads), ff(w, p + ".ff") {}

Tensor BasicTransformerBlock::forward(const Tensor& x_in, const Tensor& context) const {
    Tensor x = x_in;
    {  // self-attention: the normalised tokens attend to themselves
        Tensor h = norm1(x);
        add_(x, attn1.forward(h, h));
    }
    add_(x, attn2.forward(norm2(x), context));  // cross-attention to the prompt
    add_(x, ff.forward(norm3(x)));
    return x;
}

// ---------------------------------------------------------------------------
Transformer2D::Transformer2D(const Weights& w, const std::string& p, int heads) : norm(w, p + ".norm", 1e-6f) {
    // A 2-D proj_in weight means Linear; 4-D ([C, C, 1, 1]) means 1x1 conv.
    linear_proj = w.get(p + ".proj_in.weight").ndim() == 2;
    if (linear_proj) {
        proj_in_lin = Linear(w, p + ".proj_in");
        proj_out_lin = Linear(w, p + ".proj_out");
    } else {
        proj_in_conv = Conv2d(w, p + ".proj_in");
        proj_out_conv = Conv2d(w, p + ".proj_out");
    }
    // Bind transformer_blocks.0, .1, ... until one is missing.
    for (int i = 0; w.has(p + ".transformer_blocks." + std::to_string(i) + ".norm1.weight"); ++i)
        blocks.emplace_back(w, p + ".transformer_blocks." + std::to_string(i), heads);
}

Tensor Transformer2D::forward(const Tensor& x, const Tensor& context) const {
    const int64_t H = x.dim(2), W = x.dim(3);
    Tensor h = norm(x);
    // A 1x1 conv on NCHW equals a Linear on [N, HW, C], so the only
    // difference between the two variants is where the reshape happens.
    if (linear_proj) {
        h = proj_in_lin(nchw_to_nlc(h));
    } else {
        h = nchw_to_nlc(proj_in_conv(h));
    }
    for (const auto& blk : blocks) h = blk.forward(h, context);
    if (linear_proj) {
        h = nlc_to_nchw(proj_out_lin(h), H, W);
    } else {
        h = proj_out_conv(nlc_to_nchw(h, H, W));
    }
    add_(h, x);  // residual around the whole transformer
    return h;
}

// ---------------------------------------------------------------------------
VAEAttention::VAEAttention(const Weights& w, const std::string& p) : norm(w, p + ".group_norm", 1e-6f) {
    if (w.has(p + ".to_q.weight")) {
        q = Linear(w, p + ".to_q");
        k = Linear(w, p + ".to_k");
        v = Linear(w, p + ".to_v");
        out = Linear(w, p + ".to_out.0");
    } else {  // older diffusers naming
        q = Linear(w, p + ".query");
        k = Linear(w, p + ".key");
        v = Linear(w, p + ".value");
        out = Linear(w, p + ".proj_attn");
    }
}

Tensor VAEAttention::forward(const Tensor& x) const {
    const int64_t H = x.dim(2), W = x.dim(3);
    // Each pixel is a token; with heads = 1 the full 512 channels form one head.
    Tensor h = nchw_to_nlc(norm(x));
    h = out(attention(q(h), k(h), v(h), 1, false));
    h = nlc_to_nchw(h, H, W);
    add_(h, x);
    return h;
}

}  // namespace sd
