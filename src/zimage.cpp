// ============================================================================
// zimage.cpp - See zimage.h. Every step mirrors transformer_z_image.py.
// ============================================================================
#include "zimage.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sd {

namespace {
// Debug aid: with ZIMAGE_TRACE=<prefix> set, intermediate tensors are written
// as <prefix>_trace_<name>.bin/.json for comparison with the PyTorch trace.
void trace(const char* name, const Tensor& t) {
    const char* prefix = std::getenv("ZIMAGE_TRACE");
    if (!prefix) return;
    std::string base = std::string(prefix) + "_trace_" + name;
    std::ofstream bin(base + ".bin", std::ios::binary);
    bin.write((const char*)t.ptr(), (std::streamsize)(t.numel() * sizeof(float)));
    std::ofstream js(base + ".json");
    js << "{\"shape\": [" << t.numel() / t.dim(-1) << ", " << t.dim(-1) << "]}";
}

constexpr int kSeqMultiple = 32;  // SEQ_MULTI_OF: sequences are padded to this
int64_t pad_to(int64_t n) { return (n + kSeqMultiple - 1) / kSeqMultiple * kSeqMultiple; }

// y = x * scale (per row vector [dim]) in place
void scale_rows_(Tensor& x, const float* scale, int64_t dim) {
    const int64_t rows = x.numel() / dim;
    for (int64_t r = 0; r < rows; ++r) {
        float* p = x.ptr() + r * dim;
        for (int64_t i = 0; i < dim; ++i) p[i] *= scale[i];
    }
}
// x += gate (per row vector) * y
void add_gated_(Tensor& x, const Tensor& y, const float* gate, int64_t dim) {
    const int64_t rows = x.numel() / dim;
    for (int64_t r = 0; r < rows; ++r) {
        float* p = x.ptr() + r * dim;
        const float* q = y.ptr() + r * dim;
        for (int64_t i = 0; i < dim; ++i) p[i] += gate[i] * q[i];
    }
}
}  // namespace

ZImageTransformer::Block ZImageTransformer::load_block(const Weights& w, const std::string& p, bool modulated) const {
    Block b{&w.get(p + ".attention_norm1.weight"), &w.get(p + ".attention_norm2.weight"), &w.get(p + ".ffn_norm1.weight"),
            &w.get(p + ".ffn_norm2.weight"), &w.get(p + ".attention.norm_q.weight"), &w.get(p + ".attention.norm_k.weight"),
            Linear(w, p + ".attention.to_q"), Linear(w, p + ".attention.to_k"), Linear(w, p + ".attention.to_v"),
            Linear(w, p + ".attention.to_out.0"), Linear(w, p + ".feed_forward.w1"), Linear(w, p + ".feed_forward.w2"),
            Linear(w, p + ".feed_forward.w3"), Linear()};
    if (modulated) b.adaln = Linear(w, p + ".adaLN_modulation.0");
    return b;
}

ZImageTransformer::ZImageTransformer(const Weights& w) : ZImageTransformer(w, Config()) {}

ZImageTransformer::ZImageTransformer(const Weights& w, const Config& cfg)
    : cfg_(cfg), x_embed_(w, "all_x_embedder.2-1"), cap_embed_(w, "cap_embedder.1"), t_mlp0_(w, "t_embedder.mlp.0"),
      t_mlp2_(w, "t_embedder.mlp.2"), final_linear_(w, "all_final_layer.2-1.linear"),
      final_adaln_(w, "all_final_layer.2-1.adaLN_modulation.1"), cap_norm_(&w.get("cap_embedder.0.weight")),
      x_pad_(&w.get("x_pad_token")), cap_pad_(&w.get("cap_pad_token")) {
    dim_ = (int)x_embed_.W->dim(0);
    head_dim_ = dim_ / cfg_.heads;
    in_channels_ = (int)(x_embed_.W->dim(1) / (cfg_.patch * cfg_.patch));
    if (cfg_.axes_dims[0] + cfg_.axes_dims[1] + cfg_.axes_dims[2] != head_dim_)
        throw std::runtime_error("Z-Image: axes_dims must sum to the head dim");
    for (int i = 0; w.has("noise_refiner." + std::to_string(i) + ".attention_norm1.weight"); ++i)
        noise_refiner_.push_back(load_block(w, "noise_refiner." + std::to_string(i), true));
    for (int i = 0; w.has("context_refiner." + std::to_string(i) + ".attention_norm1.weight"); ++i)
        context_refiner_.push_back(load_block(w, "context_refiner." + std::to_string(i), false));
    for (int i = 0; w.has("layers." + std::to_string(i) + ".attention_norm1.weight"); ++i)
        layers_.push_back(load_block(w, "layers." + std::to_string(i), true));
    if (layers_.empty()) throw std::runtime_error("Z-Image: no transformer layers found");
}

// freqs for axis a: theta^(-2k/d_a), k < d_a/2; angle = position * freq.
// Interleaved layout: the complex pair (2k, 2k+1) of the concatenated axes.
void ZImageTransformer::rope_tables(const std::vector<std::array<int, 3>>& pos, Tensor& cos, Tensor& sin) const {
    const int64_t S = (int64_t)pos.size(), d = head_dim_;
    cos = Tensor({S, d});
    sin = Tensor({S, d});
    for (int64_t s = 0; s < S; ++s) {
        int64_t off = 0;
        for (int a = 0; a < 3; ++a) {
            const int da = cfg_.axes_dims[a];
            for (int k = 0; k < da / 2; ++k) {
                const double freq = 1.0 / std::pow((double)cfg_.rope_theta, (double)(2 * k) / (double)da);
                const float ang = (float)((double)pos[s][a] * freq);
                cos.ptr()[s * d + off + 2 * k] = cos.ptr()[s * d + off + 2 * k + 1] = std::cos(ang);
                sin.ptr()[s * d + off + 2 * k] = sin.ptr()[s * d + off + 2 * k + 1] = std::sin(ang);
            }
            off += da;
        }
    }
}

void ZImageTransformer::run_block(const Block& b, Tensor& x, const Tensor& cos, const Tensor& sin,
                                  const Tensor* adaln) const {
    const int64_t D = dim_;
    std::vector<float> scale_msa(D, 1.0f), gate_msa(D, 1.0f), scale_mlp(D, 1.0f), gate_mlp(D, 1.0f);
    if (b.adaln.valid()) {
        Tensor mod = b.adaln(*adaln);  // [1, 4D]: scale_msa, gate_msa, scale_mlp, gate_mlp
        const float* m = mod.ptr();
        for (int64_t i = 0; i < D; ++i) {
            scale_msa[i] = 1.0f + m[i];
            gate_msa[i] = std::tanh(m[D + i]);
            scale_mlp[i] = 1.0f + m[2 * D + i];
            gate_mlp[i] = std::tanh(m[3 * D + i]);
        }
    }
    // attention
    Tensor h = rms_norm(x, b.attn_norm1, cfg_.norm_eps);
    if (b.adaln.valid()) scale_rows_(h, scale_msa.data(), D);
    Tensor q = b.to_q(h), k = b.to_k(h), v = b.to_v(h);
    rms_norm_heads_(q, head_dim_, b.norm_q, 1e-5f);
    rms_norm_heads_(k, head_dim_, b.norm_k, 1e-5f);
    rope_(q, head_dim_, cos, sin, true);
    rope_(k, head_dim_, cos, sin, true);
    Tensor a = b.to_out(attention(q, k, v, cfg_.heads, false));
    a = rms_norm(a, b.attn_norm2, cfg_.norm_eps);
    if (b.adaln.valid())
        add_gated_(x, a, gate_msa.data(), D);
    else
        add_(x, a);
    // feed-forward
    h = rms_norm(x, b.ffn_norm1, cfg_.norm_eps);
    if (b.adaln.valid()) scale_rows_(h, scale_mlp.data(), D);
    Tensor f = b.w2(swiglu(b.w1(h), b.w3(h)));
    f = rms_norm(f, b.ffn_norm2, cfg_.norm_eps);
    if (b.adaln.valid())
        add_gated_(x, f, gate_mlp.data(), D);
    else
        add_(x, f);
}

Tensor ZImageTransformer::forward(const Tensor& x_in, float t, const Tensor& cap_in) const {
    const int64_t C = x_in.dim(1), H = x_in.dim(2), W = x_in.dim(3), p = cfg_.patch, D = dim_;
    if (x_in.dim(0) != 1 || cap_in.dim(0) != 1) throw std::runtime_error("Z-Image: batch size must be 1");
    if (C != in_channels_ || H % p || W % p) throw std::runtime_error("Z-Image: bad latent shape " + x_in.shape_str());
    const int64_t Ht = H / p, Wt = W / p, n_img = Ht * Wt, n_img_pad = pad_to(n_img);
    const int64_t n_cap = cap_in.dim(1), n_cap_pad = pad_to(n_cap);

    // 1. timestep embedding -> adaln [1, 256]
    Tensor adaln;
    {
        const int64_t F = t_mlp0_.W->dim(1), half = F / 2;
        Tensor freq({1, F});
        const float tt = t * cfg_.t_scale;
        for (int64_t i = 0; i < half; ++i) {
            const float f = std::exp(-std::log(10000.0f) * (float)i / (float)half);
            freq.ptr()[i] = std::cos(tt * f);
            freq.ptr()[half + i] = std::sin(tt * f);
        }
        adaln = t_mlp0_(freq);
        silu_(adaln);
        adaln = t_mlp2_(adaln);
        trace("adaln", adaln);
    }

    // 2. image tokens: patchify (features ordered ph, pw, c), embed, pad
    Tensor patches({n_img, p * p * C});
    for (int64_t h = 0; h < Ht; ++h)
        for (int64_t w = 0; w < Wt; ++w) {
            float* tok = patches.ptr() + (h * Wt + w) * p * p * C;
            for (int64_t ph = 0; ph < p; ++ph)
                for (int64_t pw = 0; pw < p; ++pw)
                    for (int64_t c = 0; c < C; ++c)
                        tok[(ph * p + pw) * C + c] = x_in.ptr()[(c * H + h * p + ph) * W + w * p + pw];
        }
    Tensor img = x_embed_(patches);  // [n_img, D]
    trace("x_embed", img);
    Tensor img_seq({1, n_img_pad, D});
    std::memcpy(img_seq.ptr(), img.ptr(), (size_t)(n_img * D) * sizeof(float));
    for (int64_t i = n_img; i < n_img_pad; ++i)
        for (int64_t j = 0; j < D; ++j) img_seq.ptr()[i * D + j] = x_pad_->at(j);

    // 3. text tokens: RMSNorm, embed, pad
    Tensor cap = cap_embed_(rms_norm(cap_in, cap_norm_, cfg_.norm_eps));  // [1, n_cap, D]
    Tensor cap_seq({1, n_cap_pad, D});
    std::memcpy(cap_seq.ptr(), cap.ptr(), (size_t)(n_cap * D) * sizeof(float));
    for (int64_t i = n_cap; i < n_cap_pad; ++i)
        for (int64_t j = 0; j < D; ++j) cap_seq.ptr()[i * D + j] = cap_pad_->at(j);

    // rotary positions: text (1 + i, 0, 0) for every row including its pad
    // rows (the reference builds the text grid over the padded length);
    // image (n_cap_pad + 1, h, w) with image pad rows at (0, 0, 0).
    std::vector<std::array<int, 3>> cap_pos(n_cap_pad, {0, 0, 0}), img_pos(n_img_pad, {0, 0, 0});
    for (int64_t i = 0; i < n_cap_pad; ++i) cap_pos[i] = {(int)(1 + i), 0, 0};
    for (int64_t h = 0; h < Ht; ++h)
        for (int64_t w = 0; w < Wt; ++w) img_pos[h * Wt + w] = {(int)(n_cap_pad + 1), (int)h, (int)w};
    Tensor img_cos, img_sin, cap_cos, cap_sin;
    rope_tables(img_pos, img_cos, img_sin);
    rope_tables(cap_pos, cap_cos, cap_sin);

    trace("cap_embed", cap);
    // refiners
    for (size_t i = 0; i < noise_refiner_.size(); ++i) {
        run_block(noise_refiner_[i], img_seq, img_cos, img_sin, &adaln);
        if (i == 0) trace("noise_refiner0", img_seq);
    }
    trace("noise_refined", img_seq);
    for (const Block& b : context_refiner_) run_block(b, cap_seq, cap_cos, cap_sin, nullptr);
    trace("cap_refined", cap_seq);

    // 4. unified sequence [image, text] through the main stack
    Tensor uni = concat_batch(img_seq.reshape({n_img_pad, D}), cap_seq.reshape({n_cap_pad, D}));
    uni.reshape({1, n_img_pad + n_cap_pad, D});
    std::vector<std::array<int, 3>> uni_pos = img_pos;
    uni_pos.insert(uni_pos.end(), cap_pos.begin(), cap_pos.end());
    Tensor uni_cos, uni_sin;
    rope_tables(uni_pos, uni_cos, uni_sin);
    for (size_t i = 0; i < layers_.size(); ++i) {
        run_block(layers_[i], uni, uni_cos, uni_sin, &adaln);
        if (i == 0) trace("layer0", uni);
    }
    trace("layers_out", uni);

    // 5. final layer on the image tokens, unpatchify
    Tensor img_out({n_img, D});
    std::memcpy(img_out.ptr(), uni.ptr(), (size_t)(n_img * D) * sizeof(float));
    Tensor c = adaln;
    silu_(c);
    Tensor scale = final_adaln_(c);  // [1, D]
    img_out = layer_norm(img_out, nullptr, nullptr, 1e-6f);
    for (int64_t i = 0; i < D; ++i) scale.ptr()[i] += 1.0f;
    scale_rows_(img_out, scale.ptr(), D);
    Tensor out_tok = final_linear_(img_out);  // [n_img, p*p*C]
    trace("final", out_tok);
    Tensor out({1, C, H, W});
    for (int64_t h = 0; h < Ht; ++h)
        for (int64_t w = 0; w < Wt; ++w) {
            const float* tok = out_tok.ptr() + (h * Wt + w) * p * p * C;
            for (int64_t ph = 0; ph < p; ++ph)
                for (int64_t pw = 0; pw < p; ++pw)
                    for (int64_t cc = 0; cc < C; ++cc)
                        out.ptr()[(cc * H + h * p + ph) * W + w * p + pw] = tok[(ph * p + pw) * C + cc];
        }
    return out;
}

}  // namespace sd
