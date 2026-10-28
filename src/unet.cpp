// ============================================================================
// unet.cpp - UNet2DConditionModel weight binding and forward pass.
// See unet.h for the block layout.
// ============================================================================
#include "unet.h"

#include <cmath>
#include <stdexcept>

namespace sd {

namespace {
constexpr float kResnetEps = 1e-5f;  // GroupNorm eps in UNet resnets
// idx("down_blocks", 2) -> "down_blocks.2"
std::string idx(const std::string& base, int i) { return base + "." + std::to_string(i); }
}  // namespace

UNet::UNet(const Weights& w, int heads)
    : conv_in_(w, "conv_in"), conv_out_(w, "conv_out"), norm_out_(w, "conv_norm_out", kResnetEps),
      time1_(w, "time_embedding.linear_1"), time2_(w, "time_embedding.linear_2") {
    time_dim_ = (int)time1_.W->dim(1);  // input width of the time MLP

    // Discover down_blocks.i, each with resnets.j and optional attentions.j,
    // stopping at the first index that has no weights.
    for (int i = 0; w.has(idx("down_blocks", i) + ".resnets.0.conv1.weight"); ++i) {
        const std::string bp = idx("down_blocks", i);
        DownBlock blk;
        for (int j = 0; w.has(idx(bp + ".resnets", j) + ".conv1.weight"); ++j) {
            blk.resnets.emplace_back(w, idx(bp + ".resnets", j), kResnetEps);
            if (w.has(idx(bp + ".attentions", j) + ".norm.weight"))
                blk.attns.emplace_back(w, idx(bp + ".attentions", j), heads);
        }
        if (w.has(bp + ".downsamplers.0.conv.weight"))
            blk.downsampler.reset(new Conv2d(w, bp + ".downsamplers.0.conv", /*stride=*/2));
        down_.push_back(std::move(blk));
    }

    // UNetMidBlock2DCrossAttn: resnet, transformer, resnet.
    mid_res0_.reset(new ResnetBlock(w, "mid_block.resnets.0", kResnetEps));
    mid_res1_.reset(new ResnetBlock(w, "mid_block.resnets.1", kResnetEps));
    if (w.has("mid_block.attentions.0.norm.weight")) mid_attn_.reset(new Transformer2D(w, "mid_block.attentions.0", heads));

    // Up blocks mirror the down blocks, with one extra resnet each.
    for (int i = 0; w.has(idx("up_blocks", i) + ".resnets.0.conv1.weight"); ++i) {
        const std::string bp = idx("up_blocks", i);
        UpBlock blk;
        for (int j = 0; w.has(idx(bp + ".resnets", j) + ".conv1.weight"); ++j) {
            blk.resnets.emplace_back(w, idx(bp + ".resnets", j), kResnetEps);
            if (w.has(idx(bp + ".attentions", j) + ".norm.weight"))
                blk.attns.emplace_back(w, idx(bp + ".attentions", j), heads);
        }
        if (w.has(bp + ".upsamplers.0.conv.weight")) blk.upsampler.reset(new Conv2d(w, bp + ".upsamplers.0.conv"));
        up_.push_back(std::move(blk));
    }
    if (down_.empty() || up_.empty()) throw std::runtime_error("UNet: no down/up blocks found in weights");
}

// Sinusoidal timestep embedding, diffusers Timesteps(flip_sin_to_cos=True,
// downscale_freq_shift=0):
//     freq_i = 10000^(-i / half),  i = 0 .. half-1
//     emb    = [cos(t * freq) | sin(t * freq)]      (cos first: "flipped")
// then TimestepEmbedding: linear_1 -> SiLU -> linear_2.
// Every ResnetBlock applies SiLU before its time_emb_proj, so that final
// SiLU is applied once here instead of in each of the 22 resnets.
Tensor UNet::time_embedding(float t, int64_t N) const {
    const int half = time_dim_ / 2;
    Tensor e({1, (int64_t)time_dim_});
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
        const float arg = t * freq;
        e.ptr()[i] = std::cos(arg);
        e.ptr()[half + i] = std::sin(arg);
    }
    Tensor h = time1_(e.to(time1_.W->device));
    silu_(h);
    h = time2_(h);
    silu_(h);
    return repeat_batch(h, N);
}

Tensor UNet::forward(const Tensor& x, float t, const Tensor& context) const {
    const int64_t N = x.dim(0);
    if (context.dim(0) != N) throw std::runtime_error("UNet: context batch mismatch");
    Tensor temb = time_embedding(t, N);

    // ---- down path: record every intermediate for the skip connections ----
    std::vector<Tensor> skips;
    Tensor h = conv_in_(x);
    skips.push_back(h);
    for (const DownBlock& blk : down_) {
        for (size_t j = 0; j < blk.resnets.size(); ++j) {
            h = blk.resnets[j].forward(h, &temb);
            if (j < blk.attns.size()) h = blk.attns[j].forward(h, context);
            skips.push_back(h);
        }
        if (blk.downsampler) {
            h = (*blk.downsampler)(h);
            skips.push_back(h);
        }
    }

    // ---- bottleneck at 1/8 of the latent resolution ----
    h = mid_res0_->forward(h, &temb);
    if (mid_attn_) h = mid_attn_->forward(h, context);
    h = mid_res1_->forward(h, &temb);

    // ---- up path: each resnet consumes the most recent skip (LIFO) ----
    for (const UpBlock& blk : up_) {
        for (size_t j = 0; j < blk.resnets.size(); ++j) {
            if (skips.empty()) throw std::runtime_error("UNet: skip connection stack underflow");
            Tensor skip = std::move(skips.back());
            skips.pop_back();
            h = blk.resnets[j].forward(concat_channels(h, skip), &temb);
            if (j < blk.attns.size()) h = blk.attns[j].forward(h, context);
        }
        if (blk.upsampler) h = (*blk.upsampler)(upsample_nearest2x(h));
    }

    // ---- output head: back to 4 latent channels ----
    h = norm_out_(h);
    silu_(h);
    return conv_out_(h);
}

}  // namespace sd
