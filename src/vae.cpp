// ============================================================================
// vae.cpp - AutoencoderKL decoder weight binding and forward pass.
// See vae.h for the block layout.
// ============================================================================
#include "vae.h"

#include <stdexcept>

namespace sd {

namespace {
constexpr float kEps = 1e-6f;  // every GroupNorm in the VAE uses 1e-6
}

VAEDecoder::VAEDecoder(const Weights& w)
    : post_quant_(w, "post_quant_conv"), conv_in_(w, "decoder.conv_in"), conv_out_(w, "decoder.conv_out"),
      norm_out_(w, "decoder.conv_norm_out", kEps) {
    mid_res0_.reset(new ResnetBlock(w, "decoder.mid_block.resnets.0", kEps));
    mid_res1_.reset(new ResnetBlock(w, "decoder.mid_block.resnets.1", kEps));
    mid_attn_.reset(new VAEAttention(w, "decoder.mid_block.attentions.0"));
    // Discover decoder.up_blocks.i and their resnets until one is missing.
    for (int i = 0;; ++i) {
        const std::string bp = "decoder.up_blocks." + std::to_string(i);
        if (!w.has(bp + ".resnets.0.conv1.weight")) break;
        UpBlock blk;
        for (int j = 0; w.has(bp + ".resnets." + std::to_string(j) + ".conv1.weight"); ++j)
            blk.resnets.emplace_back(w, bp + ".resnets." + std::to_string(j), kEps);
        if (w.has(bp + ".upsamplers.0.conv.weight")) blk.upsampler.reset(new Conv2d(w, bp + ".upsamplers.0.conv"));
        up_.push_back(std::move(blk));
    }
    if (up_.empty()) throw std::runtime_error("VAE: no decoder up blocks found");
}

Tensor VAEDecoder::decode(const Tensor& latents) const {
    // Undo the training-time latent scaling.
    Tensor z = latents;
    scale_(z, 1.0f / kScalingFactor);
    Tensor h = conv_in_(post_quant_(z));
    h = mid_res0_->forward(h, nullptr);
    h = mid_attn_->forward(h);
    h = mid_res1_->forward(h, nullptr);
    // Each up block refines features, then doubles the resolution (8x total).
    for (const UpBlock& blk : up_) {
        for (const ResnetBlock& r : blk.resnets) h = r.forward(h, nullptr);
        if (blk.upsampler) h = (*blk.upsampler)(upsample_nearest2x(h));
    }
    h = norm_out_(h);
    silu_(h);
    return conv_out_(h);
}

}  // namespace sd
