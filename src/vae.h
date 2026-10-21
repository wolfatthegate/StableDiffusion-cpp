// ============================================================================
// vae.h - AutoencoderKL decoder: turns 4-channel latents back into pixels.
//
// Only the decoder half is implemented because text-to-image never encodes.
// SD 1.x layout (latent 64x64 -> image 512x512):
//
//   z / 0.18215 -> post_quant_conv 4->4 -> conv_in 4->512           64x64
//   mid: Resnet, single-head self-attention, Resnet (512)            64x64
//   up 0: 3 x Resnet 512, upsample                                   64 -> 128
//   up 1: 3 x Resnet 512, upsample                                  128 -> 256
//   up 2: 3 x Resnet 256, upsample                                  256 -> 512
//   up 3: 3 x Resnet 128                                            512
//   GroupNorm, SiLU, conv_out 128->3                     RGB in roughly [-1, 1]
// ============================================================================
#pragma once

#include <memory>
#include <vector>

#include "modules.h"

namespace sd {

class VAEDecoder {
public:
    // Binds "decoder.*" weights (and "post_quant_conv.*" when present);
    // block counts and channel widths are discovered from the names.
    // scaling_factor / shift_factor come from the VAE's config.json:
    // SD 1.x uses 0.18215 / 0, the FLUX VAE used by Z-Image 0.3611 / 0.1159.
    explicit VAEDecoder(const Weights& w, float scaling_factor = kSDScalingFactor, float shift_factor = 0.0f);
    // latents: [N, C, h, w], the raw output of the diffusion loop. The
    // un-scaling  z / scaling_factor + shift_factor  happens inside.
    // Returns [N, 3, 8h, 8w]; values are nominally in [-1, 1] and should be
    // clamped by the caller.
    Tensor decode(const Tensor& latents) const;
    static constexpr float kSDScalingFactor = 0.18215f;
    int latent_channels() const { return (int)conv_in_.W->dim(1); }

private:
    struct UpBlock {
        std::vector<ResnetBlock> resnets;
        std::unique_ptr<Conv2d> upsampler;  // absent in the last block
    };
    Conv2d post_quant_, conv_in_, conv_out_;  // post_quant_ optional
    GroupNorm norm_out_;
    std::unique_ptr<ResnetBlock> mid_res0_, mid_res1_;
    std::unique_ptr<VAEAttention> mid_attn_;
    std::vector<UpBlock> up_;
    float scaling_factor_, shift_factor_;
};

}  // namespace sd
