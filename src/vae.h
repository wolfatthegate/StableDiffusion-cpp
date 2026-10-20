// AutoencoderKL decoder: latent [N,4,h,w] -> image [N,3,8h,8w] in [-1, 1].
#pragma once

#include <memory>
#include <vector>

#include "modules.h"

namespace sd {

class VAEDecoder {
public:
    explicit VAEDecoder(const Weights& w);
    // latents are the raw diffusion latents; scaling by 1/0.18215 happens here.
    Tensor decode(const Tensor& latents) const;
    static constexpr float kScalingFactor = 0.18215f;

private:
    struct UpBlock {
        std::vector<ResnetBlock> resnets;
        std::unique_ptr<Conv2d> upsampler;
    };
    Conv2d post_quant_, conv_in_, conv_out_;
    GroupNorm norm_out_;
    std::unique_ptr<ResnetBlock> mid_res0_, mid_res1_;
    std::unique_ptr<VAEAttention> mid_attn_;
    std::vector<UpBlock> up_;
};

}  // namespace sd
