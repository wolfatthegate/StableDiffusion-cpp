// UNet2DConditionModel as used by Stable Diffusion 1.x (diffusers weight layout).
#pragma once

#include <memory>
#include <vector>

#include "modules.h"

namespace sd {

class UNet {
public:
    // heads: attention heads per transformer block (8 for SD 1.x)
    explicit UNet(const Weights& w, int heads = 8);
    // x [N,4,H,W], timestep t, context [N,77,768] -> predicted noise [N,4,H,W]
    Tensor forward(const Tensor& x, float t, const Tensor& context) const;

private:
    struct DownBlock {
        std::vector<ResnetBlock> resnets;
        std::vector<Transformer2D> attns;  // empty for plain DownBlock2D
        std::unique_ptr<Conv2d> downsampler;
    };
    struct UpBlock {
        std::vector<ResnetBlock> resnets;
        std::vector<Transformer2D> attns;
        std::unique_ptr<Conv2d> upsampler;
    };
    Conv2d conv_in_, conv_out_;
    GroupNorm norm_out_;
    Linear time1_, time2_;
    std::vector<DownBlock> down_;
    std::vector<UpBlock> up_;
    std::unique_ptr<ResnetBlock> mid_res0_, mid_res1_;
    std::unique_ptr<Transformer2D> mid_attn_;
    int time_dim_;

    Tensor time_embedding(float t, int64_t N) const;
};

}  // namespace sd
