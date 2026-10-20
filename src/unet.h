// ============================================================================
// unet.h - UNet2DConditionModel as used by Stable Diffusion 1.x.
//
// The UNet predicts the noise eps contained in a latent x_t at timestep t,
// conditioned on the text embeddings. SD 1.5 layout (latent 64x64 for a
// 512x512 image):
//
//   conv_in 4->320                                              64x64
//   down 0: 2 x (Resnet + Transformer) 320,  downsample          64 -> 32
//   down 1: 2 x (Resnet + Transformer) 640,  downsample          32 -> 16
//   down 2: 2 x (Resnet + Transformer) 1280, downsample          16 -> 8
//   down 3: 2 x  Resnet                1280                       8
//   mid:    Resnet, Transformer, Resnet 1280                      8
//   up 0:   3 x  Resnet                1280, upsample             8 -> 16
//   up 1:   3 x (Resnet + Transformer) 1280, upsample            16 -> 32
//   up 2:   3 x (Resnet + Transformer) 640,  upsample            32 -> 64
//   up 3:   3 x (Resnet + Transformer) 320                       64
//   GroupNorm, SiLU, conv_out 320->4
//
// Every output of conv_in and of each down-path resnet/downsampler is pushed
// onto a skip stack (12 entries); every up-path resnet pops one and
// concatenates it to its input along channels.
//
// All of the above is discovered from the weight names, so nothing about
// block counts or widths is hard-coded.
// ============================================================================
#pragma once

#include <memory>
#include <vector>

#include "modules.h"

namespace sd {

class UNet {
public:
    // Binds all weights. heads: attention heads per transformer block
    // (8 for every SD 1.x block). Throws if the weights are not a UNet.
    explicit UNet(const Weights& w, int heads = 8);
    // x:       [N, 4, H/8, W/8] latents, already scaled by the scheduler
    // t:       timestep in [0, 999] (may be fractional for Euler)
    // context: [N, 77, 768] text embeddings, one per batch element
    // Returns the predicted noise, same shape as x.
    Tensor forward(const Tensor& x, float t, const Tensor& context) const;

private:
    // CrossAttnDownBlock2D (attns non-empty) or DownBlock2D (attns empty).
    // attns[j], when present, follows resnets[j].
    struct DownBlock {
        std::vector<ResnetBlock> resnets;
        std::vector<Transformer2D> attns;
        std::unique_ptr<Conv2d> downsampler;  // stride-2 3x3 conv; absent in the last block
    };
    // CrossAttnUpBlock2D or UpBlock2D.
    struct UpBlock {
        std::vector<ResnetBlock> resnets;
        std::vector<Transformer2D> attns;
        std::unique_ptr<Conv2d> upsampler;  // nearest 2x + 3x3 conv; absent in the last block
    };
    Conv2d conv_in_, conv_out_;
    GroupNorm norm_out_;
    Linear time1_, time2_;  // time MLP: 320 -> 1280 -> 1280
    std::vector<DownBlock> down_;
    std::vector<UpBlock> up_;
    std::unique_ptr<ResnetBlock> mid_res0_, mid_res1_;
    std::unique_ptr<Transformer2D> mid_attn_;
    int time_dim_;  // width of the sinusoidal embedding (320)

    // Sinusoidal timestep embedding passed through the time MLP, with the
    // SiLU that every resnet applies before its time_emb_proj folded in.
    // Returns [N, 1280] (the same row repeated for each batch element).
    Tensor time_embedding(float t, int64_t N) const;
};

}  // namespace sd
