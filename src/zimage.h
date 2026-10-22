// ============================================================================
// zimage.h - Z-Image single-stream diffusion transformer (S3-DiT), ported from
// diffusers' ZImageTransformer2DModel in "basic" (text-to-image) mode.
//
// Flow for one denoising step at timestep t in [0, 1]:
//   1. adaln = t_embedder(t * 1000): 256-d sinusoid -> Linear -> SiLU -> Linear
//   2. image: latent [16, H, W] -> 2x2 patches -> 64-d tokens -> Linear to 3840,
//      padded to a multiple of 32 with the learned x_pad_token, then 2
//      "noise refiner" blocks (modulated)
//   3. text: Qwen3 hidden states [L, 2560] -> RMSNorm -> Linear to 3840,
//      padded to a multiple of 32 with cap_pad_token, then 2 "context
//      refiner" blocks (not modulated)
//   4. unified = [image tokens, text tokens]; 30 modulated blocks of joint
//      self-attention (30 heads of 128, RMS QK-norm, 3-axis rotary positions)
//      and SwiGLU (10240 wide)
//   5. final layer: LayerNorm * (1 + Linear(SiLU(adaln))) -> Linear to 64,
//      unpatchify the first H/2 * W/2 tokens back to [16, H, W]
//
// Rotary positions are triples (axis0, h, w) with 32 + 48 + 48 = 128 rotary
// dims: text row i (real or pad) gets (1 + i, 0, 0), image tokens
// (L_text_padded + 1, h, w), image pad tokens (0, 0, 0). Padding is not
// masked in attention, so the pad tokens are part of the computation and
// must be reproduced exactly.
//
// Weights are kept in BF16. Tensor names follow the diffusers checkpoint:
// layers.N.*, noise_refiner.N.*, context_refiner.N.*, all_x_embedder.2-1,
// all_final_layer.2-1, cap_embedder, t_embedder, x_pad_token, cap_pad_token.
// ============================================================================
#pragma once

#include <array>
#include <vector>

#include "modules.h"

namespace sd {

class ZImageTransformer {
public:
    // Hyper-parameters not derivable from the weights come from
    // transformer/config.json; the defaults are Z-Image's.
    struct Config {
        int patch = 2;
        int heads = 30;
        float norm_eps = 1e-5f;
        float rope_theta = 256.0f;
        float t_scale = 1000.0f;
        int axes_dims[3] = {32, 48, 48};
    };
    explicit ZImageTransformer(const Weights& w);
    ZImageTransformer(const Weights& w, const Config& cfg);

    // x: [1, 16, H, W] latent (H, W even); t in [0, 1] (the pipeline passes
    // 1 - sigma); cap: [1, L, 2560] text hidden states.
    // Returns the model output, same shape as x.
    Tensor forward(const Tensor& x, float t, const Tensor& cap) const;

    int in_channels() const { return in_channels_; }

private:
    struct Block {
        const Tensor *attn_norm1, *attn_norm2, *ffn_norm1, *ffn_norm2, *norm_q, *norm_k;
        Linear to_q, to_k, to_v, to_out, w1, w2, w3;
        Linear adaln;  // optional (modulated blocks)
    };
    Block load_block(const Weights& w, const std::string& prefix, bool modulated) const;
    // One block on x [1, S, dim] with rotary tables cos/sin [S, head_dim].
    void run_block(const Block& b, Tensor& x, const Tensor& cos, const Tensor& sin, const Tensor* adaln) const;
    // Rotary tables (interleaved layout) for a list of (axis0, h, w) positions.
    void rope_tables(const std::vector<std::array<int, 3>>& pos, Tensor& cos, Tensor& sin) const;

    Config cfg_;
    int dim_, head_dim_, in_channels_;
    Linear x_embed_, cap_embed_, t_mlp0_, t_mlp2_, final_linear_, final_adaln_;
    const Tensor *cap_norm_, *x_pad_, *cap_pad_;
    std::vector<Block> noise_refiner_, context_refiner_, layers_;
};

}  // namespace sd
