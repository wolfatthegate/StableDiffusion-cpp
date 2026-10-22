// ============================================================================
// pipeline.h - The complete text-to-image pipeline with the models kept in
// memory, so one process can serve many generations. Two model families are
// supported and selected by the model directory's model_index.json:
//
//   StableDiffusionPipeline  CLIP text encoder + UNet + SD VAE (SD 1.x)
//   ZImagePipeline           Qwen3-4B text encoder + S3-DiT + FLUX VAE (Z-Image)
//
//     Pipeline pipe("models/z-image-turbo");  // loads everything
//     GenerateParams p = pipe.defaults();     // family-appropriate steps/cfg/size
//     p.prompt = "a cat";
//     GenerateResult r = pipe.generate(p, [](const Progress& pr) {
//         ... show pr.step / pr.total, pr.preview ...
//         return true;                           // false cancels
//     });
//     write_png("cat.png", r.width, r.height, r.rgb);
//
// generate() is const and does not share mutable state, but it uses every
// core, so callers should run one generation at a time.
// ============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "clip.h"
#include "qwen3.h"
#include "qwen_tokenizer.h"
#include "safetensors.h"
#include "tensor.h"
#include "tokenizer.h"
#include "unet.h"
#include "vae.h"
#include "zimage.h"

namespace sd {

struct GenerateParams {
    std::string prompt;
    std::string negative;
    std::string sampler = "euler";  // SD: "euler" | "ddim"; Z-Image: "flow"
    int steps = 20;
    float cfg = 7.5f;               // 0 or 1 disables guidance (Z-Image Turbo needs 0)
    int width = 512, height = 512;  // multiples of 64
    int64_t seed = 42;              // negative = pick a random seed
};

// Passed to the progress callback. `preview`, when non-null, is the predicted
// clean latent [1, C, H/8, W/8] after the current step; see latent_to_rgb().
struct Progress {
    enum Phase { TextEncoding, Denoising, Decoding };
    Phase phase = TextEncoding;
    int step = 0, total = 0;    // 1-based step just completed, and the step count
    float timestep = 0;         // scheduler timestep of that step
    double step_seconds = 0;    // wall time of that step
    const Tensor* preview = nullptr;
};
// Return false to cancel the generation after the current step.
using ProgressFn = std::function<bool(const Progress&)>;

struct GenerateResult {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;  // width * height * 3, empty if cancelled
    int64_t seed = 0;          // the seed actually used
    double seconds = 0;        // total wall time
    bool cancelled = false;
};

class Pipeline {
public:
    enum class Family { SD1, ZImage };

    // Loads everything from a diffusers-layout directory (see README).
    // Throws std::runtime_error if any file is missing or malformed.
    explicit Pipeline(const std::string& model_dir);

    // Runs text encoding, the sampling loop and VAE decoding. Validates the
    // parameters and throws std::runtime_error on bad values.
    GenerateResult generate(const GenerateParams& params, const ProgressFn& progress = nullptr) const;

    Family family() const { return family_; }
    const char* family_name() const { return family_ == Family::ZImage ? "z-image" : "sd1"; }
    // Sensible defaults for this family: SD 1.x 512x512, 20 Euler steps, cfg
    // 7.5; Z-Image 1024x1024, 9 flow steps, cfg 0 (no negative prompt pass).
    GenerateParams defaults() const;
    std::vector<std::string> samplers() const;
    const std::string& model_dir() const { return dir_; }
    size_t generator_params() const { return family_ == Family::ZImage ? dit_w_.total_params() : unet_w_.total_params(); }
    size_t unet_params() const { return generator_params(); }

    // Cheap approximation of what a latent will decode to: a fixed linear
    // map per latent pixel (4 or 16 channels), giving an RGB image at 1/8 of
    // the final size.
    static void latent_to_rgb(const Tensor& latent, int& width, int& height, std::vector<uint8_t>& rgb);

private:
    std::string dir_;
    Family family_ = Family::SD1;
    // SD 1.x
    std::unique_ptr<CLIPTokenizer> tokenizer_;
    Weights clip_w_, unet_w_;
    std::unique_ptr<CLIPTextEncoder> clip_;
    std::unique_ptr<UNet> unet_;
    // Z-Image
    std::unique_ptr<QwenTokenizer> qwen_tok_;
    Weights qwen_w_, dit_w_;
    std::unique_ptr<Qwen3Encoder> qwen_;
    std::unique_ptr<ZImageTransformer> dit_;
    float flow_shift_ = 3.0f;
    int max_text_tokens_ = 512;
    // shared
    Weights vae_w_;
    std::unique_ptr<VAEDecoder> vae_;

    GenerateResult generate_sd1(const GenerateParams& p, const ProgressFn& progress) const;
    GenerateResult generate_zimage(const GenerateParams& p, const ProgressFn& progress) const;
};

}  // namespace sd
