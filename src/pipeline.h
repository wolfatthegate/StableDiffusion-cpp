// ============================================================================
// pipeline.h - The complete text-to-image pipeline with the models kept in
// memory, so one process can serve many generations.
//
//     Pipeline pipe("models/sd15");            // loads tokenizer, CLIP, UNet, VAE
//     GenerateParams p; p.prompt = "a cat";
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
#include "safetensors.h"
#include "tensor.h"
#include "tokenizer.h"
#include "unet.h"
#include "vae.h"

namespace sd {

struct GenerateParams {
    std::string prompt;
    std::string negative;
    std::string sampler = "euler";  // "euler" or "ddim"
    int steps = 20;
    float cfg = 7.5f;
    int width = 512, height = 512;  // multiples of 64
    int64_t seed = 42;              // negative = pick a random seed
};

// Passed to the progress callback. `preview`, when non-null, is the predicted
// clean latent [1, 4, H/8, W/8] after the current step; see latent_to_rgb().
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
    // Loads everything from a diffusers-layout directory (see README).
    // Throws std::runtime_error if any file is missing or malformed.
    explicit Pipeline(const std::string& model_dir);

    // Runs text encoding, the sampling loop and VAE decoding. Validates the
    // parameters and throws std::runtime_error on bad values.
    GenerateResult generate(const GenerateParams& params, const ProgressFn& progress = nullptr) const;

    const std::string& model_dir() const { return dir_; }
    size_t unet_params() const { return unet_w_.total_params(); }

    // Cheap approximation of what a latent will decode to: a fixed 4x3 linear
    // map per latent pixel, giving an RGB image at 1/8 of the final size.
    //   latent: [1, 4, h, w]  ->  rgb: w * h * 3 bytes
    static void latent_to_rgb(const Tensor& latent, int& width, int& height, std::vector<uint8_t>& rgb);

private:
    std::string dir_;
    CLIPTokenizer tokenizer_;
    Weights clip_w_, unet_w_, vae_w_;
    std::unique_ptr<CLIPTextEncoder> clip_;
    std::unique_ptr<UNet> unet_;
    std::unique_ptr<VAEDecoder> vae_;
};

}  // namespace sd
