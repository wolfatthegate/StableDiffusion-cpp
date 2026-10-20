// ============================================================================
// pipeline.cpp - Model loading and the generation loop. See pipeline.h.
// ============================================================================
#include "pipeline.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

#include "scheduler.h"

namespace sd {

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// <dir>/<sub>/<base>.safetensors, or the .fp16 variant; throws if neither exists.
std::string find_weights(const std::string& dir, const std::string& sub, const std::string& base) {
    const std::string candidates[] = {dir + "/" + sub + "/" + base + ".safetensors",
                                      dir + "/" + sub + "/" + base + ".fp16.safetensors"};
    for (const auto& c : candidates)
        if (file_exists(c)) return c;
    throw std::runtime_error("cannot find " + dir + "/" + sub + "/" + base + "[.fp16].safetensors");
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

Pipeline::Pipeline(const std::string& model_dir)
    : dir_(model_dir), tokenizer_(model_dir + "/tokenizer/vocab.json", model_dir + "/tokenizer/merges.txt") {
    clip_w_.load_safetensors(find_weights(dir_, "text_encoder", "model"));
    clip_.reset(new CLIPTextEncoder(clip_w_));
    unet_w_.load_safetensors(find_weights(dir_, "unet", "diffusion_pytorch_model"));
    unet_.reset(new UNet(unet_w_));
    vae_w_.load_safetensors(find_weights(dir_, "vae", "diffusion_pytorch_model"));
    vae_.reset(new VAEDecoder(vae_w_));
}

GenerateResult Pipeline::generate(const GenerateParams& p, const ProgressFn& progress) const {
    if (p.prompt.empty()) throw std::runtime_error("prompt is empty");
    if (p.steps < 1 || p.steps > 500) throw std::runtime_error("steps must be between 1 and 500");
    if (p.width < 64 || p.height < 64 || p.width % 64 || p.height % 64 || p.width > 2048 || p.height > 2048)
        throw std::runtime_error("width and height must be multiples of 64 between 64 and 2048");
    if (!(p.cfg >= 0.0f && p.cfg <= 50.0f)) throw std::runtime_error("cfg must be between 0 and 50");

    const double t_start = now();
    GenerateResult result;
    result.seed = p.seed >= 0 ? p.seed : (int64_t)(std::random_device{}() & 0x7fffffff);
    auto report = [&](Progress pr) { return !progress || progress(pr); };

    // --- text encoding -----------------------------------------------------
    Progress pr;
    pr.phase = Progress::TextEncoding;
    pr.total = p.steps;
    if (!report(pr)) {
        result.cancelled = true;
        return result;
    }
    const bool use_cfg = p.cfg != 1.0f;
    Tensor context = clip_->forward(tokenizer_.encode(p.prompt));
    if (use_cfg) context = concat_batch(clip_->forward(tokenizer_.encode(p.negative)), context);

    // --- initial noise -------------------------------------------------------
    std::unique_ptr<Scheduler> sched = make_scheduler(p.sampler);
    sched->set_timesteps(p.steps);
    Tensor x({1, 4, p.height / 8, p.width / 8});
    {
        std::mt19937_64 rng((uint64_t)result.seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float& v : x.data) v = normal(rng) * sched->init_noise_sigma();
    }

    // --- denoising loop ------------------------------------------------------
    const auto& ts = sched->timesteps();
    for (int i = 0; i < p.steps; ++i) {
        const double t0 = now();
        Tensor xin = x;
        sched->scale_model_input(xin, i);
        Tensor eps;
        if (use_cfg) {
            Tensor both = unet_->forward(repeat_batch(xin, 2), ts[i], context);
            eps = slice_batch(both, 0);
            const Tensor cond = slice_batch(both, 1);
            for (int64_t j = 0; j < eps.numel(); ++j) eps.ptr()[j] += p.cfg * (cond.ptr()[j] - eps.ptr()[j]);
        } else {
            eps = unet_->forward(xin, ts[i], context);
        }
        Tensor x0 = sched->predict_x0(x, eps, i);
        sched->step(eps, i, x);

        pr.phase = Progress::Denoising;
        pr.step = i + 1;
        pr.timestep = ts[i];
        pr.step_seconds = now() - t0;
        pr.preview = &x0;
        if (!report(pr)) {
            result.cancelled = true;
            result.seconds = now() - t_start;
            return result;
        }
    }

    // --- decode ----------------------------------------------------------------
    pr.phase = Progress::Decoding;
    pr.preview = nullptr;
    if (!report(pr)) {
        result.cancelled = true;
        return result;
    }
    Tensor img = vae_->decode(x);
    result.width = (int)img.dim(3);
    result.height = (int)img.dim(2);
    const int HW = result.width * result.height;
    result.rgb.resize((size_t)HW * 3);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < HW; ++i) {
            float v = (img.ptr()[c * HW + i] * 0.5f + 0.5f) * 255.0f;
            result.rgb[(size_t)i * 3 + c] = (uint8_t)std::lround(std::min(255.0f, std::max(0.0f, v)));
        }
    result.seconds = now() - t_start;
    return result;
}

// Per-channel RGB weights for SD 1.x latents, widely used for cheap previews.
void Pipeline::latent_to_rgb(const Tensor& latent, int& width, int& height, std::vector<uint8_t>& rgb) {
    static const float factors[4][3] = {
        {0.298f, 0.207f, 0.208f},
        {0.187f, 0.286f, 0.173f},
        {-0.158f, 0.189f, 0.264f},
        {-0.184f, -0.271f, -0.473f},
    };
    height = (int)latent.dim(2);
    width = (int)latent.dim(3);
    const int HW = width * height;
    rgb.assign((size_t)HW * 3, 0);
    for (int i = 0; i < HW; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = 0;
            for (int l = 0; l < 4; ++l) v += latent.ptr()[l * HW + i] * factors[l][c];
            v = (v + 1.0f) * 0.5f * 255.0f;
            rgb[(size_t)i * 3 + c] = (uint8_t)std::lround(std::min(255.0f, std::max(0.0f, v)));
        }
}

}  // namespace sd
