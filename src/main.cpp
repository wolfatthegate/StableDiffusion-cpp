// sd: text-to-image with Stable Diffusion 1.x, CPU only, no dependencies.
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "clip.h"
#include "image.h"
#include "safetensors.h"
#include "scheduler.h"
#include "tokenizer.h"
#include "unet.h"
#include "vae.h"

using namespace sd;

namespace {

struct Options {
    std::string model_dir = "models/sd15";
    std::string prompt;
    std::string negative;
    std::string output = "output.png";
    std::string sampler = "euler";
    int steps = 20;
    float cfg = 7.5f;
    int width = 512, height = 512;
    uint64_t seed = 42;
    int threads = 0;
};

void usage() {
    std::fprintf(stderr,
                 "usage: sd --prompt TEXT [options]\n"
                 "  --model-dir DIR       diffusers-layout model directory (default models/sd15)\n"
                 "  --negative TEXT       negative prompt\n"
                 "  --steps N             sampling steps (default 20)\n"
                 "  --cfg F               classifier-free guidance scale (default 7.5; 1 disables)\n"
                 "  --width N --height N  image size, multiples of 64 (default 512x512)\n"
                 "  --seed N              RNG seed (default 42)\n"
                 "  --sampler NAME        euler | ddim (default euler)\n"
                 "  --threads N           worker threads (default: all cores)\n"
                 "  --output FILE         PNG path (default output.png)\n");
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", what);
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--prompt" || a == "-p") o.prompt = need("--prompt");
        else if (a == "--negative" || a == "--negative-prompt" || a == "-n") o.negative = need("--negative");
        else if (a == "--model-dir" || a == "-m") o.model_dir = need("--model-dir");
        else if (a == "--steps") o.steps = std::atoi(need("--steps").c_str());
        else if (a == "--cfg") o.cfg = (float)std::atof(need("--cfg").c_str());
        else if (a == "--width" || a == "-W") o.width = std::atoi(need("--width").c_str());
        else if (a == "--height" || a == "-H") o.height = std::atoi(need("--height").c_str());
        else if (a == "--seed" || a == "-s") o.seed = std::strtoull(need("--seed").c_str(), nullptr, 10);
        else if (a == "--sampler") o.sampler = need("--sampler");
        else if (a == "--threads" || a == "-t") o.threads = std::atoi(need("--threads").c_str());
        else if (a == "--output" || a == "-o") o.output = need("--output");
        else if (a == "--help" || a == "-h") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            std::exit(2);
        }
    }
    if (o.prompt.empty()) {
        usage();
        std::exit(2);
    }
    if (o.width % 8 || o.height % 8 || o.width < 64 || o.height < 64) {
        std::fprintf(stderr, "width/height must be multiples of 8 and at least 64\n");
        std::exit(2);
    }
    return o;
}

bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// Prefers the full-precision file, falls back to the fp16 variant.
std::string find_weights(const std::string& dir, const std::string& sub, const std::string& base) {
    const std::string candidates[] = {dir + "/" + sub + "/" + base + ".safetensors",
                                      dir + "/" + sub + "/" + base + ".fp16.safetensors"};
    for (const auto& c : candidates)
        if (file_exists(c)) return c;
    std::fprintf(stderr, "cannot find %s/%s/%s[.fp16].safetensors\n", dir.c_str(), sub.c_str(), base.c_str());
    std::exit(1);
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) try {
    Options opt = parse_args(argc, argv);
    if (opt.threads > 0) set_num_threads(opt.threads);
    std::fprintf(stderr, "threads: %d\n", num_threads());

    // --- text encoding -----------------------------------------------------
    double t0 = now();
    CLIPTokenizer tokenizer(opt.model_dir + "/tokenizer/vocab.json", opt.model_dir + "/tokenizer/merges.txt");
    std::vector<int32_t> pos_ids = tokenizer.encode(opt.prompt);
    std::vector<int32_t> neg_ids = tokenizer.encode(opt.negative);
    const bool use_cfg = opt.cfg != 1.0f;

    Tensor context;
    {
        Weights w;
        w.load_safetensors(find_weights(opt.model_dir, "text_encoder", "model"));
        CLIPTextEncoder clip(w);
        Tensor cond = clip.forward(pos_ids);
        if (use_cfg) {
            Tensor uncond = clip.forward(neg_ids);
            context = concat_batch(uncond, cond);  // [uncond, cond]
        } else {
            context = std::move(cond);
        }
        std::fprintf(stderr, "text encoder: %d layers, %.1fs\n", clip.num_layers(), now() - t0);
    }

    // --- initial latents -----------------------------------------------------
    std::unique_ptr<Scheduler> sched = make_scheduler(opt.sampler);
    sched->set_timesteps(opt.steps);
    const int64_t lh = opt.height / 8, lw = opt.width / 8;
    Tensor x({1, 4, lh, lw});
    {
        std::mt19937_64 rng(opt.seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float& v : x.data) v = normal(rng) * sched->init_noise_sigma();
    }

    // --- denoising loop ------------------------------------------------------
    {
        t0 = now();
        Weights w;
        w.load_safetensors(find_weights(opt.model_dir, "unet", "diffusion_pytorch_model"));
        UNet unet(w);
        std::fprintf(stderr, "unet: %.2fM params loaded in %.1fs\n", w.total_params() / 1e6, now() - t0);

        const auto& ts = sched->timesteps();
        for (int i = 0; i < opt.steps; ++i) {
            double ts0 = now();
            Tensor xin = x;
            sched->scale_model_input(xin, i);
            Tensor eps;
            if (use_cfg) {
                Tensor both = unet.forward(repeat_batch(xin, 2), ts[i], context);
                eps = slice_batch(both, 0);  // unconditional
                const Tensor cond = slice_batch(both, 1);
                for (int64_t j = 0; j < eps.numel(); ++j)
                    eps.ptr()[j] += opt.cfg * (cond.ptr()[j] - eps.ptr()[j]);
            } else {
                eps = unet.forward(xin, ts[i], context);
            }
            sched->step(eps, i, x);
            std::fprintf(stderr, "step %2d/%d  t=%7.2f  %.1fs\n", i + 1, opt.steps, ts[i], now() - ts0);
        }
    }

    // --- decode ----------------------------------------------------------------
    {
        t0 = now();
        Weights w;
        w.load_safetensors(find_weights(opt.model_dir, "vae", "diffusion_pytorch_model"));
        VAEDecoder vae(w);
        Tensor img = vae.decode(x);  // [1,3,H,W] in [-1,1]
        const int H = (int)img.dim(2), W = (int)img.dim(3);
        std::vector<uint8_t> rgb((size_t)H * W * 3);
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < H * W; ++i) {
                float v = (img.ptr()[c * H * W + i] * 0.5f + 0.5f) * 255.0f;
                rgb[(size_t)i * 3 + c] = (uint8_t)std::lround(std::min(255.0f, std::max(0.0f, v)));
            }
        write_png(opt.output, W, H, rgb);
        std::fprintf(stderr, "vae decode + save: %.1fs -> %s\n", now() - t0, opt.output.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
