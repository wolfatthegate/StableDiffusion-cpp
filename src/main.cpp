// ============================================================================
// main.cpp - The `sd` command-line tool: text-to-image with Stable Diffusion
// 1.x on the CPU, no dependencies.
//
// The actual work is in Pipeline (pipeline.h); this file parses arguments,
// prints progress to stderr and writes the PNG.
// ============================================================================
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "image.h"
#include "pipeline.h"

using namespace sd;

namespace {

// Command-line options and their defaults.
struct Options {
    std::string model_dir = "models/sd15";
    std::string output = "output.png";
    GenerateParams gen;
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
                 "  --seed N              RNG seed (default 42; -1 picks a random seed)\n"
                 "  --sampler NAME        euler | ddim (default euler)\n"
                 "  --threads N           worker threads (default: all cores)\n"
                 "  --output FILE         PNG path (default output.png)\n");
}

// Parses argv; prints usage and exits on --help or invalid input.
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
        if (a == "--prompt" || a == "-p") o.gen.prompt = need("--prompt");
        else if (a == "--negative" || a == "--negative-prompt" || a == "-n") o.gen.negative = need("--negative");
        else if (a == "--model-dir" || a == "-m") o.model_dir = need("--model-dir");
        else if (a == "--steps") o.gen.steps = std::atoi(need("--steps").c_str());
        else if (a == "--cfg") o.gen.cfg = (float)std::atof(need("--cfg").c_str());
        else if (a == "--width" || a == "-W") o.gen.width = std::atoi(need("--width").c_str());
        else if (a == "--height" || a == "-H") o.gen.height = std::atoi(need("--height").c_str());
        else if (a == "--seed" || a == "-s") o.gen.seed = std::strtoll(need("--seed").c_str(), nullptr, 10);
        else if (a == "--sampler") o.gen.sampler = need("--sampler");
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
    if (o.gen.prompt.empty()) {
        usage();
        std::exit(2);
    }
    return o;
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

    double t0 = now();
    Pipeline pipe(opt.model_dir);
    std::fprintf(stderr, "models loaded in %.1fs (unet: %.2fM params)\n", now() - t0, pipe.unet_params() / 1e6);

    GenerateResult r = pipe.generate(opt.gen, [&](const Progress& pr) {
        if (pr.phase == Progress::Denoising)
            std::fprintf(stderr, "step %2d/%d  t=%7.2f  %.1fs\n", pr.step, pr.total, pr.timestep, pr.step_seconds);
        else if (pr.phase == Progress::Decoding)
            std::fprintf(stderr, "decoding...\n");
        return true;
    });
    write_png(opt.output, r.width, r.height, r.rgb);
    std::fprintf(stderr, "seed %lld, %.1fs total -> %s\n", (long long)r.seed, r.seconds, opt.output.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
