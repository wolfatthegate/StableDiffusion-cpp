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

#include "backend_cuda.h"
#include "image.h"
#include "pipeline.h"

using namespace sd;

namespace {

// Command-line options and their defaults.
struct Options {
    std::string model_dir = "models/sd15";
    std::string output = "output.png";
    std::string device = "auto";
    GenerateParams gen;
    // Which fields were given on the command line; the rest take the
    // pipeline's family defaults (SD 1.x or Z-Image).
    bool have_steps = false, have_cfg = false, have_size = false, have_sampler = false;
    int threads = 0;
};

void usage() {
    std::fprintf(stderr,
                 "usage: sd --prompt TEXT [options]\n"
                 "  --model-dir DIR       diffusers-layout model directory (default models/sd15)\n"
                 "  --negative TEXT       negative prompt (needs --cfg > 0)\n"
                 "  --steps N             sampling steps (SD 1.x default 20; Z-Image 9)\n"
                 "  --cfg F               guidance scale (SD 1.x default 7.5; Z-Image Turbo 0 = off)\n"
                 "  --width N --height N  image size, multiples of 64 (SD 1.x 512; Z-Image 1024)\n"
                 "  --seed N              RNG seed (default 42; -1 picks a random seed)\n"
                 "  --sampler NAME        euler | ddim for SD 1.x, flow for Z-Image\n"
                 "  --threads N           worker threads (default: all cores)\n"
                 "  --device NAME         cpu | cuda | auto (default auto: cuda when built with CUDA and a GPU is present)\n"
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
        else if (a == "--steps") { o.gen.steps = std::atoi(need("--steps").c_str()); o.have_steps = true; }
        else if (a == "--cfg") { o.gen.cfg = (float)std::atof(need("--cfg").c_str()); o.have_cfg = true; }
        else if (a == "--width" || a == "-W") { o.gen.width = std::atoi(need("--width").c_str()); o.have_size = true; }
        else if (a == "--height" || a == "-H") { o.gen.height = std::atoi(need("--height").c_str()); o.have_size = true; }
        else if (a == "--seed" || a == "-s") o.gen.seed = std::strtoll(need("--seed").c_str(), nullptr, 10);
        else if (a == "--sampler") { o.gen.sampler = need("--sampler"); o.have_sampler = true; }
        else if (a == "--threads" || a == "-t") o.threads = std::atoi(need("--threads").c_str());
        else if (a == "--device") o.device = need("--device");
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

    const Device device = parse_device(opt.device);
    if (device == Device::CUDA) std::fprintf(stderr, "device: cuda (%s)\n", cuda::device_name());
    double t0 = now();
    Pipeline pipe(opt.model_dir, device);
    std::fprintf(stderr, "%s models loaded in %.1fs (generator: %.0fM params, %s)\n", pipe.family_name(), now() - t0,
                 pipe.generator_params() / 1e6, device_name(pipe.device()));
    GenerateParams d = pipe.defaults();
    if (!opt.have_steps) opt.gen.steps = d.steps;
    if (!opt.have_cfg) opt.gen.cfg = d.cfg;
    if (!opt.have_size) { opt.gen.width = d.width; opt.gen.height = d.height; }
    if (!opt.have_sampler) opt.gen.sampler = d.sampler;
    std::fprintf(stderr, "%dx%d, %d steps, sampler %s, cfg %g\n", opt.gen.width, opt.gen.height, opt.gen.steps,
                 opt.gen.sampler.c_str(), opt.gen.cfg);

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
