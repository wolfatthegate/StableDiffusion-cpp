// Decodes a reference latent with the C++ VAE decoder and compares with the
// PyTorch output (finetune/scripts/dump_reference.py vae).
// Usage: ./test_vae [model-dir] [dump-prefix] [scaling_factor] [shift_factor]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "safetensors.h"
#include "vae.h"

static bool load_dump(const std::string& prefix, std::vector<int64_t>& shape, std::vector<float>& data, sd::Json* meta = nullptr) {
    std::ifstream js(prefix + ".json");
    if (!js) return false;
    std::stringstream ss;
    ss << js.rdbuf();
    sd::Json j = sd::parse_json(ss.str());
    for (const sd::Json& d : j.at("shape").arr) shape.push_back((int64_t)d.num);
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    data.resize((size_t)n);
    std::ifstream bin(prefix + ".bin", std::ios::binary);
    bin.read((char*)data.data(), (std::streamsize)(n * sizeof(float)));
    if (meta) *meta = j;
    return (bool)bin;
}

int main(int argc, char** argv) {
    using namespace std::chrono;
    std::string dir = argc > 1 ? argv[1] : "models/z-image-turbo";
    std::string dump = argc > 2 ? argv[2] : "tests/fixtures/ref_vae";
    std::vector<int64_t> in_shape, out_shape;
    std::vector<float> in, ref;
    sd::Json meta;
    if (!load_dump(dump + "_in", in_shape, in, &meta) || !load_dump(dump, out_shape, ref)) {
        std::printf("no reference dump at %s; run finetune/scripts/dump_reference.py vae\n", dump.c_str());
        return 2;
    }
    const float scaling = argc > 3 ? (float)std::atof(argv[3]) : (float)meta.at("scaling_factor").num;
    const float shift = argc > 4 ? (float)std::atof(argv[4]) : (float)meta.at("shift_factor").num;

    sd::Weights w;
    w.load_model(dir + "/vae", "diffusion_pytorch_model");
    sd::VAEDecoder vae(w, scaling, shift);
    sd::Tensor z;
    z.shape = in_shape;
    z.data = in;
    auto t0 = steady_clock::now();
    sd::Tensor img = vae.decode(z);
    std::printf("decoded %s -> %s in %.2fs (latent channels %d)\n", z.shape_str().c_str(), img.shape_str().c_str(),
                duration<double>(steady_clock::now() - t0).count(), vae.latent_channels());
    if (img.numel() != (int64_t)ref.size()) {
        std::printf("shape mismatch: %lld vs %zu\n", (long long)img.numel(), ref.size());
        return 1;
    }
    double max_err = 0, sum_err2 = 0, sum_ref2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = std::fabs(img.ptr()[i] - ref[i]);
        max_err = std::max(max_err, e);
        sum_err2 += e * e;
        sum_ref2 += (double)ref[i] * ref[i];
    }
    double rel = std::sqrt(sum_err2 / sum_ref2);
    std::printf("max|err| = %.4g  relative rms error = %.3e  %s\n", max_err, rel, rel < 1e-3 ? "ok" : "FAIL");
    return rel < 1e-3 ? 0 : 1;
}
