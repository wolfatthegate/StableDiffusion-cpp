// One Z-Image transformer forward on the reference inputs
// (finetune/scripts/dump_reference.py transformer), compared with PyTorch.
// Usage: ./test_zimage [model-dir] [dump-prefix]
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
#include "zimage.h"

static bool load_dump(const std::string& prefix, sd::Tensor& t, sd::Json* meta = nullptr) {
    std::ifstream js(prefix + ".json");
    if (!js) return false;
    std::stringstream ss;
    ss << js.rdbuf();
    sd::Json j = sd::parse_json(ss.str());
    t.shape.clear();
    for (const sd::Json& d : j.at("shape").arr) t.shape.push_back((int64_t)d.num);
    t.data.resize((size_t)sd::Tensor::numel_of(t.shape));
    std::ifstream bin(prefix + ".bin", std::ios::binary);
    bin.read((char*)t.data.data(), (std::streamsize)(t.data.size() * sizeof(float)));
    if (meta) *meta = j;
    return (bool)bin;
}

int main(int argc, char** argv) {
    using namespace std::chrono;
    std::string dir = argc > 1 ? argv[1] : "models/z-image-turbo";
    std::string dump = argc > 2 ? argv[2] : "tests/fixtures/ref_transformer";
    sd::Tensor x, cap, ref;
    sd::Json meta;
    if (!load_dump(dump + "_x", x, &meta) || !load_dump(dump + "_cap", cap) || !load_dump(dump, ref)) {
        std::printf("no reference dump at %s; run finetune/scripts/dump_reference.py transformer\n", dump.c_str());
        return 2;
    }
    const float t = (float)meta.at("t").num;
    x.shape.insert(x.shape.begin(), 1);    // [1, C, H, W]
    cap.shape.insert(cap.shape.begin(), 1);  // [1, L, 2560]

    auto t0 = steady_clock::now();
    // SD_TEST_DEVICE=cuda runs the same comparison on the GPU.
    const char* dev_env = std::getenv("SD_TEST_DEVICE");
    const sd::Device dev = sd::parse_device(dev_env ? dev_env : "cpu");
    sd::Weights w;
    w.load_model(dir + "/transformer", "diffusion_pytorch_model", /*keep_bf16=*/true);
    w.to_device(dev);
    sd::ZImageTransformer tr(w);
    std::printf("loaded %zu tensors in %.1fs on %s\n", w.tensors.size(), duration<double>(steady_clock::now() - t0).count(),
                sd::device_name(dev));

    tr.forward(x.to(dev), t, cap.to(dev));  // warm-up (allocator, cuBLAS heuristics)
    t0 = steady_clock::now();
    sd::Tensor out = tr.forward(x.to(dev), t, cap.to(dev)).to(sd::Device::CPU);
    std::printf("forward %s, %lld text tokens, t=%.3f: %.1fs\n", x.shape_str().c_str(), (long long)cap.dim(1), t,
                duration<double>(steady_clock::now() - t0).count());
    if (out.numel() != ref.numel()) {
        std::printf("shape mismatch %s vs %s\n", out.shape_str().c_str(), ref.shape_str().c_str());
        return 1;
    }
    double max_err = 0, sum_err2 = 0, sum_ref2 = 0, ref_max = 0;
    for (int64_t i = 0; i < ref.numel(); ++i) {
        double e = std::fabs(out.ptr()[i] - ref.ptr()[i]);
        max_err = std::max(max_err, e);
        ref_max = std::max(ref_max, (double)std::fabs(ref.ptr()[i]));
        sum_err2 += e * e;
        sum_ref2 += (double)ref.ptr()[i] * ref.ptr()[i];
    }
    double rel = std::sqrt(sum_err2 / sum_ref2);
    std::printf("max|err| = %.4g (max|ref| = %.4g)  relative rms error = %.3e  %s\n", max_err, ref_max, rel,
                rel < 2e-2 ? "ok" : "FAIL");
    return rel < 2e-2 ? 0 : 1;
}
