// Runs the C++ Qwen3 text encoder on a prompt and compares the penultimate
// hidden state with a dump from the PyTorch model
// (finetune/scripts/dump_reference.py text ...).
// Usage: ./test_qwen3 [model-dir] [dump-prefix]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "qwen3.h"
#include "qwen_tokenizer.h"
#include "safetensors.h"

static bool load_dump(const std::string& prefix, std::vector<int64_t>& shape, std::vector<float>& data) {
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
    return (bool)bin;
}

int main(int argc, char** argv) {
    using namespace std::chrono;
    std::string dir = argc > 1 ? argv[1] : "models/z-image-turbo";
    std::string dump = argc > 2 ? argv[2] : "tests/fixtures/ref_text";
    std::vector<int64_t> shape;
    std::vector<float> ref;
    if (!load_dump(dump, shape, ref)) {
        std::printf("no reference dump at %s.{json,bin}; run finetune/scripts/dump_reference.py text\n", dump.c_str());
        return 2;
    }
    std::ifstream js(dump + ".json");
    std::stringstream ss;
    ss << js.rdbuf();
    sd::Json meta = sd::parse_json(ss.str());
    const std::string prompt = meta.at("prompt").str;
    const int layer_index = (int)meta.at("layer").num;  // hidden_states index used (-2 => n_layers - 1 layers run)

    auto t0 = steady_clock::now();
    sd::Weights w;
    w.load_model(dir + "/text_encoder", "model", /*keep_bf16=*/true);
    sd::Qwen3Encoder enc(w);
    std::printf("loaded %zu tensors (%.1fs), %d layers, hidden %d\n", w.tensors.size(),
                duration<double>(steady_clock::now() - t0).count(), enc.num_layers(), enc.hidden_size());

    sd::QwenTokenizer tok(dir + "/tokenizer");
    std::vector<int32_t> ids = tok.encode_prompt(prompt);
    t0 = steady_clock::now();
    sd::Tensor out = enc.forward(ids, layer_index, /*final_norm=*/false);
    std::printf("forward on %zu tokens: %.2fs\n", ids.size(), duration<double>(steady_clock::now() - t0).count());

    if (out.numel() != (int64_t)ref.size()) {
        std::printf("shape mismatch: got %lld values, reference %zu\n", (long long)out.numel(), ref.size());
        return 1;
    }
    double max_err = 0, ref_max = 0, sum_err2 = 0, sum_ref2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = std::fabs(out.ptr()[i] - ref[i]);
        max_err = std::max(max_err, e);
        ref_max = std::max(ref_max, (double)std::fabs(ref[i]));
        sum_err2 += e * e;
        sum_ref2 += (double)ref[i] * ref[i];
    }
    double rel = std::sqrt(sum_err2 / sum_ref2);
    std::printf("max|err| = %.4g  (max|ref| = %.4g)  relative rms error = %.3e  %s\n", max_err, ref_max, rel,
                rel < 2e-2 ? "ok" : "FAIL");
    return rel < 2e-2 ? 0 : 1;
}
