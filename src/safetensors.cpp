// ============================================================================
// safetensors.cpp - mmap-based .safetensors reader with half-precision
// conversion. See safetensors.h for the file format.
// ============================================================================
#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "json.h"

namespace sd {

namespace {

// IEEE 754 half -> single precision, bit-exact, including subnormals,
// infinities and NaN. Half: 1 sign, 5 exponent (bias 15), 10 mantissa bits.
// Single: 1 sign, 8 exponent (bias 127), 23 mantissa bits.
float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: shift until the implicit leading 1 appears, adjusting the exponent
            uint32_t e = 113;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FF;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {  // inf / NaN
        bits = sign | 0x7F800000u | (mant << 13);
    } else {  // normal: rebias the exponent (127 - 15 = 112), widen the mantissa
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}


// RAII read-only memory map of a whole file. The OS pages data in on
// demand, so a 1.7 GB file costs no up-front read.
struct MappedFile {
    void* base = nullptr;
    size_t size = 0;
    explicit MappedFile(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st;
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("cannot stat " + path);
        }
        size = (size_t)st.st_size;
        base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
    }
    ~MappedFile() {
        if (base) munmap(base, size);
    }
};

}  // namespace

void Weights::load_safetensors(const std::string& path, bool keep_bf16, int64_t bf16_min_numel) {
    // 1. Map the file and parse the JSON header that follows the 8-byte length.
    MappedFile f(path);
    if (f.size < 8) throw std::runtime_error("safetensors: file too small: " + path);
    uint64_t hlen = 0;
    std::memcpy(&hlen, f.base, 8);
    if (hlen + 8 > f.size) throw std::runtime_error("safetensors: bad header length in " + path);
    std::string header((const char*)f.base + 8, (size_t)hlen);
    Json meta = parse_json(header);
    const uint8_t* data = (const uint8_t*)f.base + 8 + hlen;
    const size_t data_size = f.size - 8 - (size_t)hlen;

    // 2. Collect tensor names ("__metadata__" holds free-form strings).
    std::vector<std::string> names;
    for (const auto& kv : meta.obj)
        if (kv.first != "__metadata__") names.push_back(kv.first);

    // 3. Validate each descriptor and allocate its destination serially (the
    //    map must not be modified concurrently), remembering where its bytes
    //    are. Unsupported dtypes leave slots[i] null.
    std::vector<Tensor*> slots(names.size());
    std::vector<std::string> dtypes(names.size());
    std::vector<std::pair<size_t, size_t>> ranges(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        const Json& info = *meta.find(names[i]);
        std::string dtype = info.at("dtype").str;
        std::vector<int64_t> shape;
        for (const Json& d : info.at("shape").arr) shape.push_back((int64_t)d.num);
        const auto& off = info.at("data_offsets").arr;
        size_t b = (size_t)off[0].num, e = (size_t)off[1].num;
        if (e > data_size || b > e) throw std::runtime_error("safetensors: bad offsets for " + names[i]);
        if (dtype != "F32" && dtype != "F16" && dtype != "BF16") {
            std::fprintf(stderr, "  skipping %s (dtype %s)\n", names[i].c_str(), dtype.c_str());
            continue;
        }
        Tensor& t = tensors[names[i]];
        t.shape = shape;
        const int64_t n = Tensor::numel_of(shape);
        if (keep_bf16 && n >= bf16_min_numel) {
            t.dtype = DType::BF16;
            t.data16.resize((size_t)n);
        } else {
            t.data.resize((size_t)n);
        }
        slots[i] = &t;
        dtypes[i] = dtype;
        ranges[i] = {b, e};
    }
    // 4. Convert all tensors in parallel; each task writes only its own slot.
    parallel_for((int64_t)names.size(), [&](int64_t i) {
        Tensor* t = slots[i];
        if (!t) return;
        const uint8_t* src = data + ranges[i].first;
        const size_t n = (size_t)t->numel();
        const size_t bytes = ranges[i].second - ranges[i].first;
        if (t->is_bf16()) {
            const size_t elem = dtypes[i] == "F32" ? 4 : 2;
            if (bytes != n * elem) throw std::runtime_error("safetensors: size mismatch for " + names[i]);
            if (dtypes[i] == "BF16") {
                std::memcpy(t->data16.data(), src, bytes);
            } else if (dtypes[i] == "F16") {
                for (size_t j = 0; j < n; ++j) {
                    uint16_t h;
                    std::memcpy(&h, src + 2 * j, 2);
                    t->data16[j] = f32_to_bf16(f16_to_f32(h));
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    float f;
                    std::memcpy(&f, src + 4 * j, 4);
                    t->data16[j] = f32_to_bf16(f);
                }
            }
            return;
        }
        if (dtypes[i] == "F32") {
            if (bytes != n * 4) throw std::runtime_error("safetensors: size mismatch for " + names[i]);
            std::memcpy(t->data.data(), src, bytes);
        } else {
            if (bytes != n * 2) throw std::runtime_error("safetensors: size mismatch for " + names[i]);
            const bool bf = dtypes[i] == "BF16";
            for (size_t j = 0; j < n; ++j) {
                uint16_t h;
                std::memcpy(&h, src + 2 * j, 2);
                t->data[j] = bf ? bf16_to_f32(h) : f16_to_f32(h);
            }
        }
    });
}

void Weights::load_model(const std::string& dir, const std::string& base, bool keep_bf16) {
    struct stat st;
    auto exists = [&](const std::string& p) { return stat(p.c_str(), &st) == 0; };
    const std::string single = dir + "/" + base + ".safetensors";
    const std::string fp16 = dir + "/" + base + ".fp16.safetensors";
    const std::string index = dir + "/" + base + ".safetensors.index.json";
    if (exists(single)) {
        load_safetensors(single, keep_bf16);
    } else if (exists(index)) {
        std::ifstream in(index);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Json j = parse_json(text);
        std::vector<std::string> shards;
        for (const auto& kv : j.at("weight_map").obj)
            if (std::find(shards.begin(), shards.end(), kv.second.str) == shards.end()) shards.push_back(kv.second.str);
        std::sort(shards.begin(), shards.end());
        for (const auto& s : shards) load_safetensors(dir + "/" + s, keep_bf16);
    } else if (exists(fp16)) {
        load_safetensors(fp16, keep_bf16);
    } else {
        throw std::runtime_error("cannot find " + single + " (or a .fp16 / sharded .index.json variant)");
    }
}

const Tensor& Weights::get(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing weight: " + name);
    return it->second;
}

const Tensor* Weights::try_get(const std::string& name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
}

size_t Weights::total_params() const {
    size_t n = 0;
    for (const auto& kv : tensors) n += (size_t)kv.second.numel();
    return n;
}

}  // namespace sd
