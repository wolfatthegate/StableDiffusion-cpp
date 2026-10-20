// Loads .safetensors files (F32 / F16 / BF16 tensors) into float32 Tensors.
#pragma once

#include <string>
#include <unordered_map>

#include "tensor.h"

namespace sd {

struct Weights {
    std::unordered_map<std::string, Tensor> tensors;

    // Reads every float tensor in the file and adds it to `tensors`.
    void load_safetensors(const std::string& path);

    const Tensor& get(const std::string& name) const;        // throws if missing
    const Tensor* try_get(const std::string& name) const;    // nullptr if missing
    bool has(const std::string& name) const { return tensors.count(name) != 0; }
    size_t total_params() const;
};

}  // namespace sd
