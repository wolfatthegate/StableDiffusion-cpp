// ============================================================================
// safetensors.h - Loader for the .safetensors weight format.
//
// File layout (https://github.com/huggingface/safetensors):
//   [8 bytes]  little-endian uint64 N = header length
//   [N bytes]  JSON header: { "name": {"dtype": "F16", "shape": [..],
//                                      "data_offsets": [begin, end]}, ...,
//                             "__metadata__": {...} }
//   [rest]     raw tensor bytes; offsets are relative to the start of this
//              section
//
// The file is memory-mapped and every F32 / F16 / BF16 tensor is converted to
// float32 in parallel. Other dtypes (e.g. CLIP's I64 position_ids buffer) are
// skipped with a message on stderr.
// ============================================================================
#pragma once

#include <string>
#include <unordered_map>

#include "tensor.h"

namespace sd {

// A bag of named float32 tensors, keyed by their checkpoint names, for
// example "down_blocks.0.resnets.0.conv1.weight".
//
// Model classes (UNet, VAEDecoder, ...) keep raw pointers into `tensors`, so
// a Weights object must outlive every model built from it. Adding more
// tensors later is safe (unordered_map never moves its elements), but
// erasing or replacing a tensor that a model points to is not.
struct Weights {
    std::unordered_map<std::string, Tensor> tensors;

    // Reads every float tensor in the file and adds it to `tensors`.
    // With keep_bf16, tensors whose element count is at least bf16_min_numel
    // are stored as BF16 (converted from F16/F32 if needed) instead of F32;
    // small tensors such as norms and biases stay F32.
    // Throws std::runtime_error on I/O errors or a malformed file.
    void load_safetensors(const std::string& path, bool keep_bf16 = false, int64_t bf16_min_numel = 1 << 20);
    // Loads <dir>/<base>.safetensors, or every shard named in
    // <dir>/<base>.safetensors.index.json, or the .fp16 variant.
    void load_model(const std::string& dir, const std::string& base, bool keep_bf16 = false);

    // Moves every tensor to a device (GPU memory for Device::CUDA, freeing the
    // host copy). Models keep their pointers into `tensors`, which stay valid.
    void to_device(Device d);

    // Returns the tensor called `name`; throws "missing weight: name" if absent.
    const Tensor& get(const std::string& name) const;
    // Returns the tensor called `name`, or nullptr if absent (optional biases).
    const Tensor* try_get(const std::string& name) const;
    bool has(const std::string& name) const { return tensors.count(name) != 0; }
    // Total number of scalar parameters across all loaded tensors.
    size_t total_params() const;
};

}  // namespace sd
