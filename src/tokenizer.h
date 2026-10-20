// CLIP byte-level BPE tokenizer (vocab.json + merges.txt from openai/clip-vit-large-patch14).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

class CLIPTokenizer {
public:
    CLIPTokenizer(const std::string& vocab_path, const std::string& merges_path);

    // [BOS] tokens [EOS] padded with EOS to max_len (SD 1.x convention), truncated if needed.
    std::vector<int32_t> encode(const std::string& text, int max_len = 77) const;
    // Raw BPE ids without special tokens.
    std::vector<int32_t> tokenize(const std::string& text) const;

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }

private:
    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<std::string, int32_t> ranks_;  // "first second" -> merge priority
    std::string byte_enc_[256];                        // byte -> unicode symbol (UTF-8)
    int32_t bos_ = 49406, eos_ = 49407;

    std::vector<std::string> bpe(std::vector<std::string> word) const;
    static std::vector<std::string> pretokenize(const std::string& text);
};

}  // namespace sd
