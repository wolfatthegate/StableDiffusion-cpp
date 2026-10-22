// ============================================================================
// qwen_tokenizer.h - Qwen2/Qwen3 byte-level BPE tokenizer (the text encoder
// tokenizer of Z-Image).
//
// Differences from the CLIP tokenizer in tokenizer.h: no lower-casing, no
// end-of-word marker, a different pre-tokenisation regex, special tokens
// (<|im_start|> ...) that are matched before BPE, and a chat template that
// wraps the prompt the way the Z-Image pipeline does:
//     <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
// Unicode normalisation (NFC) is not applied; text that is already composed,
// which is almost all text, tokenises identically.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

class QwenTokenizer {
public:
    // dir holds vocab.json, merges.txt and tokenizer_config.json (for the
    // special tokens).
    explicit QwenTokenizer(const std::string& dir);

    // Plain BPE ids for `text`, special tokens recognised literally.
    std::vector<int32_t> encode(const std::string& text) const;
    // Ids for the prompt wrapped in the chat template, truncated to max_len.
    std::vector<int32_t> encode_prompt(const std::string& prompt, int max_len = 512) const;
    // The chat-template text itself (for tests and logging).
    static std::string apply_chat_template(const std::string& prompt);

    int32_t id_of(const std::string& token) const;  // -1 if absent
    size_t vocab_size() const { return encoder_.size() + special_by_text_.size(); }

private:
    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<std::string, int32_t> ranks_;
    std::unordered_map<std::string, int32_t> special_by_text_;
    std::vector<std::string> special_texts_;  // longest first
    std::string byte_enc_[256];

    std::vector<std::string> bpe(std::vector<std::string> word) const;
    void encode_plain(const std::string& text, std::vector<int32_t>& out) const;
    static std::vector<std::string> pretokenize(const std::string& text);
};

}  // namespace sd
