// ============================================================================
// tokenizer.h - CLIP byte-level BPE tokenizer.
//
// Reimplements the tokenizer of openai/clip-vit-large-patch14 (the one SD 1.x
// was trained with), using its vocab.json and merges.txt:
//
//   1. Lower-case the text (ASCII only, see limitations in README).
//   2. Pre-tokenise with CLIP's regex into words, numbers and punctuation runs.
//   3. Map each byte of a piece to a printable Unicode symbol (GPT-2's
//      bytes_to_unicode table) so every byte sequence has a string form.
//   4. Append "</w>" to the last symbol to mark the end of a word.
//   5. Repeatedly merge the adjacent symbol pair with the lowest rank in
//      merges.txt until no listed pair remains.
//   6. Look each resulting symbol up in vocab.json.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

class CLIPTokenizer {
public:
    // Loads the vocabulary (symbol -> id) and the ordered merge list.
    // Throws std::runtime_error if either file cannot be read or parsed.
    CLIPTokenizer(const std::string& vocab_path, const std::string& merges_path);

    // Full model input: [BOS] tokens... [EOS] [EOS]... of exactly max_len ids.
    // Text that does not fit is truncated so that BOS and EOS always remain.
    // SD 1.x pads with EOS (id 49407), not with a separate pad token.
    std::vector<int32_t> encode(const std::string& text, int max_len = 77) const;
    // BPE ids for the text with no BOS/EOS and no padding.
    std::vector<int32_t> tokenize(const std::string& text) const;

    int32_t bos() const { return bos_; }  // <|startoftext|>
    int32_t eos() const { return eos_; }  // <|endoftext|>

private:
    std::unordered_map<std::string, int32_t> encoder_;  // symbol -> token id
    // "first second" (the merges.txt line) -> merge priority; lower merges first
    std::unordered_map<std::string, int32_t> ranks_;
    std::string byte_enc_[256];  // byte -> its Unicode stand-in, UTF-8 encoded
    int32_t bos_ = 49406, eos_ = 49407;

    // Applies BPE merges to one pre-tokenised word given as a symbol list.
    std::vector<std::string> bpe(std::vector<std::string> word) const;
    // Splits text into pieces the way CLIP's regex does (whitespace dropped).
    static std::vector<std::string> pretokenize(const std::string& text);
};

}  // namespace sd
