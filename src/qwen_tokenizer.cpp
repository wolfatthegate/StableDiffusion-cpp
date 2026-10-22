// ============================================================================
// qwen_tokenizer.cpp - See qwen_tokenizer.h.
//
// Pre-tokenisation follows the regex in tokenizer.json, tried alternative by
// alternative at each position (regex alternation is ordered):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)          contractions
//   [^\r\n\p{L}\p{N}]?\p{L}+               optional non-letter + letters
//   \p{N}                                  one digit
//    ?[^\s\p{L}\p{N}]+[\r\n]*              optional space + symbols + newlines
//   \s*[\r\n]+                             whitespace ending in newlines
//   \s+(?!\S)                              whitespace except its last char
//   \s+                                    whitespace
// ============================================================================
#include "qwen_tokenizer.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.h"

namespace sd {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

uint32_t next_cp(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (len == 1 || i + len > s.size()) {
        ++i;
        return c;
    }
    uint32_t cp = c & (0xFF >> (len + 1));
    for (int k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += len;
    return cp;
}

// Unicode character classes, approximated: exact for ASCII and the common
// punctuation, symbol and emoji blocks; every other non-ASCII code point is
// treated as a letter.
bool is_newline(uint32_t c) { return c == '\r' || c == '\n'; }
bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}
bool is_number(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0xFF10 && c <= 0xFF19) || (c >= 0x0660 && c <= 0x0669) ||
           (c >= 0x06F0 && c <= 0x06F9) || (c >= 0x0966 && c <= 0x096F) || c == 0xB2 || c == 0xB3 || c == 0xB9 ||
           (c >= 0xBC && c <= 0xBE) || (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) ||
           (c >= 0x2150 && c <= 0x2189) || (c >= 0x2460 && c <= 0x249B) || (c >= 0x24EA && c <= 0x24FF);
}
bool is_symbol_or_punct(uint32_t c) {
    if (c < 0x80) return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) && !is_space(c);
    return (c >= 0xA1 && c <= 0xBF && c != 0xAA && c != 0xB5 && c != 0xBA) || c == 0xD7 || c == 0xF7 ||
           (c >= 0x2010 && c <= 0x2027) || (c >= 0x2030 && c <= 0x205E) || (c >= 0x2070 && c <= 0x209F) ||
           (c >= 0x20A0 && c <= 0x20CF) || (c >= 0x2100 && c <= 0x214F) || (c >= 0x2190 && c <= 0x2BFF) ||
           (c >= 0x2E00 && c <= 0x2E7F) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE30 && c <= 0xFE4F) ||
           (c >= 0xFE50 && c <= 0xFE6F) || (c >= 0xFF00 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) ||
           (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65) || (c >= 0xFFE0 && c <= 0xFFEF) ||
           (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0xE000 && c <= 0xF8FF);
}
bool is_letter(uint32_t c) { return !is_space(c) && !is_number(c) && !is_symbol_or_punct(c); }

}  // namespace

QwenTokenizer::QwenTokenizer(const std::string& dir) {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) put_utf8(byte_enc_[b], in_bs[b] ? (uint32_t)b : (uint32_t)(256 + n++));

    Json vocab = parse_json(read_file(dir + "/vocab.json"));
    encoder_.reserve(vocab.obj.size() * 2);
    for (const auto& kv : vocab.obj) encoder_[kv.first] = (int32_t)kv.second.num;

    std::istringstream merges(read_file(dir + "/merges.txt"));
    std::string line;
    int32_t rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        ranks_[line] = rank++;
    }

    Json cfg = parse_json(read_file(dir + "/tokenizer_config.json"));
    if (const Json* added = cfg.find("added_tokens_decoder")) {
        for (const auto& kv : added->obj) {
            const Json* content = kv.second.find("content");
            if (content) special_by_text_[content->str] = (int32_t)std::stol(kv.first);
        }
    }
    for (const auto& kv : special_by_text_) special_texts_.push_back(kv.first);
    std::sort(special_texts_.begin(), special_texts_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

int32_t QwenTokenizer::id_of(const std::string& token) const {
    auto s = special_by_text_.find(token);
    if (s != special_by_text_.end()) return s->second;
    auto it = encoder_.find(token);
    return it == encoder_.end() ? -1 : it->second;
}

std::string QwenTokenizer::apply_chat_template(const std::string& prompt) {
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<std::string> QwenTokenizer::pretokenize(const std::string& t) {
    static const char* suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::vector<std::string> out;
    size_t i = 0;
    auto cp_at = [&](size_t pos, size_t& next) -> uint32_t {
        next = pos;
        return next_cp(t, next);
    };
    while (i < t.size()) {
        size_t j;
        uint32_t c = cp_at(i, j);
        // 1. contractions, case-insensitive
        if (c == '\'') {
            bool done = false;
            for (const char* sf : suffixes) {
                size_t n = std::char_traits<char>::length(sf);
                if (i + n <= t.size()) {
                    bool eq = true;
                    for (size_t k = 0; k < n && eq; ++k) eq = std::tolower((unsigned char)t[i + k]) == sf[k];
                    if (eq) {
                        out.push_back(t.substr(i, n));
                        i += n;
                        done = true;
                        break;
                    }
                }
            }
            if (done) continue;
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t k = i, k2;
            uint32_t c0 = cp_at(k, k2);
            if (!is_newline(c0) && !is_letter(c0) && !is_number(c0)) k = k2;  // optional prefix char
            size_t l = k, l2;
            size_t letters = 0;
            while (l < t.size() && is_letter(cp_at(l, l2))) {
                l = l2;
                ++letters;
            }
            if (letters > 0) {
                out.push_back(t.substr(i, l - i));
                i = l;
                continue;
            }
        }
        // 3. \p{N}
        if (is_number(c)) {
            out.push_back(t.substr(i, j - i));
            i = j;
            continue;
        }
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t k = i, k2;
            if (c == ' ') k = j;
            size_t l = k, l2;
            size_t syms = 0;
            while (l < t.size()) {
                uint32_t cc = cp_at(l, l2);
                if (is_space(cc) || is_letter(cc) || is_number(cc)) break;
                l = l2;
                ++syms;
            }
            if (syms > 0) {
                while (l < t.size() && is_newline(cp_at(l, k2))) l = k2;
                out.push_back(t.substr(i, l - i));
                i = l;
                continue;
            }
        }
        // 5. \s*[\r\n]+   6. \s+(?!\S)   7. \s+
        if (is_space(c)) {
            size_t l = i, l2;
            while (l < t.size() && is_space(cp_at(l, l2))) l = l2;
            // find the last newline inside the run (for rule 5)
            size_t last_nl_end = 0;
            size_t p = i;
            while (p < l) {
                size_t p2;
                uint32_t cc = cp_at(p, p2);
                if (is_newline(cc)) last_nl_end = p2;
                p = p2;
            }
            if (last_nl_end > 0) {
                out.push_back(t.substr(i, last_nl_end - i));
                i = last_nl_end;
                continue;
            }
            if (l < t.size()) {  // followed by non-space: keep the last whitespace char for the next piece
                size_t q = i, q2, last_start = i;
                while (q < l) {
                    last_start = q;
                    cp_at(q, q2);
                    q = q2;
                }
                if (last_start > i) {
                    out.push_back(t.substr(i, last_start - i));
                    i = last_start;
                    continue;
                }
                // single space before a symbol run: rule 4 handles " ?" only for symbols; a lone
                // space before a letter was consumed by rule 2. Fall through to rule 7.
            }
            out.push_back(t.substr(i, l - i));
            i = l;
            continue;
        }
        // Unreachable in practice; emit the code point as its own piece.
        out.push_back(t.substr(i, j - i));
        i = j;
    }
    return out;
}

std::vector<std::string> QwenTokenizer::bpe(std::vector<std::string> word) const {
    while (word.size() > 1) {
        int32_t best = INT_MAX;
        size_t bi = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = ranks_.find(word[i] + " " + word[i + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                bi = i;
            }
        }
        if (best == INT_MAX) break;
        const std::string first = word[bi], second = word[bi + 1];
        std::vector<std::string> merged;
        merged.reserve(word.size());
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(word[i]);
                ++i;
            }
        }
        word.swap(merged);
    }
    return word;
}

void QwenTokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out) const {
    for (const std::string& piece : pretokenize(text)) {
        std::vector<std::string> word;
        word.reserve(piece.size());
        for (unsigned char b : piece) word.push_back(byte_enc_[b]);
        for (const std::string& sym : bpe(word)) {
            auto it = encoder_.find(sym);
            if (it == encoder_.end()) throw std::runtime_error("qwen tokenizer: symbol not in vocabulary: " + sym);
            out.push_back(it->second);
        }
    }
}

std::vector<int32_t> QwenTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const std::string& sp : special_texts_) {
            if (text.compare(i, sp.size(), sp) == 0) {
                if (i > start) encode_plain(text.substr(start, i - start), out);
                out.push_back(special_by_text_.at(sp));
                i += sp.size();
                start = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    if (start < text.size()) encode_plain(text.substr(start), out);
    return out;
}

std::vector<int32_t> QwenTokenizer::encode_prompt(const std::string& prompt, int max_len) const {
    std::vector<int32_t> ids = encode(apply_chat_template(prompt));
    if ((int)ids.size() > max_len) ids.resize(max_len);
    return ids;
}

}  // namespace sd
