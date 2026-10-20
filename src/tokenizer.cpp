#include "tokenizer.h"

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

// Decodes one UTF-8 code point starting at s[i]; advances i. Invalid bytes map to themselves.
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

bool is_space(uint32_t cp) { return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v' || cp == 0xA0; }
bool is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
// Approximation of \p{L}: ASCII letters plus every non-ASCII code point.
bool is_letter(uint32_t cp) { return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= 0x80 && !is_space(cp)); }

}  // namespace

CLIPTokenizer::CLIPTokenizer(const std::string& vocab_path, const std::string& merges_path) {
    // bytes_to_unicode(): printable ranges map to themselves, the rest to 256+n.
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = in_bs[b] ? (uint32_t)b : (uint32_t)(256 + n++);
        put_utf8(byte_enc_[b], cp);
    }

    Json vocab = parse_json(read_file(vocab_path));
    if (vocab.type != Json::Object) throw std::runtime_error("vocab.json: expected an object");
    encoder_.reserve(vocab.obj.size() * 2);
    for (const auto& kv : vocab.obj) encoder_[kv.first] = (int32_t)kv.second.num;

    std::istringstream merges(read_file(merges_path));
    std::string line;
    int32_t rank = 0;
    bool first = true;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first) {  // "#version: 0.2" header
            first = false;
            if (line.rfind("#version", 0) == 0) continue;
        }
        if (line.empty()) continue;
        ranks_[line] = rank++;
    }

    auto it = encoder_.find("<|startoftext|>");
    if (it != encoder_.end()) bos_ = it->second;
    it = encoder_.find("<|endoftext|>");
    if (it != encoder_.end()) eos_ = it->second;
}

// Mirrors CLIP's regex:  <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
std::vector<std::string> CLIPTokenizer::pretokenize(const std::string& text) {
    static const char* specials[] = {"<|startoftext|>", "<|endoftext|>"};
    static const char* suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        size_t j = i;
        uint32_t cp = next_cp(text, j);
        if (is_space(cp)) {
            i = j;
            continue;
        }
        bool matched = false;
        for (const char* sp : specials) {
            size_t n = std::string(sp).size();
            if (text.compare(i, n, sp) == 0) {
                out.emplace_back(sp);
                i += n;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        if (cp == '\'') {
            for (const char* sf : suffixes) {
                size_t n = std::string(sf).size();
                if (text.compare(i, n, sf) == 0) {
                    out.emplace_back(sf);
                    i += n;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        if (is_letter(cp)) {
            size_t k = j;
            while (k < text.size()) {
                size_t k2 = k;
                if (!is_letter(next_cp(text, k2))) break;
                k = k2;
            }
            out.push_back(text.substr(i, k - i));
            i = k;
        } else if (is_digit(cp)) {
            out.push_back(text.substr(i, j - i));
            i = j;
        } else {
            size_t k = j;
            while (k < text.size()) {
                size_t k2 = k;
                uint32_t c2 = next_cp(text, k2);
                if (is_space(c2) || is_letter(c2) || is_digit(c2)) break;
                k = k2;
            }
            out.push_back(text.substr(i, k - i));
            i = k;
        }
    }
    return out;
}

std::vector<std::string> CLIPTokenizer::bpe(std::vector<std::string> word) const {
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

std::vector<int32_t> CLIPTokenizer::tokenize(const std::string& text_in) const {
    std::string text = text_in;
    for (char& c : text)
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    std::vector<int32_t> ids;
    for (const std::string& piece : pretokenize(text)) {
        auto sp = encoder_.find(piece);
        if ((piece == "<|startoftext|>" || piece == "<|endoftext|>") && sp != encoder_.end()) {
            ids.push_back(sp->second);
            continue;
        }
        std::vector<std::string> word;
        for (unsigned char b : piece) word.push_back(byte_enc_[b]);
        word.back() += "</w>";
        for (const std::string& sym : bpe(word)) {
            auto it = encoder_.find(sym);
            ids.push_back(it != encoder_.end() ? it->second : eos_);
        }
    }
    return ids;
}

std::vector<int32_t> CLIPTokenizer::encode(const std::string& text, int max_len) const {
    std::vector<int32_t> ids = tokenize(text);
    if ((int)ids.size() > max_len - 2) ids.resize(max_len - 2);
    std::vector<int32_t> out;
    out.reserve(max_len);
    out.push_back(bos_);
    out.insert(out.end(), ids.begin(), ids.end());
    out.push_back(eos_);
    while ((int)out.size() < max_len) out.push_back(eos_);
    return out;
}

}  // namespace sd
