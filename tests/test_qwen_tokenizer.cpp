// Compares QwenTokenizer with reference ids produced by the Hugging Face
// tokenizer (tests/fixtures/qwen3_tokens.json, made by finetune/.venv).
// Usage: ./test_qwen_tokenizer [tokenizer-dir] [fixture]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "qwen_tokenizer.h"

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models/z-image-turbo/tokenizer";
    std::string fixture = argc > 2 ? argv[2] : "tests/fixtures/qwen3_tokens.json";
    sd::QwenTokenizer tok(dir);
    std::ifstream in(fixture);
    std::stringstream ss;
    ss << in.rdbuf();
    sd::Json cases = sd::parse_json(ss.str());
    int fails = 0;
    for (const sd::Json& c : cases.arr) {
        const std::string& prompt = c.at("prompt").str;
        std::vector<int32_t> want, want_raw;
        for (const sd::Json& v : c.at("ids").arr) want.push_back((int32_t)v.num);
        for (const sd::Json& v : c.at("raw_ids").arr) want_raw.push_back((int32_t)v.num);
        std::vector<int32_t> got = tok.encode_prompt(prompt), got_raw = tok.encode(prompt);
        bool ok = got == want && got_raw == want_raw && sd::QwenTokenizer::apply_chat_template(prompt) == c.at("templated").str;
        std::printf("%-45.45s %s\n", prompt.empty() ? "(empty)" : prompt.c_str(), ok ? "ok" : "MISMATCH");
        if (!ok) {
            ++fails;
            std::printf("   want:"); for (int32_t id : want_raw) std::printf(" %d", id);
            std::printf("\n   got: "); for (int32_t id : got_raw) std::printf(" %d", id);
            std::printf("\n");
        }
    }
    std::printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails ? 1 : 0;
}
