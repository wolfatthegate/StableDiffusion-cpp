#include <cstdio>
#include <string>
#include <vector>

#include "tokenizer.h"

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models/sd15";
    sd::CLIPTokenizer tok(dir + "/tokenizer/vocab.json", dir + "/tokenizer/merges.txt");
    struct Case { const char* text; std::vector<int32_t> want; };
    // Reference ids from the HF CLIPTokenizer for openai/clip-vit-large-patch14.
    const Case cases[] = {
        {"a photo of a cat", {320, 1125, 539, 320, 2368}},
        {"A Photo of a CAT", {320, 1125, 539, 320, 2368}},
        {"it's 2 dogs!!", {585, 568, 273, 3255, 748}},
        {"", {}},
    };
    int fails = 0;
    for (const Case& c : cases) {
        std::vector<int32_t> got = tok.tokenize(c.text);
        bool ok = got == c.want;
        std::printf("%-20s ->", c.text);
        for (int32_t id : got) std::printf(" %d", id);
        std::printf("  %s\n", ok ? "ok" : "MISMATCH");
        fails += !ok;
    }
    std::vector<int32_t> full = tok.encode("a photo of a cat");
    bool ok = full.size() == 77 && full[0] == 49406 && full[6] == 49407 && full[76] == 49407;
    std::printf("encode(): len %zu bos %d eos %d pad %d  %s\n", full.size(), full[0], full[6], full[76], ok ? "ok" : "FAIL");
    fails += !ok;
    return fails ? 1 : 0;
}
