// Tiny JSON parser: enough for safetensors headers and CLIP's vocab.json.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace sd {

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* find(const std::string& key) const;
    const Json& at(const std::string& key) const;  // throws if missing
};

Json parse_json(const std::string& text);

}  // namespace sd
