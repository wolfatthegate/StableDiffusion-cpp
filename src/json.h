// ============================================================================
// json.h - Tiny JSON parser.
//
// Only what the project needs: the safetensors header (an object of tensor
// descriptors) and CLIP's vocab.json (an object of ~49k string->int pairs).
// Supports the full JSON grammar including \uXXXX escapes and surrogate
// pairs. Numbers are stored as double. Not optimised for huge documents.
// ============================================================================
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace sd {

// A parsed JSON value. Only the field matching `type` is meaningful.
// Objects keep their keys in document order (a vector, not a map), which is
// fine for the small lookups done here and preserves file order.
struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool b = false;                                 // Bool
    double num = 0;                                 // Number
    std::string str;                                // String (decoded UTF-8)
    std::vector<Json> arr;                          // Array
    std::vector<std::pair<std::string, Json>> obj;  // Object, in file order

    // Linear search for `key` in an object; nullptr if absent.
    const Json* find(const std::string& key) const;
    // Like find() but throws std::runtime_error if the key is missing.
    const Json& at(const std::string& key) const;
};

// Parses a complete JSON document. Throws std::runtime_error with the byte
// offset of the problem on malformed input or trailing garbage.
Json parse_json(const std::string& text);

}  // namespace sd
