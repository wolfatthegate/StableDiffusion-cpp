// ============================================================================
// json.cpp - Recursive-descent JSON parser. See json.h.
// ============================================================================
#include "json.h"

#include <cstdlib>
#include <stdexcept>

namespace sd {

const Json* Json::find(const std::string& key) const {
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const Json& Json::at(const std::string& key) const {
    const Json* v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + key + "'");
    return *v;
}

namespace {

// Single-pass parser over a string; pos_ is the current byte offset.
// One method per grammar rule: value() dispatches on the first character.
class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    Json parse() {
        Json v = value();
        ws();
        if (pos_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("json: " + msg + " at offset " + std::to_string(pos_));
    }
    // Skips JSON whitespace.
    void ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) ++pos_;
    }
    char peek() {
        if (pos_ >= s_.size()) fail("unexpected end");
        return s_[pos_];
    }
    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }
    // Consumes `lit` if the input continues with it.
    bool consume(const char* lit) {
        size_t n = 0;
        while (lit[n]) ++n;
        if (s_.compare(pos_, n, lit) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    // Appends code point cp to out as UTF-8 (1 to 4 bytes).
    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    // Reads the 4 hex digits of a backslash-u (XXXX) escape.
    uint32_t hex4() {
        if (pos_ + 4 > s_.size()) fail("bad \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }

    // Parses a quoted string, decoding escapes. A high surrogate followed by
    // an escaped low surrogate is combined into one supplementary code point.
    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("bad escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && consume("\\u")) {
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }

    // Parses any value: object, array, string, true/false/null, or number
    // (numbers are delegated to strtod).
    Json value() {
        ws();
        Json v;
        char c = peek();
        if (c == '{') {
            v.type = Json::Object;
            ++pos_;
            ws();
            if (peek() == '}') {
                ++pos_;
                return v;
            }
            for (;;) {
                ws();
                std::string key = string();
                ws();
                expect(':');
                Json val = value();
                v.obj.emplace_back(std::move(key), std::move(val));
                ws();
                if (peek() == ',') {
                    ++pos_;
                    continue;
                }
                expect('}');
                break;
            }
        } else if (c == '[') {
            v.type = Json::Array;
            ++pos_;
            ws();
            if (peek() == ']') {
                ++pos_;
                return v;
            }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (peek() == ',') {
                    ++pos_;
                    continue;
                }
                expect(']');
                break;
            }
        } else if (c == '"') {
            v.type = Json::String;
            v.str = string();
        } else if (consume("true")) {
            v.type = Json::Bool;
            v.b = true;
        } else if (consume("false")) {
            v.type = Json::Bool;
        } else if (consume("null")) {
            v.type = Json::Null;
        } else {
            v.type = Json::Number;
            const char* start = s_.c_str() + pos_;
            char* end = nullptr;
            v.num = std::strtod(start, &end);
            if (end == start) fail("bad token");
            pos_ += (size_t)(end - start);
        }
        return v;
    }
};

}  // namespace

Json parse_json(const std::string& text) { return Parser(text).parse(); }

}  // namespace sd
