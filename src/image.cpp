// ============================================================================
// image.cpp - Dependency-free PNG writer.
//
// A PNG is an 8-byte signature followed by chunks (length, 4-char type,
// payload, CRC-32 of type + payload). We emit three:
//   IHDR  width, height, bit depth 8, colour type 2 (RGB), no interlace
//   IDAT  a zlib stream of the scanlines, each prefixed with filter byte 0
//   IEND  empty terminator
// The zlib stream uses deflate "stored" blocks (raw bytes, at most 65535
// per block) followed by an Adler-32 checksum, so no compressor is needed.
// ============================================================================
#include "image.h"

#include <fstream>
#include <stdexcept>

namespace sd {

namespace {

// Table-driven CRC-32 (polynomial 0xEDB88320), as required by PNG chunks.
// The caller XORs the result with 0xFFFFFFFF to finalise.
uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

// Appends a big-endian 32-bit integer (PNG byte order).
void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

// Appends one complete chunk: length, type, payload, CRC.
void chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& payload) {
    put32(out, (uint32_t)payload.size());
    std::vector<uint8_t> body(type, type + 4);
    body.insert(body.end(), payload.begin(), payload.end());
    out.insert(out.end(), body.begin(), body.end());
    put32(out, crc32(body.data(), body.size()) ^ 0xFFFFFFFFu);
}

}  // namespace

std::vector<uint8_t> encode_png(int width, int height, const std::vector<uint8_t>& rgb) {
    if (rgb.size() != (size_t)width * height * 3) throw std::runtime_error("encode_png: buffer size mismatch");

    // Raw scanlines, each prefixed with filter type 0.
    std::vector<uint8_t> raw;
    raw.reserve((size_t)height * (width * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + (size_t)y * width * 3, rgb.begin() + (size_t)(y + 1) * width * 3);
    }

    // zlib stream: header 0x78 0x01 (deflate, 32K window, no preset
    // dictionary), then stored blocks. Each block header is 1 byte (BFINAL
    // bit, BTYPE = 00) followed by LEN and its one's complement NLEN.
    std::vector<uint8_t> z = {0x78, 0x01};
    size_t pos = 0;
    while (pos < raw.size() || raw.empty()) {
        size_t len = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back((uint8_t)(len & 0xFF));
        z.push_back((uint8_t)(len >> 8));
        z.push_back((uint8_t)(~len & 0xFF));
        z.push_back((uint8_t)((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
        if (raw.empty()) break;
    }
    // Adler-32 of the uncompressed data closes the zlib stream.
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    put32(z, (b << 16) | a);

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    put32(ihdr, (uint32_t)width);
    put32(ihdr, (uint32_t)height);
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // RGB
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});
    return png;
}

void write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> png = encode_png(width, height, rgb);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write((const char*)png.data(), (std::streamsize)png.size());
}

std::string base64_encode(const uint8_t* data, size_t n) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < n) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < n) v |= data[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < n ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < n ? tbl[v & 63] : '=';
    }
    return out;
}

}  // namespace sd
