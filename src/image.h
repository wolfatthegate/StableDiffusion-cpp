// ============================================================================
// image.h - Minimal PNG output.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sd {

// Encodes an 8-bit-per-channel RGB image as PNG bytes (see write_png).
std::vector<uint8_t> encode_png(int width, int height, const std::vector<uint8_t>& rgb);

// Standard base64 (RFC 4648, with padding), for embedding images in JSON.
std::string base64_encode(const uint8_t* data, size_t n);

// Writes an 8-bit-per-channel RGB PNG.
//   rgb: width * height * 3 bytes, row-major, interleaved R, G, B.
// The image data is wrapped in "stored" (uncompressed) deflate blocks, so no
// compression library is needed; files are about 3 bytes per pixel.
// Throws std::runtime_error if the buffer size is wrong or the file cannot be
// written.
void write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb);

}  // namespace sd
