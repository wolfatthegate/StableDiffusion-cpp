#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sd {

// Writes an 8-bit RGB PNG (stored/uncompressed deflate, no external libraries).
void write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb);

}  // namespace sd
