/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — SHA-256 helpers (ALP migration dry-run)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_SHA256_H
#define THOTH_ALP_SHA256_H

#include <cstdint>
#include <string>
#include <vector>

namespace Thoth {

std::vector<std::uint8_t> sha256Bytes(const std::string& data);
std::string sha256Hex(const std::string& data);
std::string sha256HexFile(const std::string& path);

} // namespace Thoth

#endif // THOTH_ALP_SHA256_H
