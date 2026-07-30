/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — UUID v4 generation (ALP)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_UUID_H
#define THOTH_ALP_UUID_H

#include <cstdio>
#include <random>
#include <string>

namespace Thoth {
namespace AlpUuid {

inline std::string generateV4() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned> dis(0, 255);
    unsigned char bytes[16];
    for (auto& b : bytes) {
        b = static_cast<unsigned char>(dis(gen));
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3F) | 0x80);
    char buf[37];
    std::snprintf(buf,
                  sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  bytes[0],
                  bytes[1],
                  bytes[2],
                  bytes[3],
                  bytes[4],
                  bytes[5],
                  bytes[6],
                  bytes[7],
                  bytes[8],
                  bytes[9],
                  bytes[10],
                  bytes[11],
                  bytes[12],
                  bytes[13],
                  bytes[14],
                  bytes[15]);
    return buf;
}

} // namespace AlpUuid
} // namespace Thoth

#endif // THOTH_ALP_UUID_H
