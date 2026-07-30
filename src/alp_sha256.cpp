/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — SHA-256 helpers (ALP migration dry-run)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "alp_sha256.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace Thoth {
namespace {

struct Sha256Ctx {
    std::uint32_t state[8]{};
    std::uint64_t bitlen = 0;
    std::uint8_t data[64]{};
    std::size_t datalen = 0;
};

constexpr std::uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32U - n));
}

void sha256Transform(Sha256Ctx& ctx, const std::uint8_t data[64]) {
    std::uint32_t m[64]{};
    for (int i = 0; i < 16; ++i) {
        m[i] = (static_cast<std::uint32_t>(data[i * 4]) << 24)
            | (static_cast<std::uint32_t>(data[i * 4 + 1]) << 16)
            | (static_cast<std::uint32_t>(data[i * 4 + 2]) << 8)
            | static_cast<std::uint32_t>(data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
        const std::uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    std::uint32_t a = ctx.state[0];
    std::uint32_t b = ctx.state[1];
    std::uint32_t c = ctx.state[2];
    std::uint32_t d = ctx.state[3];
    std::uint32_t e = ctx.state[4];
    std::uint32_t f = ctx.state[5];
    std::uint32_t g = ctx.state[6];
    std::uint32_t h = ctx.state[7];

    for (int i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + m[i];
        const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
    ctx.state[5] += f;
    ctx.state[6] += g;
    ctx.state[7] += h;
}

void sha256Init(Sha256Ctx& ctx) {
    ctx.state[0] = 0x6a09e667;
    ctx.state[1] = 0xbb67ae85;
    ctx.state[2] = 0x3c6ef372;
    ctx.state[3] = 0xa54ff53a;
    ctx.state[4] = 0x510e527f;
    ctx.state[5] = 0x9b05688c;
    ctx.state[6] = 0x1f83d9ab;
    ctx.state[7] = 0x5be0cd19;
    ctx.bitlen = 0;
    ctx.datalen = 0;
}

void sha256Update(Sha256Ctx& ctx, const std::uint8_t* data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        ctx.data[ctx.datalen++] = data[i];
        if (ctx.datalen == 64) {
            sha256Transform(ctx, ctx.data);
            ctx.bitlen += 512;
            ctx.datalen = 0;
        }
    }
}

void sha256Final(Sha256Ctx& ctx, std::uint8_t hash[32]) {
    const std::uint32_t i = static_cast<std::uint32_t>(ctx.datalen);

    if (ctx.datalen < 56) {
        ctx.data[i] = 0x80;
        for (std::uint32_t j = i + 1; j < 56; ++j) {
            ctx.data[j] = 0x00;
        }
    } else {
        ctx.data[i] = 0x80;
        for (std::uint32_t j = i + 1; j < 64; ++j) {
            ctx.data[j] = 0x00;
        }
        sha256Transform(ctx, ctx.data);
        for (std::uint32_t j = 0; j < 56; ++j) {
            ctx.data[j] = 0x00;
        }
    }

    ctx.bitlen += ctx.datalen * 8;
    ctx.data[63] = static_cast<std::uint8_t>(ctx.bitlen);
    ctx.data[62] = static_cast<std::uint8_t>(ctx.bitlen >> 8);
    ctx.data[61] = static_cast<std::uint8_t>(ctx.bitlen >> 16);
    ctx.data[60] = static_cast<std::uint8_t>(ctx.bitlen >> 24);
    ctx.data[59] = static_cast<std::uint8_t>(ctx.bitlen >> 32);
    ctx.data[58] = static_cast<std::uint8_t>(ctx.bitlen >> 40);
    ctx.data[57] = static_cast<std::uint8_t>(ctx.bitlen >> 48);
    ctx.data[56] = static_cast<std::uint8_t>(ctx.bitlen >> 56);
    sha256Transform(ctx, ctx.data);

    for (int j = 0; j < 4; ++j) {
        hash[j] = static_cast<std::uint8_t>((ctx.state[0] >> (24 - j * 8)) & 0xff);
        hash[j + 4] = static_cast<std::uint8_t>((ctx.state[1] >> (24 - j * 8)) & 0xff);
        hash[j + 8] = static_cast<std::uint8_t>((ctx.state[2] >> (24 - j * 8)) & 0xff);
        hash[j + 12] = static_cast<std::uint8_t>((ctx.state[3] >> (24 - j * 8)) & 0xff);
        hash[j + 16] = static_cast<std::uint8_t>((ctx.state[4] >> (24 - j * 8)) & 0xff);
        hash[j + 20] = static_cast<std::uint8_t>((ctx.state[5] >> (24 - j * 8)) & 0xff);
        hash[j + 24] = static_cast<std::uint8_t>((ctx.state[6] >> (24 - j * 8)) & 0xff);
        hash[j + 28] = static_cast<std::uint8_t>((ctx.state[7] >> (24 - j * 8)) & 0xff);
    }
}

} // namespace

std::vector<std::uint8_t> sha256Bytes(const std::string& data) {
    Sha256Ctx ctx;
    sha256Init(ctx);
    sha256Update(ctx, reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
    std::uint8_t hash[32];
    sha256Final(ctx, hash);
    return {hash, hash + 32};
}

std::string sha256Hex(const std::string& data) {
    const auto bytes = sha256Bytes(data);
    std::ostringstream out;
    for (auto b : bytes) {
        out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return out.str();
}

std::string sha256HexFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    Sha256Ctx ctx;
    sha256Init(ctx);
    std::array<char, 8192> buffer{};
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto got = static_cast<std::size_t>(in.gcount());
        if (got > 0) {
            sha256Update(ctx, reinterpret_cast<const std::uint8_t*>(buffer.data()), got);
        }
    }
    std::uint8_t hash[32];
    sha256Final(ctx, hash);
    std::ostringstream out;
    for (int i = 0; i < 32; ++i) {
        out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return out.str();
}

} // namespace Thoth
