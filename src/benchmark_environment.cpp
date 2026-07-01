/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 benchmark environment pinning (Checkpoint A)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_environment.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace Thoth {

namespace {

struct CorpusManifestEntry {
    std::string path;
    std::uintmax_t size_bytes = 0;
    std::int64_t mtime_ms = 0;
    std::string content_sha256;
};

bool envFlagTruthy(const nlohmann::json& flags, const char* name) {
    if (!flags.contains(name)) {
        return false;
    }
    const auto& value = flags.at(name);
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer()) {
        return value.get<int>() != 0;
    }
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        return text == "1" || text == "true" || text == "TRUE" || text == "yes";
    }
    return false;
}

std::int64_t fileMtimeMs(const fs::path& path) {
    std::error_code ec;
    const auto ftime = fs::last_write_time(path, ec);
    if (ec) {
        return 0;
    }
    const auto sctp = std::chrono::time_point_cast<std::chrono::milliseconds>(
        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    return sctp.time_since_epoch().count();
}

std::vector<fs::path> expandCorpusPaths(const std::vector<std::string>& corpusPaths) {
    std::vector<fs::path> files;
    for (const auto& rawPath : corpusPaths) {
        if (rawPath.empty()) {
            continue;
        }
        fs::path path(rawPath);
        std::error_code ec;
        if (!fs::exists(path, ec)) {
            continue;
        }
        if (fs::is_regular_file(path, ec)) {
            files.push_back(path);
            continue;
        }
        if (fs::is_directory(path, ec)) {
            for (const auto& entry : fs::recursive_directory_iterator(path, ec)) {
                if (ec) {
                    break;
                }
                if (entry.is_regular_file()) {
                    files.push_back(entry.path());
                }
            }
        }
    }
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) { return a.string() < b.string(); });
    return files;
}

std::string readFileForHash(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::vector<CorpusManifestEntry> buildCorpusManifest(const std::vector<std::string>& corpusPaths,
                                                     CorpusFingerprintMode mode) {
    std::vector<CorpusManifestEntry> manifest;
    for (const auto& filePath : expandCorpusPaths(corpusPaths)) {
        CorpusManifestEntry entry;
        entry.path = filePath.string();
        std::error_code ec;
        entry.size_bytes = fs::file_size(filePath, ec);
        entry.mtime_ms = fileMtimeMs(filePath);
        if (mode == CorpusFingerprintMode::STRICT) {
            entry.content_sha256 = sha256Hex(readFileForHash(filePath));
        }
        manifest.push_back(std::move(entry));
    }
    return manifest;
}

nlohmann::json manifestToJson(const std::vector<CorpusManifestEntry>& manifest,
                              CorpusFingerprintMode mode) {
    nlohmann::json rows = nlohmann::json::array();
    for (const auto& entry : manifest) {
        nlohmann::json row = {
            {"path", entry.path},
            {"size_bytes", entry.size_bytes},
            {"mtime_ms", entry.mtime_ms},
        };
        if (mode == CorpusFingerprintMode::STRICT) {
            row["content_sha256"] = entry.content_sha256;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::string computeCorpusFingerprint(const std::vector<std::string>& corpusPaths,
                                     CorpusFingerprintMode mode) {
    const auto manifest = buildCorpusManifest(corpusPaths, mode);
    const nlohmann::json canonical = manifestToJson(manifest, mode);
    return sha256Hex(canonical.dump());
}

std::string normalizeOllamaModelsDigest(const OllamaSnapshot& snapshot) {
    auto models = snapshot.models;
    std::sort(models.begin(), models.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    nlohmann::json rows = nlohmann::json::array();
    for (const auto& [name, digest] : models) {
        rows.push_back({{"name", name}, {"digest", digest}});
    }
    return sha256Hex(rows.dump());
}

nlohmann::json canonicalEnvironmentIdentityJson(const BenchmarkEnvironment& env) {
    return {
        {"tier", benchmarkTierToString(env.runtime.tier)},
        {"harness", env.runtime.harness},
        {"thoth_git_sha", env.prov.thoth_git_sha},
        {"basic_agent_git_sha", env.prov.basic_agent_git_sha},
        {"llm_model", env.model.llm_model},
        {"embedding_model", env.model.embedding_model},
        {"embedding_method", env.model.embedding_method},
        {"embedding_dimension", env.model.embedding_dimension},
        {"embedding_internal_version", env.model.embedding_internal_version},
        {"corpus_fingerprint", env.corpus.fingerprint},
        {"corpus_fingerprint_mode", env.corpus.fingerprint_mode},
        {"ollama_version", env.ollama.version},
        {"ollama_models_digest", env.ollama.models_digest},
        {"ollama_models_digest_version", env.ollama.models_digest_version},
    };
}

nlohmann::json canonicalIndexIdentityJson(const IndexEnvironment& index) {
    const auto& header = index.rag_index_header;
    return {
        {"model_name", header.value("model_name", "")},
        {"embedding_dimension", header.value("embedding_dimension", 0)},
        {"embedding_version", header.value("embedding_version", 0)},
        {"chunk_count", header.value("chunk_count", 0)},
    };
}

// Minimal SHA-256 (public-domain style implementation for E1 hashing).
class Sha256 {
public:
    Sha256() { reset(); }

    void update(const uint8_t* data, size_t length) {
        for (size_t i = 0; i < length; ++i) {
            buffer_[buffer_len_++] = data[i];
            if (buffer_len_ == 64) {
                transform(buffer_);
                bit_len_ += 512;
                buffer_len_ = 0;
            }
        }
    }

    void update(const std::string& data) {
        update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
    }

    std::array<uint8_t, 32> digest() {
        const uint64_t bitLenBefore = bit_len_ + (static_cast<uint64_t>(buffer_len_) * 8ULL);
        buffer_[buffer_len_++] = 0x80;
        if (buffer_len_ > 56) {
            while (buffer_len_ < 64) {
                buffer_[buffer_len_++] = 0;
            }
            transform(buffer_);
            buffer_len_ = 0;
        }
        while (buffer_len_ < 56) {
            buffer_[buffer_len_++] = 0;
        }
        for (int i = 7; i >= 0; --i) {
            buffer_[buffer_len_++] = static_cast<uint8_t>((bitLenBefore >> (i * 8)) & 0xff);
        }
        transform(buffer_);

        std::array<uint8_t, 32> hash{};
        for (size_t i = 0; i < 8; ++i) {
            hash[i * 4 + 0] = (state_[i] >> 24) & 0xff;
            hash[i * 4 + 1] = (state_[i] >> 16) & 0xff;
            hash[i * 4 + 2] = (state_[i] >> 8) & 0xff;
            hash[i * 4 + 3] = state_[i] & 0xff;
        }
        return hash;
    }

private:
    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void transform(const uint8_t block[64]) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};

        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    void reset() {
        state_[0] = 0x6a09e667;
        state_[1] = 0xbb67ae85;
        state_[2] = 0x3c6ef372;
        state_[3] = 0xa54ff53a;
        state_[4] = 0x510e527f;
        state_[5] = 0x9b05688c;
        state_[6] = 0x1f83d9ab;
        state_[7] = 0x5be0cd19;
        bit_len_ = 0;
        buffer_len_ = 0;
    }

    uint32_t state_[8]{};
    uint64_t bit_len_ = 0;
    uint8_t buffer_[64]{};
    size_t buffer_len_ = 0;
};

} // namespace

std::string sha256Hex(const std::string& data) {
    Sha256 hasher;
    hasher.update(data);
    const auto digest = hasher.digest();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const uint8_t byte : digest) {
        out << std::setw(2) << static_cast<int>(byte);
    }
    return out.str();
}

std::string benchmarkTierToString(BenchmarkTier tier) {
    switch (tier) {
    case BenchmarkTier::DEV:
        return "DEV";
    case BenchmarkTier::FULL:
        return "FULL";
    case BenchmarkTier::MOCK:
        return "MOCK";
    case BenchmarkTier::OLLAMA:
        return "OLLAMA";
    case BenchmarkTier::GUI:
        return "GUI";
    case BenchmarkTier::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

BenchmarkTier benchmarkTierFromString(const std::string& value) {
    if (value == "DEV") {
        return BenchmarkTier::DEV;
    }
    if (value == "FULL") {
        return BenchmarkTier::FULL;
    }
    if (value == "MOCK") {
        return BenchmarkTier::MOCK;
    }
    if (value == "OLLAMA") {
        return BenchmarkTier::OLLAMA;
    }
    if (value == "GUI") {
        return BenchmarkTier::GUI;
    }
    return BenchmarkTier::UNKNOWN;
}

std::string corpusFingerprintModeToString(CorpusFingerprintMode mode) {
    return mode == CorpusFingerprintMode::STRICT ? "STRICT" : "FAST";
}

BenchmarkTier inferTier(const BenchmarkEnvironmentInputs& inputs) {
    const nlohmann::json& flags = inputs.thoth_env_flags;

    if (envFlagTruthy(flags, "THOTH_TEST_SUITE_DEV")) {
        return BenchmarkTier::DEV;
    }

    if (inputs.harness == "test_suite") {
        if (inputs.tier == BenchmarkTier::FULL && inputs.ollama_reachable) {
            return BenchmarkTier::FULL;
        }
        if (inputs.model.embedding_method == "TfIdf" &&
            (envFlagTruthy(flags, "THOTH_MOCK_LLM") || inputs.model.llm_model == "mock")) {
            return BenchmarkTier::DEV;
        }
    }

    if (inputs.harness == "reflection_ab_benchmark" || inputs.harness == "robustness_suite" ||
        envFlagTruthy(flags, "THOTH_MOCK_LLM") || envFlagTruthy(flags, "THOTH_MOCK_EPISODIC")) {
        return BenchmarkTier::MOCK;
    }

    if (inputs.model.embedding_method == "External" && inputs.ollama_reachable &&
        (inputs.harness == "chat_rag_benchmark" || inputs.harness == "grag_benchmark" ||
         inputs.harness == "benchmark_reporter")) {
        return BenchmarkTier::OLLAMA;
    }

    if (inputs.tier != BenchmarkTier::UNKNOWN) {
        return inputs.tier;
    }

    return BenchmarkTier::UNKNOWN;
}

bool hasTierMismatch(const BenchmarkEnvironmentInputs& inputs) {
    return inputs.tier != inferTier(inputs);
}

BenchmarkEnvironment assembleEnvironment(const BenchmarkEnvironmentInputs& inputs) {
    BenchmarkEnvironment env;

    env.prov.thoth_git_sha = inputs.provenance.thoth_git_sha.empty() ? "unknown"
                                                                     : inputs.provenance.thoth_git_sha;
    env.prov.basic_agent_git_sha = inputs.provenance.basic_agent_git_sha.empty()
                                       ? "unknown"
                                       : inputs.provenance.basic_agent_git_sha;
    env.prov.captured_at_ms = inputs.provenance.captured_at_ms;
    if (inputs.include_hostname && inputs.provenance.hostname.has_value()) {
        env.prov.hostname = inputs.provenance.hostname;
    }

    env.model = {
        inputs.model.llm_model,
        inputs.model.embedding_model,
        inputs.model.embedding_method,
        inputs.model.embedding_dimension,
        inputs.model.embedding_internal_version,
    };

    env.runtime.tier = inputs.tier;
    env.runtime.harness = inputs.harness;
    env.runtime.thoth_env_flags = inputs.thoth_env_flags;

    env.corpus.fingerprint_mode = corpusFingerprintModeToString(inputs.corpus_mode);
    env.corpus.chunk_count = inputs.corpus_chunk_count;
    if (inputs.corpus_fingerprint_override.has_value()) {
        env.corpus.fingerprint = *inputs.corpus_fingerprint_override;
    } else if (!inputs.corpus_paths.empty()) {
        env.corpus.fingerprint = computeCorpusFingerprint(inputs.corpus_paths, inputs.corpus_mode);
        if (inputs.include_corpus_manifest) {
            env.corpus.file_manifest =
                manifestToJson(buildCorpusManifest(inputs.corpus_paths, inputs.corpus_mode),
                               inputs.corpus_mode);
        }
    }

    env.index.rag_index_header = inputs.rag_index_header;
    env.index.index_mismatch = inputs.index_mismatch;

    if (inputs.ollama.has_value()) {
        env.ollama.version = inputs.ollama->version;
        if (!inputs.ollama->version.empty() || !inputs.ollama->models.empty()) {
            env.ollama.models_digest = normalizeOllamaModelsDigest(*inputs.ollama);
            env.ollama.models_digest_version = "v1";
        }
    }

    env.environment_hash = computeEnvironmentHash(env);
    return env;
}

std::string computeEnvironmentHash(const BenchmarkEnvironment& env) {
    return sha256Hex(canonicalEnvironmentIdentityJson(env).dump());
}

std::string computeIndexHash(const IndexEnvironment& index) {
    if (index.rag_index_header.empty()) {
        return {};
    }
    return sha256Hex(canonicalIndexIdentityJson(index).dump());
}

nlohmann::json benchmarkEnvironmentToJson(const BenchmarkEnvironment& env) {
    nlohmann::json json = {
        {"prov",
         {
             {"thoth_git_sha", env.prov.thoth_git_sha},
             {"basic_agent_git_sha", env.prov.basic_agent_git_sha},
             {"captured_at_ms", env.prov.captured_at_ms},
         }},
        {"model",
         {
             {"llm_model", env.model.llm_model},
             {"embedding_model", env.model.embedding_model},
             {"embedding_method", env.model.embedding_method},
             {"embedding_dimension", env.model.embedding_dimension},
             {"embedding_internal_version", env.model.embedding_internal_version},
         }},
        {"runtime",
         {
             {"tier", benchmarkTierToString(env.runtime.tier)},
             {"harness", env.runtime.harness},
             {"thoth_env_flags", env.runtime.thoth_env_flags},
         }},
        {"corpus",
         {
             {"fingerprint", env.corpus.fingerprint},
             {"fingerprint_mode", env.corpus.fingerprint_mode},
             {"chunk_count", env.corpus.chunk_count},
         }},
        {"index",
         {
             {"rag_index_header", env.index.rag_index_header},
         }},
        {"ollama",
         {
             {"version", env.ollama.version},
             {"models_digest", env.ollama.models_digest},
             {"models_digest_version", env.ollama.models_digest_version},
         }},
        {"environment_hash", env.environment_hash},
    };

    if (env.prov.hostname.has_value()) {
        json["prov"]["hostname"] = *env.prov.hostname;
    }
    if (env.corpus.file_manifest.has_value()) {
        json["corpus"]["file_manifest"] = *env.corpus.file_manifest;
    }
    if (env.index.index_mismatch.has_value()) {
        json["index"]["index_mismatch"] = *env.index.index_mismatch;
    }
    return json;
}

BenchmarkEnvironment benchmarkEnvironmentFromJson(const nlohmann::json& json) {
    BenchmarkEnvironment env;
    if (json.contains("prov") && json["prov"].is_object()) {
        const auto& prov = json["prov"];
        env.prov.thoth_git_sha = prov.value("thoth_git_sha", "");
        env.prov.basic_agent_git_sha = prov.value("basic_agent_git_sha", "");
        env.prov.captured_at_ms = prov.value("captured_at_ms", 0);
        if (prov.contains("hostname") && prov["hostname"].is_string()) {
            env.prov.hostname = prov["hostname"].get<std::string>();
        }
    }
    if (json.contains("model") && json["model"].is_object()) {
        const auto& model = json["model"];
        env.model.llm_model = model.value("llm_model", "");
        env.model.embedding_model = model.value("embedding_model", "");
        env.model.embedding_method = model.value("embedding_method", "");
        env.model.embedding_dimension = model.value("embedding_dimension", 0);
        env.model.embedding_internal_version = model.value("embedding_internal_version", 0);
    }
    if (json.contains("runtime") && json["runtime"].is_object()) {
        const auto& runtime = json["runtime"];
        env.runtime.tier = benchmarkTierFromString(runtime.value("tier", "UNKNOWN"));
        env.runtime.harness = runtime.value("harness", "");
        env.runtime.thoth_env_flags = runtime.value("thoth_env_flags", nlohmann::json::object());
    }
    if (json.contains("corpus") && json["corpus"].is_object()) {
        const auto& corpus = json["corpus"];
        env.corpus.fingerprint = corpus.value("fingerprint", "");
        env.corpus.fingerprint_mode = corpus.value("fingerprint_mode", "FAST");
        env.corpus.chunk_count = corpus.value("chunk_count", 0);
        if (corpus.contains("file_manifest")) {
            env.corpus.file_manifest = corpus["file_manifest"];
        }
    }
    if (json.contains("index") && json["index"].is_object()) {
        const auto& index = json["index"];
        env.index.rag_index_header = index.value("rag_index_header", nlohmann::json::object());
        if (index.contains("index_mismatch")) {
            env.index.index_mismatch = index["index_mismatch"];
        }
    }
    if (json.contains("ollama") && json["ollama"].is_object()) {
        const auto& ollama = json["ollama"];
        env.ollama.version = ollama.value("version", "");
        env.ollama.models_digest = ollama.value("models_digest", "");
        env.ollama.models_digest_version = ollama.value("models_digest_version", "");
    }
    env.environment_hash = json.value("environment_hash", "");
    if (env.environment_hash.empty()) {
        env.environment_hash = computeEnvironmentHash(env);
    }
    return env;
}

} // namespace Thoth
