/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 benchmark environment pinning (Checkpoint A)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_BENCHMARK_ENVIRONMENT_H
#define THOTH_BENCHMARK_ENVIRONMENT_H

#include "json.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Thoth {

enum class BenchmarkTier : uint8_t {
    DEV,
    FULL,
    MOCK,
    OLLAMA,
    GUI,
    UNKNOWN,
};

enum class CorpusFingerprintMode : uint8_t {
    FAST,
    STRICT,
};

struct EnvironmentProvenance {
    std::string thoth_git_sha;
    std::string basic_agent_git_sha;
    std::int64_t captured_at_ms = 0;
    std::optional<std::string> hostname;
};

struct ModelEnvironment {
    std::string llm_model;
    std::string embedding_model;
    std::string embedding_method;
    int embedding_dimension = 0;
    int embedding_internal_version = 0;
};

struct RuntimeEnvironment {
    BenchmarkTier tier = BenchmarkTier::UNKNOWN;
    std::string harness;
    nlohmann::json thoth_env_flags = nlohmann::json::object();
};

struct CorpusEnvironment {
    std::string fingerprint;
    std::string fingerprint_mode;
    int chunk_count = 0;
    std::optional<nlohmann::json> file_manifest;
};

struct IndexEnvironment {
    nlohmann::json rag_index_header = nlohmann::json::object();
    std::optional<nlohmann::json> index_mismatch;
};

struct OllamaEnvironment {
    std::string version;
    std::string models_digest;
    std::string models_digest_version;
};

struct BenchmarkEnvironment {
    EnvironmentProvenance prov;
    ModelEnvironment model;
    RuntimeEnvironment runtime;
    CorpusEnvironment corpus;
    IndexEnvironment index;
    OllamaEnvironment ollama;

    std::string environment_hash;
};

/** Lightweight run attribution (Checkpoint B/C wiring). */
struct BenchmarkAttribution {
    std::string run_id;
    std::string env_hash;

    bool empty() const { return run_id.empty() && env_hash.empty(); }
};

struct OllamaSnapshot {
    std::string version;
    std::vector<std::pair<std::string, std::string>> models;
};

struct ProvenanceInputs {
    std::string thoth_git_sha;
    std::string basic_agent_git_sha;
    std::int64_t captured_at_ms = 0;
    std::optional<std::string> hostname;
};

struct ModelInputs {
    std::string llm_model;
    std::string embedding_model;
    std::string embedding_method;
    int embedding_dimension = 0;
    int embedding_internal_version = 0;
};

struct BenchmarkEnvironmentInputs {
    BenchmarkTier tier = BenchmarkTier::UNKNOWN;
    std::string harness;

    ProvenanceInputs provenance;
    ModelInputs model;

    std::vector<std::string> corpus_paths;
    CorpusFingerprintMode corpus_mode = CorpusFingerprintMode::FAST;
    int corpus_chunk_count = 0;
    /** When set, used directly (deterministic unit tests). Otherwise computed from corpus_paths. */
    std::optional<std::string> corpus_fingerprint_override;
    bool include_corpus_manifest = false;

    std::optional<OllamaSnapshot> ollama;
    nlohmann::json thoth_env_flags = nlohmann::json::object();

    nlohmann::json rag_index_header = nlohmann::json::object();
    std::optional<nlohmann::json> index_mismatch;

    bool include_hostname = false;
    /** Used by inferTier for FULL / OLLAMA classification. */
    bool ollama_reachable = false;
};

BenchmarkEnvironment assembleEnvironment(const BenchmarkEnvironmentInputs& inputs);
BenchmarkTier inferTier(const BenchmarkEnvironmentInputs& inputs);
bool hasTierMismatch(const BenchmarkEnvironmentInputs& inputs);

std::string computeEnvironmentHash(const BenchmarkEnvironment& env);
std::string computeIndexHash(const IndexEnvironment& index);

nlohmann::json benchmarkEnvironmentToJson(const BenchmarkEnvironment& env);
BenchmarkEnvironment benchmarkEnvironmentFromJson(const nlohmann::json& json);

std::string benchmarkTierToString(BenchmarkTier tier);
BenchmarkTier benchmarkTierFromString(const std::string& value);
std::string corpusFingerprintModeToString(CorpusFingerprintMode mode);

/** SHA-256 hex digest of UTF-8 text (used for environment_hash / index_hash). */
std::string sha256Hex(const std::string& data);

} // namespace Thoth

#endif // THOTH_BENCHMARK_ENVIRONMENT_H
