/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 benchmark context (Checkpoint B)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_BENCHMARK_CONTEXT_H
#define THOTH_BENCHMARK_CONTEXT_H

#include "benchmark_environment.h"

#include <optional>
#include <string>

namespace Thoth {

struct BenchmarkContextOptions {
    /** When set, sidecar + JSONL write here instead of <project>/logs. */
    std::optional<std::string> logs_directory;
    bool auto_fill_git = true;
    bool auto_collect_env_flags = true;
};

class BenchmarkRun {
public:
    static BenchmarkRun create(const BenchmarkEnvironmentInputs& inputs,
                               const BenchmarkContextOptions& options = {});

    BenchmarkAttribution attribution() const;

    void emit(const std::string& event, const nlohmann::json& payload = {}) const;
    void bindIndex(const IndexEnvironment& index);

    const std::string& run_id() const { return run_id_; }
    const std::string& environment_hash() const { return environment_hash_; }
    const std::string& index_hash() const { return index_hash_; }
    const BenchmarkEnvironment& environment() const { return environment_; }

private:
    BenchmarkRun() = default;

    void writeSidecarLocked() const;
    void appendJsonlLocked(const nlohmann::json& envelope) const;
    std::string logsDirectory() const;
    std::string sidecarPath() const;
    std::string jsonlPath() const;
    nlohmann::json sidecarDocument() const;
    nlohmann::json makeEnvelope(const std::string& event,
                                const nlohmann::json& payload,
                                bool includeFullEnv) const;

    BenchmarkContextOptions options_;
    std::string run_id_;
    std::string environment_hash_;
    std::string index_hash_;
    BenchmarkEnvironment environment_;
};

/** Collect THOTH_* env vars relevant to benchmark tier classification. */
nlohmann::json collectThothEnvFlags();

} // namespace Thoth

#endif // THOTH_BENCHMARK_CONTEXT_H
