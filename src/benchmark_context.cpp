/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 benchmark context (Checkpoint B)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_context.h"
#include "../include/git_metadata.h"
#include "file_handler.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace fs = std::filesystem;

namespace Thoth {

namespace {

std::mutex& sidecarMutex() {
    static std::mutex mutex;
    return mutex;
}

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void appendEnvFlagIfSet(nlohmann::json& flags, const char* name) {
    const char* value = std::getenv(name);
    if (value && *value) {
        flags[name] = value;
    }
}

BenchmarkEnvironmentInputs enrichInputs(BenchmarkEnvironmentInputs inputs,
                                        const BenchmarkContextOptions& options) {
    if (inputs.provenance.captured_at_ms == 0) {
        inputs.provenance.captured_at_ms = nowMs();
    }

    if (options.auto_fill_git) {
        FileHandler fh;
        const std::string root = fh.getProjectRoot();
        std::string thothSha;
        std::string basicAgentSha;
        GitMetadata::readThothProject(root, thothSha, basicAgentSha);
        if (inputs.provenance.thoth_git_sha.empty()) {
            inputs.provenance.thoth_git_sha = thothSha;
        }
        if (inputs.provenance.basic_agent_git_sha.empty()) {
            inputs.provenance.basic_agent_git_sha = basicAgentSha;
        }
    }

    if (options.auto_collect_env_flags && inputs.thoth_env_flags.empty()) {
        inputs.thoth_env_flags = collectThothEnvFlags();
    }

    if (inputs.ollama.has_value() && !inputs.ollama->version.empty()) {
        inputs.ollama_reachable = true;
    }

    return inputs;
}

} // namespace

nlohmann::json collectThothEnvFlags() {
    nlohmann::json flags = nlohmann::json::object();
    appendEnvFlagIfSet(flags, "THOTH_TEST_SUITE_DEV");
    appendEnvFlagIfSet(flags, "THOTH_MOCK_LLM");
    appendEnvFlagIfSet(flags, "THOTH_MOCK_EPISODIC");
    appendEnvFlagIfSet(flags, "THOTH_MOCK_SCRAPE");
    appendEnvFlagIfSet(flags, "THOTH_MOCK_LLM_UNAVAILABLE");
    appendEnvFlagIfSet(flags, "OLLAMA_MODEL");
    appendEnvFlagIfSet(flags, "OLLAMA_EMBED_MODEL");
    appendEnvFlagIfSet(flags, "THOTH_INFERENCE_BASE_URL");
    appendEnvFlagIfSet(flags, "THOTH_EMBED_BASE_URL");
    appendEnvFlagIfSet(flags, "OLLAMA_HOST");
    return flags;
}

BenchmarkRun BenchmarkRun::create(const BenchmarkEnvironmentInputs& inputs,
                                  const BenchmarkContextOptions& options) {
    BenchmarkRun run;
    run.options_ = options;

    const BenchmarkEnvironmentInputs enriched = enrichInputs(inputs, options);
    run.environment_ = assembleEnvironment(enriched);
    run.environment_hash_ = run.environment_.environment_hash;
    run.run_id_ = "run-" + std::to_string(enriched.provenance.captured_at_ms);

    if (hasTierMismatch(enriched)) {
        run.emit("TIER_MISMATCH",
                 {{"declared_tier", benchmarkTierToString(enriched.tier)},
                  {"inferred_tier", benchmarkTierToString(inferTier(enriched))}});
    }

    {
        std::lock_guard<std::mutex> lock(sidecarMutex());
        run.writeSidecarLocked();
        run.appendJsonlLocked(run.makeEnvelope("BENCHMARK_ENV", nlohmann::json::object(), true));
    }

    return run;
}

BenchmarkAttribution BenchmarkRun::attribution() const {
    return BenchmarkAttribution{run_id_, environment_hash_};
}

void BenchmarkRun::emit(const std::string& event, const nlohmann::json& payload) const {
    const bool includeFullEnv = event == "BENCHMARK_ENV";
    const nlohmann::json envelope = makeEnvelope(event, payload, includeFullEnv);
    std::lock_guard<std::mutex> lock(sidecarMutex());
    appendJsonlLocked(envelope);
}

void BenchmarkRun::bindIndex(const IndexEnvironment& index) {
    const std::string newIndexHash = computeIndexHash(index);

    IndexEnvironment boundIndex = index;
    if (!index_hash_.empty() && !newIndexHash.empty() && newIndexHash != index_hash_) {
        boundIndex.index_mismatch = nlohmann::json{
            {"prior_hash", index_hash_},
            {"new_hash", newIndexHash},
        };
    }

    environment_.index = boundIndex;
    index_hash_ = newIndexHash;

    nlohmann::json payload = {
        {"index_hash", index_hash_},
        {"rag_index_header", boundIndex.rag_index_header},
    };
    if (boundIndex.index_mismatch.has_value()) {
        payload["index_mismatch"] = *boundIndex.index_mismatch;
    }

    std::lock_guard<std::mutex> lock(sidecarMutex());
    writeSidecarLocked();
    appendJsonlLocked(makeEnvelope("BENCHMARK_INDEX_BOUND", payload, false));
}

std::string BenchmarkRun::logsDirectory() const {
    if (options_.logs_directory.has_value()) {
        return *options_.logs_directory;
    }
    FileHandler fh;
    return fh.getLogsPath();
}

std::string BenchmarkRun::sidecarPath() const {
    return (fs::path(logsDirectory()) / "benchmark_env.latest.json").string();
}

std::string BenchmarkRun::jsonlPath() const {
    return (fs::path(logsDirectory()) / "benchmark_env.jsonl").string();
}

nlohmann::json BenchmarkRun::sidecarDocument() const {
    return {
        {"run_id", run_id_},
        {"environment_hash", environment_hash_},
        {"index_hash", index_hash_},
        {"environment", benchmarkEnvironmentToJson(environment_)},
    };
}

nlohmann::json BenchmarkRun::makeEnvelope(const std::string& event,
                                          const nlohmann::json& payload,
                                          bool includeFullEnv) const {
    nlohmann::json envelope = {
        {"event", event},
        {"ts", nowMs()},
        {"run_id", run_id_},
        {"env_hash", environment_hash_},
        {"payload", payload},
    };
    if (includeFullEnv) {
        envelope["env"] = benchmarkEnvironmentToJson(environment_);
    }
    return envelope;
}

void BenchmarkRun::writeSidecarLocked() const {
    fs::create_directories(logsDirectory());
    std::ofstream out(sidecarPath(), std::ios::trunc);
    if (out.is_open()) {
        out << sidecarDocument().dump(2) << '\n';
    }
}

void BenchmarkRun::appendJsonlLocked(const nlohmann::json& envelope) const {
    fs::create_directories(logsDirectory());
    std::ofstream out(jsonlPath(), std::ios::app);
    if (out.is_open()) {
        out << envelope.dump() << '\n';
    }
}

} // namespace Thoth
