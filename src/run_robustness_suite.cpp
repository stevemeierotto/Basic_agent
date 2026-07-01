/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C5 robustness & failure scenario harness
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_context.h"
#include "../include/embedding_engine.h"
#include "../include/index_manager.h"
#include "../include/robustness_cases.h"
#include "../include/robustness_mock_responses.h"
#include "file_handler.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string logPath() {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "robustness_suite.jsonl").string();
}

void appendJsonLine(const std::string& path, const nlohmann::json& event) {
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << event.dump() << '\n';
    }
}

Thoth::BenchmarkEnvironmentInputs makeRobustnessBenchmarkInputs(EmbeddingEngine* engine,
                                                                IndexManager* idx) {
    Thoth::BenchmarkEnvironmentInputs inputs;
    inputs.harness = "robustness_suite";
    inputs.tier = Thoth::BenchmarkTier::MOCK;
    inputs.model.llm_model = "mock";
    inputs.model.embedding_model = "tfidf-local";
    if (engine) {
        inputs.model.embedding_method = "TfIdf";
        inputs.model.embedding_dimension = engine->getDimension();
        inputs.model.embedding_internal_version = engine->getInternalVersion();
    }
    inputs.corpus_mode = Thoth::CorpusFingerprintMode::FAST;
    inputs.corpus_chunk_count = idx ? static_cast<int>(idx->getChunks().size()) : 0;
    inputs.thoth_env_flags = Thoth::collectThothEnvFlags();
    return inputs;
}

Thoth::IndexEnvironment indexEnvironmentFrom(EmbeddingEngine* engine, IndexManager* idx) {
    Thoth::IndexEnvironment index;
    if (!engine || !idx) {
        return index;
    }
    index.rag_index_header = {
        {"model_name", engine->getModelName()},
        {"embedding_dimension", engine->getDimension()},
        {"embedding_version", engine->getInternalVersion()},
        {"chunk_count", static_cast<int>(idx->getChunks().size())},
    };
    return index;
}

/** RAII: emit ROBUSTNESS_COMPLETE on normal exit, ROBUSTNESS_ABORTED if scope exits early. */
class RobustnessRunRecorder {
public:
    explicit RobustnessRunRecorder(Thoth::BenchmarkRun& run) : run_(run) {}

    ~RobustnessRunRecorder() {
        if (!finished_) {
            run_.emit("ROBUSTNESS_ABORTED", payload());
        }
    }

    void complete(int casesPassed, std::size_t caseCount) {
        cases_passed_ = casesPassed;
        case_count_ = caseCount;
        run_.emit("ROBUSTNESS_COMPLETE", payload());
        finished_ = true;
    }

private:
    nlohmann::json payload() const {
        return {{"cases_passed", cases_passed_}, {"case_count", case_count_}};
    }

    Thoth::BenchmarkRun& run_;
    int cases_passed_ = 0;
    std::size_t case_count_ = 0;
    bool finished_ = false;
};

nlohmann::json outcomeToJson(const Thoth::RobustnessCaseOutcome& out,
                             std::int64_t runTs,
                             const Thoth::BenchmarkAttribution& attribution) {
    return {
        {"event", "ROBUSTNESS_CASE"},
        {"timestamp_ms", runTs},
        {"run_id", attribution.run_id},
        {"env_hash", attribution.env_hash},
        {"case_id", out.case_id},
        {"category", out.category},
        {"scenario", out.scenario},
        {"terminal_state", out.terminal_state},
        {"failure_reason", out.failure_reason},
        {"reflection_cycles", out.reflection_cycles},
        {"fallback_used", out.fallback_used},
        {"planner_calls", out.planner_calls},
        {"structurally_valid_plan", out.structurally_valid_plan},
        {"valid_dependencies", out.valid_dependencies},
        {"synthesis_prompt_ok", out.synthesis_prompt_ok},
        {"duration_ms", out.duration_ms},
        {"pass", out.pass},
        {"pass_reason", out.pass_reason},
        {"details", out.details},
    };
}

} // namespace

int main() {
    std::cout << "C5 — Robustness Suite (mock-only, no Ollama)\n";

    unsetenv("THOTH_TEST_SUITE_DEV");
    setenv("THOTH_MOCK_LLM", "true", 1);

    auto probeEngine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    IndexManager probeIdx(probeEngine.get());

    Thoth::BenchmarkRun benchmarkRun = Thoth::BenchmarkRun::create(
        makeRobustnessBenchmarkInputs(probeEngine.get(), &probeIdx));
    benchmarkRun.bindIndex(indexEnvironmentFrom(probeEngine.get(), &probeIdx));
    const Thoth::BenchmarkAttribution suiteAttribution = benchmarkRun.attribution();

    std::cout << "BENCHMARK_ENV run_id=" << benchmarkRun.run_id()
              << " env_hash=" << benchmarkRun.environment_hash()
              << " index_hash=" << benchmarkRun.index_hash() << " tier=mock\n";

    RobustnessRunRecorder suiteRecorder(benchmarkRun);

    if (const char* abortSmoke = std::getenv("THOTH_ROBUSTNESS_BENCHMARK_ABORT_SMOKE");
        abortSmoke && (std::string(abortSmoke) == "1" || std::string(abortSmoke) == "true")) {
        std::cerr << "ROBUSTNESS: benchmark abort smoke — exiting before complete()\n";
        return 2;
    }

    const auto cases = Thoth::getRobustnessCases();
    const std::string logFile = logPath();
    const std::int64_t runTs = nowMs();

    {
        std::ofstream truncateLog(logFile, std::ios::trunc);
    }

    int passed = 0;
    for (const auto& spec : cases) {
        Thoth::RobustnessMockResponses::reset();
        const Thoth::RobustnessCaseOutcome out = Thoth::runRobustnessCase(spec, suiteAttribution);

        if (out.pass) {
            ++passed;
        }

        std::cout << "\n" << out.case_id << " [" << out.category << "] — " << out.scenario << '\n';
        std::cout << "  terminal_state=" << out.terminal_state;
        if (!out.failure_reason.empty()) {
            std::cout << " failure_reason=" << out.failure_reason;
        }
        if (out.reflection_cycles > 0) {
            std::cout << " reflection_cycles=" << out.reflection_cycles;
        }
        if (out.planner_calls > 0) {
            std::cout << " planner_calls=" << out.planner_calls;
        }
        if (out.fallback_used) {
            std::cout << " fallback_used=true";
        }
        if (out.valid_dependencies) {
            std::cout << " valid_dependencies=true";
        }
        if (out.synthesis_prompt_ok) {
            std::cout << " synthesis_prompt_ok=true";
        }
        std::cout << " duration_ms=" << out.duration_ms << '\n';
        std::cout << "  pass: " << (out.pass ? "YES" : "NO") << " — " << out.pass_reason << '\n';

        appendJsonLine(logFile, outcomeToJson(out, runTs, suiteAttribution));
    }

    appendJsonLine(logFile, {
        {"event", "ROBUSTNESS_SUMMARY"},
        {"timestamp_ms", runTs},
        {"run_id", suiteAttribution.run_id},
        {"env_hash", suiteAttribution.env_hash},
        {"case_count", cases.size()},
        {"cases_passed", passed},
    });

    std::cout << "\nSummary\n";
    std::cout << "  cases passed: " << passed << '/' << cases.size() << '\n';
    std::cout << "  log: " << logFile << '\n';

    suiteRecorder.complete(passed, cases.size());
    return passed == static_cast<int>(cases.size()) ? 0 : 2;
}
