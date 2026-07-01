/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic memory learning benchmark harness
 *
 * Spec: docs/E2_PROTOCOL.md v1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_context.h"
#include "../include/config.h"
#include "../include/e2_strict_enforcement.h"
#include "../include/embedding_engine.h"
#include "../include/episodic_learning_cases.h"
#include "../include/episodic_learning_eval.h"
#include "../include/executive_controller.h"
#include "../include/index_manager.h"
#include "../include/memory.h"
#include "../include/rag.h"
#include "../include/tools.h"
#include "file_handler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string benchmarkLogPath() {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "episodic_learning_benchmark.jsonl").string();
}

void appendJsonLine(const std::string& path, const nlohmann::json& event) {
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << event.dump() << '\n';
    }
}

std::string stateName(Thoth::ControllerState state) {
    switch (state) {
        case Thoth::ControllerState::COMPLETED:
            return "COMPLETED";
        case Thoth::ControllerState::FAILED:
            return "FAILED";
        case Thoth::ControllerState::ABORTED:
            return "ABORTED";
        default:
            return "INCOMPLETE";
    }
}

Thoth::BenchmarkEnvironmentInputs makeEpisodicBenchmarkInputs(EmbeddingEngine* engine,
                                                              IndexManager* idx) {
    Thoth::BenchmarkEnvironmentInputs inputs;
    inputs.harness = "episodic_learning_benchmark";
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

class EpisodicLearningRunRecorder {
public:
    explicit EpisodicLearningRunRecorder(Thoth::BenchmarkRun& run) : run_(run) {}

    ~EpisodicLearningRunRecorder() {
        if (!finished_) {
            run_.emit("EPISODIC_LEARNING_ABORTED", payload());
        }
    }

    void complete(const std::string& e2_outcome, float mean_lift, int cases_passed, std::size_t case_count) {
        e2_outcome_ = e2_outcome;
        mean_lift_ = mean_lift;
        cases_passed_ = cases_passed;
        case_count_ = case_count;
        run_.emit("EPISODIC_LEARNING_COMPLETE", payload());
        finished_ = true;
    }

private:
    nlohmann::json payload() const {
        return {{"e2_outcome", e2_outcome_},
                {"mean_episodic_lift", mean_lift_},
                {"cases_passed", cases_passed_},
                {"case_count", case_count_},
                {"scoring_function", Thoth::kEpisodicLearningScoringFunction}};
    }

    Thoth::BenchmarkRun& run_;
    std::string e2_outcome_;
    float mean_lift_ = 0.0f;
    int cases_passed_ = 0;
    std::size_t case_count_ = 0;
    bool finished_ = false;
};

bool plantAndConsolidate(Memory& memory,
                         EmbeddingEngine* engine,
                         const std::string& sessionId,
                         const std::string& plantMessage) {
    memory.configureConsolidation(nullptr, engine);
    memory.setActiveSessionId(sessionId);
    memory.addMessage("user", plantMessage);
    for (int i = 1; i < 60; ++i) {
        memory.addMessage("user", "filler turn " + std::to_string(i));
    }
    return !memory.getRecentWarmMemory(1).empty();
}

void addDistractorChunk(EmbeddingEngine* engine, IndexManager* idx, const std::string& text) {
    if (!engine || !idx || text.empty()) {
        return;
    }
    CodeChunk chunk;
    chunk.code = text;
    chunk.fileName = "e2-distractor.md";
    chunk.embedding = engine->embed(text);
    idx->addChunkToIndex(std::move(chunk));
}

std::optional<nlohmann::json> readLatestMetricsForGoal(const std::string& logPath,
                                                       const std::string& goalSubstring) {
    std::ifstream in(logPath);
    if (!in.is_open()) {
        return std::nullopt;
    }
    nlohmann::json last;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            auto row = nlohmann::json::parse(line);
            if (row.value("goal", "").find(goalSubstring) != std::string::npos) {
                last = row;
            }
        } catch (...) {
        }
    }
    if (last.is_null()) {
        return std::nullopt;
    }
    return last;
}

Thoth::EpisodicLearningArmObservation runCaseArm(const Thoth::EpisodicLearningCase& spec,
                                                 const std::string& armLabel,
                                                 const Thoth::BenchmarkAttribution& attribution,
                                                 const std::string& metricsLogPath) {
    setenv("THOTH_MOCK_EPISODIC", "1", 1);
    setenv("THOTH_MOCK_LLM", "true", 1);

    Config cfg;
    cfg.max_reflections = 0;
    cfg.database_path =
        (fs::temp_directory_path() / ("thoth_e2_" + spec.id + "_" + armLabel + ".db")).string();
    if (fs::exists(cfg.database_path)) {
        fs::remove(cfg.database_path);
    }

    auto memory = std::make_shared<Memory>(cfg);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf, &cfg);
    EmbeddingEngine* enginePtr = engine.get();

    const bool needsPlant =
        !spec.plant_message.empty() &&
        (spec.cold_arm_pre_consolidated || armLabel == "warm");
    if (needsPlant) {
        if (!plantAndConsolidate(*memory, enginePtr, spec.plant_session_id, spec.plant_message)) {
            std::cerr << "[E2] plant/consolidate failed for " << spec.id << " arm " << armLabel
                      << '\n';
        }
    }

    auto idx = new IndexManager(enginePtr);
    addDistractorChunk(enginePtr, idx, spec.index_distractor_text);
    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());

    Thoth::EpisodicRetrievalProvenance retrievalProv;
    rag->setEventCallback([&](const ControllerEvent& ev) {
        (void)ev;
    });

    auto planner = std::make_shared<Thoth::EpisodicLearningMockPlanner>(spec.validation_token);
    auto registry = std::make_shared<ToolRegistry>();
    Thoth::ExecutiveController controller(planner, registry, rag, memory);
    controller.set_max_reflections(0);

    const std::string goalSession = spec.id + "-" + armLabel + "-goal";
    memory->setActiveSessionId(goalSession);

    std::atomic<bool> terminal{false};
    controller.set_event_callback([&](const ControllerEvent& ev) {
        if (ev.type == EventType::PLAN_COMPLETED || ev.type == EventType::PLAN_FAILED ||
            ev.type == EventType::PLAN_ABORTED) {
            terminal.store(true);
        }
    });

    const auto start = nowMs();
    controller.execute_goal(spec.goal, attribution);

    int timeout = 150;
    while (!terminal.load() && timeout > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        --timeout;
    }

    Thoth::EpisodicLearningArmObservation obs;
    obs.arm_label = armLabel;
    obs.terminal_state = stateName(controller.get_state());
    for (const auto& step : controller.get_current_plan().steps) {
        if (step.type == StepType::RETRIEVAL && !step.result.is_null()) {
            retrievalProv =
                Thoth::provenanceFromRetrievalStepResult(step.result, spec.expectations);
            break;
        }
    }
    obs.retrieval = retrievalProv;
    obs.wall_clock_ms = nowMs() - start;

    if (const auto metrics = readLatestMetricsForGoal(metricsLogPath, spec.goal)) {
        obs.final_success_score = metrics->value("final_success_score", 0.0f);
        obs.planning_time_ms = metrics->value("planning_time_ms", 0);
        obs.total_tokens = metrics->value("total_tokens", 0);
        if (metrics->contains("run_id") && (*metrics)["run_id"].get<std::string>() != attribution.run_id) {
            std::cerr << "[E2] metrics run_id mismatch for " << spec.id << " arm " << armLabel << '\n';
        }
    } else if (obs.terminal_state == "COMPLETED") {
        obs.final_success_score = 1.0f;
    }

    fs::remove(cfg.database_path);
    return obs;
}

} // namespace

int main() {
    std::cout << "E2 — Episodic Memory Learning Benchmark (mock, no Ollama)\n";

    setenv("THOTH_MOCK_EPISODIC", "1", 1);
    setenv("THOTH_MOCK_LLM", "true", 1);

    FileHandler fh;
    const fs::path metricsLog =
        fs::path(fh.getProjectRoot()) / "logs" / "cognitive_metrics.jsonl";
    fs::create_directories(metricsLog.parent_path());
    setenv("THOTH_COGNITIVE_METRICS_LOG", metricsLog.string().c_str(), 1);

    auto probeEngine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    IndexManager probeIdx(probeEngine.get());

    Thoth::BenchmarkRun benchmarkRun =
        Thoth::BenchmarkRun::create(makeEpisodicBenchmarkInputs(probeEngine.get(), &probeIdx));
    benchmarkRun.bindIndex(indexEnvironmentFrom(probeEngine.get(), &probeIdx));
    const Thoth::BenchmarkAttribution suiteAttribution = benchmarkRun.attribution();

    std::cout << "BENCHMARK_ENV run_id=" << benchmarkRun.run_id()
              << " env_hash=" << benchmarkRun.environment_hash()
              << " index_hash=" << benchmarkRun.index_hash() << " tier=mock\n";

    EpisodicLearningRunRecorder suiteRecorder(benchmarkRun);

    Thoth::E2EvalConfig strictConfig;
    strictConfig.tier = Thoth::E2EvalTier::STRICT;
    strictConfig.versions.corpus_snapshot_id = benchmarkRun.index_hash();
    strictConfig.versions.model_version_or_weights_hash = "mock";
    strictConfig.versions.embedding_model_version = probeEngine->getInternalVersion();
    strictConfig.versions.retrieval_engine_version = Thoth::kE2StrictRetrievalEngineVersion;

    try {
        Thoth::validateStrictConfigForOfficialRun(strictConfig, /*uses_embeddings=*/true);
    } catch (const Thoth::E2StrictValidationError& e) {
        std::cerr << "STRICT config validation failed: " << e.what() << '\n';
        return 2;
    }

    const Thoth::E2EvaluationFingerprint evalFingerprint =
        Thoth::computeEvaluationFingerprint(strictConfig);
    Thoth::assertOfficialHarnessBuild();

    std::cout << "E2 evaluation fingerprint=" << evalFingerprint.fingerprint_hash << '\n';

    if (const char* abortSmoke = std::getenv("THOTH_EPISODIC_LEARNING_BENCHMARK_ABORT_SMOKE");
        abortSmoke && (std::string(abortSmoke) == "1" || std::string(abortSmoke) == "true")) {
        std::cerr << "EPISODIC_LEARNING: benchmark abort smoke — exiting before complete()\n";
        return 2;
    }

    const auto cases = Thoth::getEpisodicLearningCases();
    const std::string logPath = benchmarkLogPath();
    const std::int64_t ts = nowMs();

    std::vector<Thoth::EpisodicLearningCaseEvaluation> evaluations;
    std::vector<Thoth::EpisodicLearningExpectations> expectations;
    evaluations.reserve(cases.size());
    expectations.reserve(cases.size());

    int casesPassed = 0;

    for (const auto& spec : cases) {
        std::cout << "\n" << spec.id << " — " << spec.description << '\n';

        const Thoth::EpisodicLearningArmObservation coldArm =
            runCaseArm(spec, "cold", suiteAttribution, metricsLog.string());
        const Thoth::EpisodicLearningArmObservation warmArm =
            runCaseArm(spec, "warm", suiteAttribution, metricsLog.string());

        const auto eval = Thoth::evaluateEpisodicLearningCase(
            spec.id, spec.expectations, coldArm, warmArm, strictConfig);
        evaluations.push_back(eval);
        expectations.push_back(spec.expectations);

        if (eval.passes) {
            ++casesPassed;
        }

        std::cout << "  cold: state=" << coldArm.terminal_state
                  << " score=" << coldArm.final_success_score
                  << " warm_hit=" << (coldArm.retrieval.warm_retrieval_hit ? "yes" : "no") << '\n';
        std::cout << "  warm: state=" << warmArm.terminal_state
                  << " score=" << warmArm.final_success_score
                  << " warm_hit=" << (warmArm.retrieval.warm_retrieval_hit ? "yes" : "no");
        if (!warmArm.retrieval.retrieved_memory_id.empty()) {
            std::cout << " mem_id=" << warmArm.retrieval.retrieved_memory_id;
        }
        std::cout << '\n';
        std::cout << "  lift=" << eval.lift << " pass=" << (eval.passes ? "YES" : "NO");
        if (!eval.failure_reason.empty()) {
            std::cout << " (" << eval.failure_reason << ')';
        }
        std::cout << '\n';

        appendJsonLine(logPath, {{"event", "EPISODIC_LEARNING_CASE"},
                                 {"timestamp_ms", ts},
                                 {"run_id", suiteAttribution.run_id},
                                 {"env_hash", suiteAttribution.env_hash},
                                 {"scoring_function", Thoth::kEpisodicLearningScoringFunction},
                                 {"evaluation_fingerprint", evalFingerprint.toJson()},
                                 {"e2_eval_config", strictConfig.toJson()},
                                 {"case", Thoth::caseEvaluationToJson(eval)}});
    }

    const Thoth::EpisodicLearningSummary summary =
        Thoth::summarizeEpisodicLearning(evaluations, expectations, strictConfig);
    const std::string outcomeStr = Thoth::e2OutcomeToString(summary.outcome);

    appendJsonLine(logPath, {{"event", "EPISODIC_LEARNING_SUMMARY"},
                             {"timestamp_ms", ts},
                             {"run_id", suiteAttribution.run_id},
                             {"env_hash", suiteAttribution.env_hash},
                             {"scoring_function", Thoth::kEpisodicLearningScoringFunction},
                             {"evaluation_fingerprint", evalFingerprint.toJson()},
                             {"e2_eval_config", strictConfig.toJson()},
                             {"scoring_tier", "STRICT"},
                             {"official_scoring", true},
                             {"mean_episodic_lift", summary.mean_episodic_lift},
                             {"e2_outcome", outcomeStr},
                             {"outcome_rationale", summary.outcome_rationale},
                             {"cases_passed", casesPassed},
                             {"case_count", cases.size()},
                             {"case_results", [&]() {
                                  nlohmann::json arr = nlohmann::json::array();
                                  for (const auto& e : evaluations) {
                                      arr.push_back(Thoth::caseEvaluationToJson(e));
                                  }
                                  return arr;
                              }()}});

    std::cout << "\nSummary\n";
    std::cout << "  E2 outcome: " << outcomeStr << '\n';
    std::cout << "  mean episodic lift: " << summary.mean_episodic_lift << '\n';
    std::cout << "  cases passed: " << casesPassed << '/' << cases.size() << '\n';
    std::cout << "  log: " << logPath << '\n';

    suiteRecorder.complete(outcomeStr, summary.mean_episodic_lift, casesPassed, cases.size());

    const bool allCasesPass = casesPassed == static_cast<int>(cases.size());
    return (allCasesPass && summary.outcome == Thoth::E2Outcome::SUCCESS) ? 0 : 2;
}
