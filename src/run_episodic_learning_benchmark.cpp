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
#include "../include/e2_strict_retrieval.h"
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

    void completeWiringCheckpoint(const std::string& wiring_stage,
                                  std::size_t case_count,
                                  bool retrieval_enabled = false,
                                  bool evaluation_boundary_verified = false) {
        wiring_stage_ = wiring_stage;
        case_count_ = case_count;
        retrieval_enabled_ = retrieval_enabled;
        evaluation_boundary_verified_ = evaluation_boundary_verified;
        run_.emit("E2_WIRING_CHECKPOINT", wiringPayload());
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

    nlohmann::json wiringPayload() const {
        return {{"wiring_stage", wiring_stage_},
                {"scoring_enabled", false},
                {"retrieval_enabled", retrieval_enabled_},
                {"evaluation_boundary_verified", evaluation_boundary_verified_},
                {"official_scoring", false},
                {"case_count", case_count_},
                {"scoring_function", Thoth::kEpisodicLearningScoringFunction}};
    }

    Thoth::BenchmarkRun& run_;
    std::string e2_outcome_;
    std::string wiring_stage_;
    float mean_lift_ = 0.0f;
    int cases_passed_ = 0;
    std::size_t case_count_ = 0;
    bool retrieval_enabled_ = false;
    bool evaluation_boundary_verified_ = false;
    bool finished_ = false;
};

[[maybe_unused]] bool plantAndConsolidate(Memory& memory,
                         EmbeddingEngine* engine,
                         const std::string& sessionId,
                         const std::string& plantMessage) {
    // Retained for E2-INTEGRATION tier (Phase C); uncalled on STRICT arm path (A2+).
    memory.configureConsolidation(nullptr, engine);
    memory.setActiveSessionId(sessionId);
    memory.addMessage("user", plantMessage);
    for (int i = 1; i < 60; ++i) {
        memory.addMessage("user", "filler turn " + std::to_string(i));
    }
    return !memory.getRecentWarmMemory(1).empty();
}

void addDistractorChunk(EmbeddingEngine* engine, IndexManager* idx, const std::string& text) {
    Thoth::addEpisodicEvalCorpusChunk(engine, idx, text);
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

struct E2CaseArmPlumbingResult {
    Thoth::EpisodicLearningArmObservation observation;
    Thoth::SealedEpisodeInjectionLog sealed_log;
    Thoth::E2StrictRetrievalResult strict_retrieval;
    Thoth::E2StrictRetrievalResult executive_strict_retrieval;
    bool harness_executive_equivalent = false;
};

nlohmann::json strictRetrievalDiagFields(const Thoth::E2StrictRetrievalResult& retrieval) {
    nlohmann::json chunkSummary = nlohmann::json::array();
    for (const auto& chunk : retrieval.chunks) {
        chunkSummary.push_back(
            {{"chunk_id", chunk.chunk_id},
             {"source", Thoth::retrievedChunkSourceToString(chunk.source)},
             {"source_id", chunk.source_id}});
    }
    return {{"retrieval_enabled", true},
            {"strict_retrieval_status",
             Thoth::e2ArmScoringStatusToString(retrieval.status)},
            {"strict_retrieval_chunk_count", retrieval.chunks.size()},
            {"strict_retrieval_chunks", chunkSummary}};
}

E2CaseArmPlumbingResult runCaseArm(const Thoth::EpisodicLearningCase& spec,
                                   const std::string& armLabel,
                                   const Thoth::BenchmarkAttribution& attribution,
                                   const Thoth::E2EvalConfig& strictConfig,
                                   const std::string& metricsLogPath,
                                   std::int64_t builderTimestampMs,
                                   bool strictBoundaryRetrieval,
                                   bool executiveStrictDispatch) {
    setenv("THOTH_MOCK_EPISODIC", "1", 1);
    setenv("THOTH_MOCK_LLM", "true", 1);

    const Thoth::SealedEpisodeInjectionLog sealedLog =
        Thoth::buildStrictInjectionLogFromCaseTable(spec, armLabel, builderTimestampMs);

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

    auto idx = new IndexManager(enginePtr);
    addDistractorChunk(enginePtr, idx, spec.index_distractor_text);

    const bool episodicRequired =
        Thoth::strictEpisodicContentRequired(spec, armLabel);

    Thoth::E2StrictRetrievalResult strictRetrieval;
    if (strictBoundaryRetrieval) {
        Thoth::E2StrictRetrievalInput retrievalInput;
        retrievalInput.query = spec.goal;
        retrievalInput.episode_log = &sealedLog;
        retrievalInput.config = strictConfig;
        retrievalInput.index = idx;
        retrievalInput.engine = enginePtr;
        retrievalInput.top_k = 5;
        strictRetrieval = Thoth::e2StrictRetrieve(retrievalInput);
    }

    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
    rag->setEventCallback([&](const ControllerEvent& ev) { (void)ev; });

    auto planner = std::make_shared<Thoth::EpisodicLearningMockPlanner>(spec.validation_token);
    auto registry = std::make_shared<ToolRegistry>();
    Thoth::ExecutiveController controller(planner, registry, rag, memory);
    controller.set_max_reflections(0);
    if (executiveStrictDispatch) {
        controller.set_e2_strict_eval_context(&sealedLog, &strictConfig);
    }

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

    Thoth::E2StrictRetrievalResult executiveRetrieval;
    if (executiveStrictDispatch) {
        if (const auto execResult =
                Thoth::executiveStrictRetrievalFromPlan(controller.get_current_plan())) {
            executiveRetrieval = *execResult;
        } else {
            executiveRetrieval.status = Thoth::E2ArmScoringStatus::FAILED_RETRIEVAL;
            executiveRetrieval.error_message = "missing executive STRICT RETRIEVAL step result";
        }
    }

    if (executiveStrictDispatch) {
        obs.retrieval = Thoth::provenanceFromStrictRetrievalResult(
            executiveRetrieval, spec.expectations, episodicRequired);
        obs.arm_scoring_status = obs.retrieval.arm_scoring_status;
    } else if (strictBoundaryRetrieval) {
        obs.retrieval = Thoth::provenanceFromStrictRetrievalResult(
            strictRetrieval, spec.expectations, episodicRequired);
    }
    obs.terminal_state = stateName(controller.get_state());
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

    (void)sealedLog;

    const bool harnessExecutiveEquivalent =
        executiveStrictDispatch && strictBoundaryRetrieval &&
        Thoth::e2StrictRetrievalResultsEquivalent(strictRetrieval, executiveRetrieval);

    if (executiveStrictDispatch) {
        controller.clear_e2_strict_eval_context();
    }

    fs::remove(cfg.database_path);
    return {obs,
            sealedLog,
            strictRetrieval,
            executiveRetrieval,
            harnessExecutiveEquivalent};
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
    strictConfig.versions.embedding_model_version =
        Thoth::makeEmbeddingModelVersionPin("TfIdf", probeEngine->getInternalVersion());
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

    std::string wiringStage = "A5";
    if (const char* stageEnv = std::getenv("THOTH_E2_WIRING_STAGE")) {
        wiringStage = stageEnv;
    }

    const auto cases = Thoth::getEpisodicLearningCases();
    const std::string logPath = benchmarkLogPath();
    const std::int64_t ts = nowMs();

    if (wiringStage == "A1") {
        std::cout << "E2 wiring checkpoint A1 — evaluation disabled (builder diag only)\n";

        for (const auto& spec : cases) {
            for (const char* armLabel : {"cold", "warm"}) {
                const Thoth::SealedEpisodeInjectionLog log =
                    Thoth::buildStrictInjectionLogFromCaseTable(spec, armLabel, ts);
                appendJsonLine(logPath, {{"event", "E2_STRICT_INJECTION_LOG_DIAG"},
                                         {"timestamp_ms", ts},
                                         {"run_id", suiteAttribution.run_id},
                                         {"env_hash", suiteAttribution.env_hash},
                                         {"wiring_stage", wiringStage},
                                         {"scoring_enabled", false},
                                         {"retrieval_enabled", false},
                                         {"official_scoring", false},
                                         {"case_id", spec.id},
                                         {"arm", armLabel},
                                         {"strict_injection_log", log.toJson()}});
            }
        }

        appendJsonLine(logPath, {{"event", "E2_WIRING_CHECKPOINT"},
                                 {"timestamp_ms", ts},
                                 {"run_id", suiteAttribution.run_id},
                                 {"env_hash", suiteAttribution.env_hash},
                                 {"wiring_stage", wiringStage},
                                 {"scoring_enabled", false},
                                 {"retrieval_enabled", false},
                                 {"official_scoring", false},
                                 {"case_count", cases.size()},
                                 {"evaluation_fingerprint", evalFingerprint.toJson()},
                                 {"e2_eval_config", strictConfig.toJson()}});

        std::cout << "  wiring checkpoint complete — " << cases.size() << " case(s), log: " << logPath
                  << '\n';
        suiteRecorder.completeWiringCheckpoint(wiringStage, cases.size());
        return 0;
    }

    if (wiringStage == "A2") {
        std::cout << "E2 wiring checkpoint A2 — arm plumbing smoke (no scoring, no retrieval)\n";

        for (const auto& spec : cases) {
            for (const char* armLabel : {"cold", "warm"}) {
                const E2CaseArmPlumbingResult armResult = runCaseArm(
                    spec, armLabel, suiteAttribution, strictConfig, metricsLog.string(), ts,
                    /*strictBoundaryRetrieval=*/false,
                    /*executiveStrictDispatch=*/false);
                appendJsonLine(logPath, {{"event", "E2_STRICT_INJECTION_LOG_DIAG"},
                                         {"timestamp_ms", ts},
                                         {"run_id", suiteAttribution.run_id},
                                         {"env_hash", suiteAttribution.env_hash},
                                         {"wiring_stage", wiringStage},
                                         {"scoring_enabled", false},
                                         {"retrieval_enabled", false},
                                         {"official_scoring", false},
                                         {"case_id", spec.id},
                                         {"arm", armLabel},
                                         {"strict_injection_log", armResult.sealed_log.toJson()},
                                         {"terminal_state", armResult.observation.terminal_state},
                                         {"wall_clock_ms", armResult.observation.wall_clock_ms}});
            }
        }

        appendJsonLine(logPath, {{"event", "E2_WIRING_CHECKPOINT"},
                                 {"timestamp_ms", ts},
                                 {"run_id", suiteAttribution.run_id},
                                 {"env_hash", suiteAttribution.env_hash},
                                 {"wiring_stage", wiringStage},
                                 {"scoring_enabled", false},
                                 {"retrieval_enabled", false},
                                 {"official_scoring", false},
                                 {"case_count", cases.size()},
                                 {"evaluation_fingerprint", evalFingerprint.toJson()},
                                 {"e2_eval_config", strictConfig.toJson()}});

        std::cout << "  wiring checkpoint complete — " << cases.size() << " case(s), log: " << logPath
                  << '\n';
        suiteRecorder.completeWiringCheckpoint(wiringStage, cases.size());
        return 0;
    }

    if (wiringStage == "A3") {
        std::cout << "E2 wiring checkpoint A3 — kernel retrieval @ boundary (no scoring)\n";

        for (const auto& spec : cases) {
            for (const char* armLabel : {"cold", "warm"}) {
                const E2CaseArmPlumbingResult armResult = runCaseArm(
                    spec, armLabel, suiteAttribution, strictConfig, metricsLog.string(), ts,
                    /*strictBoundaryRetrieval=*/true,
                    /*executiveStrictDispatch=*/false);
                nlohmann::json row = {{"event", "E2_STRICT_INJECTION_LOG_DIAG"},
                                      {"timestamp_ms", ts},
                                      {"run_id", suiteAttribution.run_id},
                                      {"env_hash", suiteAttribution.env_hash},
                                      {"wiring_stage", wiringStage},
                                      {"scoring_enabled", false},
                                      {"official_scoring", false},
                                      {"evaluation_boundary_verified", true},
                                      {"case_id", spec.id},
                                      {"arm", armLabel},
                                      {"strict_injection_log", armResult.sealed_log.toJson()},
                                      {"terminal_state", armResult.observation.terminal_state},
                                      {"wall_clock_ms", armResult.observation.wall_clock_ms},
                                      {"warm_retrieval_hit",
                                       armResult.observation.retrieval.warm_retrieval_hit},
                                      {"boundary_provenance_status",
                                       Thoth::e2ArmScoringStatusToString(
                                           armResult.observation.retrieval.arm_scoring_status)}};
                row.update(strictRetrievalDiagFields(armResult.strict_retrieval));
                appendJsonLine(logPath, row);
            }
        }

        appendJsonLine(logPath, {{"event", "E2_WIRING_CHECKPOINT"},
                                 {"timestamp_ms", ts},
                                 {"run_id", suiteAttribution.run_id},
                                 {"env_hash", suiteAttribution.env_hash},
                                 {"wiring_stage", wiringStage},
                                 {"scoring_enabled", false},
                                 {"retrieval_enabled", true},
                                 {"evaluation_boundary_verified", true},
                                 {"official_scoring", false},
                                 {"case_count", cases.size()},
                                 {"evaluation_fingerprint", evalFingerprint.toJson()},
                                 {"e2_eval_config", strictConfig.toJson()}});

        std::cout << "  wiring checkpoint complete — " << cases.size() << " case(s), log: " << logPath
                  << '\n';
        suiteRecorder.completeWiringCheckpoint(
            wiringStage, cases.size(), /*retrieval_enabled=*/true,
            /*evaluation_boundary_verified=*/true);
        return 0;
    }

    if (wiringStage == "A4" || wiringStage == "A5") {
        const bool runtimeHeuristicGuard = (wiringStage == "A5");
        std::cout << "E2 wiring checkpoint " << wiringStage;
        if (runtimeHeuristicGuard) {
            std::cout << " — executive strict kernel + runtime heuristic guard";
        } else {
            std::cout << " — executive RETRIEVAL → strict kernel";
        }
        std::cout << '\n';

        bool allEquivalent = true;
        for (const auto& spec : cases) {
            for (const char* armLabel : {"cold", "warm"}) {
                const E2CaseArmPlumbingResult armResult = runCaseArm(
                    spec, armLabel, suiteAttribution, strictConfig, metricsLog.string(), ts,
                    /*strictBoundaryRetrieval=*/true,
                    /*executiveStrictDispatch=*/true);
                if (!armResult.harness_executive_equivalent) {
                    allEquivalent = false;
                    std::cerr << "[E2 " << wiringStage
                              << "] harness/executive retrieval mismatch: " << spec.id << " arm "
                              << armLabel << '\n';
                }
                nlohmann::json row = {{"event", "E2_STRICT_INJECTION_LOG_DIAG"},
                                      {"timestamp_ms", ts},
                                      {"run_id", suiteAttribution.run_id},
                                      {"env_hash", suiteAttribution.env_hash},
                                      {"wiring_stage", wiringStage},
                                      {"scoring_enabled", false},
                                      {"official_scoring", false},
                                      {"evaluation_boundary_verified", true},
                                      {"executive_strict_retrieval", true},
                                      {"harness_executive_retrieval_equivalent",
                                       armResult.harness_executive_equivalent},
                                      {"case_id", spec.id},
                                      {"arm", armLabel},
                                      {"strict_injection_log", armResult.sealed_log.toJson()},
                                      {"terminal_state", armResult.observation.terminal_state},
                                      {"wall_clock_ms", armResult.observation.wall_clock_ms},
                                      {"warm_retrieval_hit",
                                       armResult.observation.retrieval.warm_retrieval_hit},
                                      {"executive_provenance_status",
                                       Thoth::e2ArmScoringStatusToString(
                                           armResult.observation.retrieval.arm_scoring_status)}};
                if (runtimeHeuristicGuard) {
                    row["runtime_heuristic_guard"] = true;
                }
                row.update(strictRetrievalDiagFields(armResult.strict_retrieval));
                appendJsonLine(logPath, row);
            }
        }

        nlohmann::json checkpointRow = {{"event", "E2_WIRING_CHECKPOINT"},
                                        {"timestamp_ms", ts},
                                        {"run_id", suiteAttribution.run_id},
                                        {"env_hash", suiteAttribution.env_hash},
                                        {"wiring_stage", wiringStage},
                                        {"scoring_enabled", false},
                                        {"retrieval_enabled", true},
                                        {"evaluation_boundary_verified", true},
                                        {"executive_strict_retrieval", true},
                                        {"harness_executive_retrieval_equivalent", allEquivalent},
                                        {"official_scoring", false},
                                        {"case_count", cases.size()},
                                        {"evaluation_fingerprint", evalFingerprint.toJson()},
                                        {"e2_eval_config", strictConfig.toJson()}};
        if (runtimeHeuristicGuard) {
            checkpointRow["runtime_heuristic_guard"] = true;
        }
        appendJsonLine(logPath, checkpointRow);

        std::cout << "  wiring checkpoint complete — " << cases.size() << " case(s), log: "
                  << logPath << ", equivalence=" << (allEquivalent ? "yes" : "NO") << '\n';
        suiteRecorder.completeWiringCheckpoint(
            wiringStage, cases.size(), /*retrieval_enabled=*/true,
            /*evaluation_boundary_verified=*/true);
        return allEquivalent ? 0 : 3;
    }

    if (wiringStage == "SCORING") {
        std::cout << "E2 wiring SCORING — legacy full loop (dev only, not authoritative)\n";
    }

    std::vector<Thoth::EpisodicLearningCaseEvaluation> evaluations;
    std::vector<Thoth::EpisodicLearningExpectations> expectations;
    evaluations.reserve(cases.size());
    expectations.reserve(cases.size());

    int casesPassed = 0;

    for (const auto& spec : cases) {
        std::cout << "\n" << spec.id << " — " << spec.description << '\n';

        const E2CaseArmPlumbingResult coldArm = runCaseArm(
            spec, "cold", suiteAttribution, strictConfig, metricsLog.string(), ts,
            /*strictBoundaryRetrieval=*/true,
            /*executiveStrictDispatch=*/true);
        const E2CaseArmPlumbingResult warmArm = runCaseArm(
            spec, "warm", suiteAttribution, strictConfig, metricsLog.string(), ts,
            /*strictBoundaryRetrieval=*/true,
            /*executiveStrictDispatch=*/true);

        const auto eval = Thoth::evaluateEpisodicLearningCase(
            spec.id, spec.expectations, coldArm.observation, warmArm.observation, strictConfig);
        evaluations.push_back(eval);
        expectations.push_back(spec.expectations);

        if (eval.passes) {
            ++casesPassed;
        }

        std::cout << "  cold: state=" << coldArm.observation.terminal_state
                  << " score=" << coldArm.observation.final_success_score
                  << " warm_hit="
                  << (coldArm.observation.retrieval.warm_retrieval_hit ? "yes" : "no") << '\n';
        std::cout << "  warm: state=" << warmArm.observation.terminal_state
                  << " score=" << warmArm.observation.final_success_score
                  << " warm_hit="
                  << (warmArm.observation.retrieval.warm_retrieval_hit ? "yes" : "no");
        if (!warmArm.observation.retrieval.retrieved_memory_id.empty()) {
            std::cout << " mem_id=" << warmArm.observation.retrieval.retrieved_memory_id;
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
