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
        official_scoring_ = false;
        run_.emit("EPISODIC_LEARNING_COMPLETE", payload());
        finished_ = true;
    }

    void completeOfficial(const Thoth::EpisodicLearningRunEnvelope& envelope,
                          const std::string& outcome_display,
                          float mean_lift,
                          int cases_passed,
                          std::size_t case_count) {
        envelope_ = envelope;
        e2_outcome_ = outcome_display;
        mean_lift_ = mean_lift;
        cases_passed_ = cases_passed;
        case_count_ = case_count;
        official_scoring_ = envelope.official_scoring;
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
        nlohmann::json row = {{"mean_episodic_lift", mean_lift_},
                              {"cases_passed", cases_passed_},
                              {"case_count", case_count_},
                              {"scoring_function", Thoth::kEpisodicLearningScoringFunction},
                              {"official_scoring", official_scoring_},
                              {"scoring_enabled", envelope_.scoring_enabled}};
        if (!envelope_.wiring_stage.empty()) {
            row["wiring_stage"] = envelope_.wiring_stage;
        }
        if (!e2_outcome_.empty()) {
            row["e2_outcome"] = e2_outcome_;
        }
        return row;
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
    Thoth::EpisodicLearningRunEnvelope envelope_;
    float mean_lift_ = 0.0f;
    int cases_passed_ = 0;
    std::size_t case_count_ = 0;
    bool retrieval_enabled_ = false;
    bool evaluation_boundary_verified_ = false;
    bool official_scoring_ = false;
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
    Thoth::E2RunBlockReason run_block_reason = Thoth::E2RunBlockReason::NONE;
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

    const Thoth::E2RunBlockReason runBlockReason =
        Thoth::runBlockReasonFromPlan(controller.get_current_plan());

    fs::remove(cfg.database_path);
    return {obs,
            runBlockReason,
            sealedLog,
            strictRetrieval,
            executiveRetrieval,
            harnessExecutiveEquivalent};
}

struct ScoredLoopOutcome {
    std::vector<Thoth::EpisodicLearningCaseEvaluation> evaluations;
    std::vector<Thoth::EpisodicLearningExpectations> expectations;
    Thoth::EpisodicLearningSummary summary;
    int cases_passed = 0;
    std::string outcome_display;
};

/** B5 — sole scored-loop implementation; no wiring_stage conditionals inside. */
ScoredLoopOutcome runScoredEvaluationLoop(
    const std::vector<Thoth::EpisodicLearningCase>& cases,
    const Thoth::BenchmarkAttribution& suiteAttribution,
    const Thoth::E2EvalConfig& strictConfig,
    const Thoth::E2EvaluationFingerprint& evalFingerprint,
    const std::string& logPath,
    const std::string& metricsLogPath,
    std::int64_t ts) {
    ScoredLoopOutcome result;
    result.evaluations.reserve(cases.size());
    result.expectations.reserve(cases.size());

    const Thoth::EpisodicLearningLogContext logCtx{ts,
                                                   suiteAttribution.run_id,
                                                   suiteAttribution.env_hash,
                                                   evalFingerprint.toJson(),
                                                   strictConfig.toJson()};

    for (const auto& spec : cases) {
        std::cout << "\n" << spec.id << " — " << spec.description << '\n';

        const E2CaseArmPlumbingResult coldArm = runCaseArm(
            spec, "cold", suiteAttribution, strictConfig, metricsLogPath, ts,
            /*strictBoundaryRetrieval=*/true,
            /*executiveStrictDispatch=*/true);
        const E2CaseArmPlumbingResult warmArm = runCaseArm(
            spec, "warm", suiteAttribution, strictConfig, metricsLogPath, ts,
            /*strictBoundaryRetrieval=*/true,
            /*executiveStrictDispatch=*/true);

        const auto eval = Thoth::evaluateEpisodicLearningCase(
            spec.id, spec.expectations, coldArm.observation, warmArm.observation, strictConfig);
        Thoth::EpisodicLearningCaseEvaluation resolvedEval = eval;
        resolvedEval.run_block_reason = warmArm.run_block_reason;
        Thoth::applyCaseEvaluationResolution(resolvedEval);
        result.evaluations.push_back(resolvedEval);
        result.expectations.push_back(spec.expectations);

        if (resolvedEval.passes) {
            ++result.cases_passed;
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
        std::cout << "  lift=" << resolvedEval.lift << " pass=" << (resolvedEval.passes ? "YES" : "NO");
        if (!resolvedEval.failure_reason.empty()) {
            std::cout << " (" << resolvedEval.failure_reason << ')';
        }
        if (resolvedEval.evaluation_resolution.has_value()) {
            std::cout << " resolution="
                      << Thoth::e2EvaluationResolutionToString(*resolvedEval.evaluation_resolution);
        }
        std::cout << '\n';

        appendJsonLine(logPath, Thoth::episodicLearningCaseLogRow(logCtx, resolvedEval));
    }

    result.summary = Thoth::summarizeEpisodicLearning(
        result.evaluations, result.expectations, strictConfig);
    if (const auto exported = Thoth::e2OutcomeForExport(result.summary)) {
        result.outcome_display = Thoth::e2OutcomeToString(*exported);
    } else if (result.summary.evaluation_resolution.has_value()) {
        result.outcome_display =
            Thoth::e2EvaluationResolutionToString(*result.summary.evaluation_resolution);
    } else {
        result.outcome_display = Thoth::e2OutcomeToString(result.summary.outcome);
    }
    return result;
}

int runScoredEvaluationHarness(const std::vector<Thoth::EpisodicLearningCase>& cases,
                               const Thoth::BenchmarkAttribution& suiteAttribution,
                               const Thoth::E2EvalConfig& strictConfig,
                               const Thoth::E2EvaluationFingerprint& evalFingerprint,
                               const std::string& logPath,
                               const std::string& metricsLogPath,
                               std::int64_t ts,
                               const Thoth::EpisodicLearningRunEnvelope& envelope,
                               EpisodicLearningRunRecorder& suiteRecorder) {
    const ScoredLoopOutcome scored = runScoredEvaluationLoop(
        cases, suiteAttribution, strictConfig, evalFingerprint, logPath, metricsLogPath, ts);

    const Thoth::EpisodicLearningLogContext logCtx{ts,
                                                   suiteAttribution.run_id,
                                                   suiteAttribution.env_hash,
                                                   evalFingerprint.toJson(),
                                                   strictConfig.toJson()};
    appendJsonLine(logPath,
                   Thoth::episodicLearningSummaryLogRow(
                       logCtx, scored.summary, scored.cases_passed, cases.size(), envelope));

    std::cout << "\nSummary\n";
    std::cout << "  E2 outcome: " << scored.outcome_display << '\n';
    std::cout << "  mean episodic lift: " << scored.summary.mean_episodic_lift << '\n';
    std::cout << "  cases passed: " << scored.cases_passed << '/' << cases.size() << '\n';
    if (envelope.official_scoring) {
        std::cout << "  scorable_cases: " << scored.summary.scorable_cases
                  << " not_scorable_cases: " << scored.summary.not_scorable_cases << '\n';
    }
    std::cout << "  log: " << logPath << '\n';

    if (envelope.official_scoring) {
        suiteRecorder.completeOfficial(
            envelope, scored.outcome_display, scored.summary.mean_episodic_lift, scored.cases_passed,
            cases.size());
    } else {
        suiteRecorder.complete(
            scored.outcome_display, scored.summary.mean_episodic_lift, scored.cases_passed,
            cases.size());
    }

    const bool allCasesPass = scored.cases_passed == static_cast<int>(cases.size());
    return (allCasesPass && scored.summary.outcome == Thoth::E2Outcome::SUCCESS) ? 0 : 2;
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

    std::string wiringStage = "B";
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

    if (wiringStage == "B") {
        std::cout << "E2 wiring B — authoritative official scoring\n";
        const Thoth::EpisodicLearningRunEnvelope envelope{true, true, "B"};
        return runScoredEvaluationHarness(
            cases, suiteAttribution, strictConfig, evalFingerprint, logPath, metricsLog.string(), ts,
            envelope, suiteRecorder);
    }

    if (wiringStage == "SCORING") {
        std::cout << "E2 wiring SCORING — scored loop configuration (dev only, not authoritative)\n";
        const Thoth::EpisodicLearningRunEnvelope envelope{false, true, "SCORING"};
        return runScoredEvaluationHarness(
            cases, suiteAttribution, strictConfig, evalFingerprint, logPath, metricsLog.string(), ts,
            envelope, suiteRecorder);
    }

    std::cerr << "E2 wiring stage '" << wiringStage
              << "' is not supported. Use A1–A5 checkpoints, B (official), or SCORING (dev).\n";
    return 2;
}
