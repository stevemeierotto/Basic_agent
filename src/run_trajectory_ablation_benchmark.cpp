/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — G1d trajectory bucket ablation harness
 *
 * Three-arm ablation on TRAJECTORY_DISAMBIGUATES cases only.
 * Spec: docs/trajectory_ablation_benchmark.md v1.0
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_case_registry.h"
#include "../include/benchmark_context.h"
#include "../include/benchmark_runner.h"
#include "../include/config.h"
#include "../include/embedding_engine.h"
#include "../include/index_manager.h"
#include "../include/inference_backend_probe.h"
#include "../include/memory.h"
#include "../include/rag.h"
#include "../include/trajectory_ablation.h"
#include "file_handler.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string ablationLogPath() {
    FileHandler fh;
    return fh.getLogsPath("trajectory_ablation_benchmark.jsonl");
}

void appendJsonLine(const std::string& path, const nlohmann::json& event) {
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << event.dump() << '\n';
    }
}

void applyInferenceSnapshot(Thoth::BenchmarkEnvironmentInputs& inputs,
                            const Thoth::InferenceBackendSnapshot& snap) {
    inputs.inference_backend_name = snap.backend_name;
    inputs.inference_base_url = snap.base_url;
    inputs.inference_embed_base_url = snap.embed_base_url;
    inputs.inference_diagnostics = snap.diagnostics;
    inputs.inference_reachable = snap.reachable;
    // Legacy Ollama fields: populate only when exercised backend is ollama.
    if (snap.backend_name == "ollama" && snap.diagnostics.contains("ollama_version")) {
        Thoth::OllamaSnapshot ollama;
        ollama.version = snap.diagnostics.value("ollama_version", "");
        if (snap.diagnostics.contains("ollama_models") && snap.diagnostics["ollama_models"].is_array()) {
            for (const auto& row : snap.diagnostics["ollama_models"]) {
                ollama.models.emplace_back(row.value("name", ""), row.value("digest", ""));
            }
        }
        inputs.ollama = std::move(ollama);
        inputs.ollama_reachable = snap.reachable;
    }
}

Thoth::BenchmarkEnvironmentInputs makeAblationInputs(EmbeddingEngine* engine,
                                                     IndexManager* idx,
                                                     const std::vector<std::string>& corpusPaths,
                                                     const Config& config,
                                                     bool tfidfMode,
                                                     const Thoth::InferenceBackendSnapshot* snap) {
    Thoth::BenchmarkEnvironmentInputs inputs;
    inputs.harness = "trajectory_ablation_benchmark";
    // A0: authoritative External path uses FULL (backend-neutral), not OLLAMA.
    inputs.tier = tfidfMode ? Thoth::BenchmarkTier::MOCK : Thoth::BenchmarkTier::FULL;
    inputs.model.llm_model = tfidfMode ? "mock" : config.llm_model;
    inputs.model.embedding_model = tfidfMode ? "tfidf-local" : config.embedding_model;
    if (engine) {
        inputs.model.embedding_method = tfidfMode ? "TfIdf" : "External";
        inputs.model.embedding_dimension = engine->getDimension();
        inputs.model.embedding_internal_version = engine->getInternalVersion();
    }
    inputs.corpus_paths = corpusPaths;
    inputs.corpus_mode = Thoth::CorpusFingerprintMode::FAST;
    inputs.corpus_chunk_count = idx ? static_cast<int>(idx->getChunks().size()) : 0;
    inputs.thoth_env_flags = Thoth::collectThothEnvFlags();
    if (!tfidfMode && snap) {
        applyInferenceSnapshot(inputs, *snap);
    }
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

std::vector<std::string> researchCorpusFiles() {
    return {
        "agent_workspace/docs/2210.03629v3.txt",
        "agent_workspace/docs/2005.11401v4.txt",
        "agent_workspace/docs/2304.03442v2.txt",
        "agent_workspace/docs/2310.08560v2.txt",
        "agent_workspace/docs/2201.11903v6.txt",
    };
}

bool indexResearchCorpus(IndexManager& indexManager, std::vector<std::string>& indexedPaths) {
    FileHandler fh;
    const std::string sandboxPrefix =
        fs::absolute(fs::path(fh.getProjectRoot()) / "agent_workspace")
            .lexically_normal()
            .string();

    indexManager.clear();
    bool any = false;
    for (const auto& f : researchCorpusFiles()) {
        if (!fs::exists(f)) {
            std::cerr << "[WARN] Benchmark corpus file missing: " << f << '\n';
            continue;
        }
        const std::string absPath = fs::absolute(f).lexically_normal().string();
        if (absPath.rfind(sandboxPrefix, 0) != 0) {
            std::cerr << "[SECURITY ALERT] REJECTED path outside sandbox: " << absPath << '\n';
            continue;
        }
        std::cout << "[Benchmark] Indexing: " << f << "...\n";
        indexManager.indexFile(f);
        indexedPaths.push_back(absPath);
        any = true;
    }
    return any;
}

class TrajectoryAblationRunRecorder {
public:
    explicit TrajectoryAblationRunRecorder(Thoth::BenchmarkRun& run) : run_(run) {}

    ~TrajectoryAblationRunRecorder() {
        if (!finished_) {
            run_.emit("TRAJECTORY_ABLATION_ABORTED", payload());
        }
    }

    void complete(const Thoth::TrajectoryAblationSummary& summary,
                  bool sampleMode,
                  std::optional<float> tuneWt,
                  const std::string& fork) {
        summary_ = summary;
        sample_mode_ = sampleMode;
        tune_wt_ = tuneWt;
        fork_ = fork;
        run_.emit("TRAJECTORY_ABLATION_COMPLETE", payload());
        finished_ = true;
    }

private:
    nlohmann::json payload() const {
        nlohmann::json j = {{"cases_run", summary_.cases_run},
                            {"sample_mode", sample_mode_},
                            {"a_wins", summary_.a_wins},
                            {"b_wins", summary_.b_wins},
                            {"c_wins", summary_.c_wins},
                            {"ties", summary_.ties},
                            {"mean_ndcg_a", summary_.mean_ndcg_a},
                            {"mean_ndcg_b", summary_.mean_ndcg_b},
                            {"mean_ndcg_c", summary_.mean_ndcg_c},
                            {"mean_ndcg_delta_b_vs_a", summary_.mean_ndcg_delta_b_vs_a},
                            {"decision", Thoth::g1dDecisionToString(summary_.decision)},
                            {"decision_rationale", summary_.decision_rationale}};
        if (tune_wt_.has_value()) {
            j["tune_wt"] = *tune_wt_;
        }
        if (!fork_.empty()) {
            j["fork"] = fork_;
        }
        return j;
    }

    Thoth::BenchmarkRun& run_;
    Thoth::TrajectoryAblationSummary summary_;
    bool sample_mode_ = false;
    std::optional<float> tune_wt_;
    std::string fork_;
    bool finished_ = false;
};

} // namespace

int main(int argc, char** argv) {
    bool useSample = false;
    int sampleLimit = 10;
    std::optional<float> tuneWt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sample") {
            useSample = true;
            if (i + 1 < argc) {
                sampleLimit = std::max(1, std::atoi(argv[i + 1]));
                ++i;
            }
        } else if (arg == "--wt") {
            if (i + 1 >= argc) {
                std::cerr << "[FAIL] --wt requires a value (G1d TUNE: 0.05|0.1; "
                             "G1e: -0.05|-0.10|-0.20).\n";
                return 1;
            }
            tuneWt = static_cast<float>(std::atof(argv[++i]));
            if (!Thoth::isTrajectoryAblationCliWt(*tuneWt)) {
                std::cerr << "[FAIL] --wt must be G1d TUNE (0.05|0.1) or G1e "
                             "(-0.05|-0.10|-0.20). Got: "
                          << argv[i] << '\n';
                return 1;
            }
        }
    }

    const bool tfidfMode = []() {
        const char* v = std::getenv("THOTH_TRAJECTORY_ABLATION_TFIDF");
        return v && (std::string(v) == "1" || std::string(v) == "true");
    }();

    const float wtBc = tuneWt.value_or(Thoth::kTrajectoryAblationDefaultWt);
    const bool g1eFork = tuneWt.has_value() && Thoth::isTrajectoryAblationG1eWt(*tuneWt);
    const bool g1dTune = tuneWt.has_value() && Thoth::isTrajectoryAblationTuneWt(*tuneWt);
    const std::string forkLabel = g1eFork ? "g1e" : (g1dTune ? "g1d_tune" : "");

    std::cout << "G1d — Trajectory Bucket Ablation (protocol v1.0 / close-out)\n";
    if (g1eFork) {
        std::cout << "G1e polarity probe: fork=g1e tune_wt=" << *tuneWt << '\n';
    } else if (g1dTune) {
        std::cout << "Phase C TUNE micro-run: tune_wt=" << *tuneWt << '\n';
    }

    Thoth::InferenceBackendSnapshot inferenceSnap;
    if (!tfidfMode) {
        inferenceSnap = Thoth::probeInferenceBackend();
        if (!inferenceSnap.reachable) {
            std::cerr << "[FAIL] Inference backend not reachable"
                      << (inferenceSnap.backend_name.empty()
                              ? ""
                              : (" (" + inferenceSnap.backend_name + ")"))
                      << ". "
                      << (inferenceSnap.error.empty() ? "" : inferenceSnap.error + " ")
                      << "Set THOTH_TRAJECTORY_ABLATION_TFIDF=1 for TfIdf smoke.\n";
            return 1;
        }
    }

    Config config;
    Memory memory(config);

    std::unique_ptr<EmbeddingEngine> engine;
    if (tfidfMode) {
        engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf, &config);
    } else {
        engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::External, &config);
    }

    IndexManager indexManager(engine.get());
    std::vector<std::string> indexedPaths;
    if (!indexResearchCorpus(indexManager, indexedPaths)) {
        std::cerr << "[FAIL] No research corpus files indexed.\n";
        return 1;
    }

    const auto chunks = indexManager.getChunks();
    std::cout << "Corpus size: " << chunks.size() << " chunks.\n";

    Thoth::BenchmarkRun benchmarkRun = Thoth::BenchmarkRun::create(
        makeAblationInputs(engine.get(),
                           &indexManager,
                           indexedPaths,
                           config,
                           tfidfMode,
                           tfidfMode ? nullptr : &inferenceSnap));
    benchmarkRun.bindIndex(indexEnvironmentFrom(engine.get(), &indexManager));
    const Thoth::BenchmarkAttribution suiteAttribution = benchmarkRun.attribution();

    const std::string backendLabel =
        tfidfMode ? std::string("tfidf")
                  : (inferenceSnap.backend_name.empty() ? "unknown" : inferenceSnap.backend_name);

    std::cout << "BENCHMARK_ENV run_id=" << benchmarkRun.run_id()
              << " env_hash=" << benchmarkRun.environment_hash()
              << " index_hash=" << benchmarkRun.index_hash()
              << " backend=" << backendLabel;
    if (tuneWt.has_value()) {
        std::cout << " tune_wt=" << *tuneWt;
    }
    if (!forkLabel.empty()) {
        std::cout << " fork=" << forkLabel;
    }
    std::cout << '\n';

    TrajectoryAblationRunRecorder suiteRecorder(benchmarkRun);

    if (const char* abortSmoke = std::getenv("THOTH_TRAJECTORY_ABLATION_ABORT_SMOKE");
        abortSmoke && (std::string(abortSmoke) == "1" || std::string(abortSmoke) == "true")) {
        std::cerr << "TRAJECTORY_ABLATION: abort smoke — exiting before complete()\n";
        return 2;
    }

    auto cases = Thoth::filterTrajectoryDisambiguatesCases(Thoth::BenchmarkCaseRegistry::getCases());
    std::cout << "TRAJECTORY_DISAMBIGUATES cases: " << cases.size() << '\n';

    if (cases.empty()) {
        std::cerr << "[FAIL] No trajectory-disambiguation cases in registry.\n";
        return 1;
    }

    if (useSample && static_cast<int>(cases.size()) > sampleLimit) {
        std::cout << "[INFO] Sampling: running " << sampleLimit << " of " << cases.size() << " cases.\n";
        cases.resize(static_cast<std::size_t>(sampleLimit));
    }

    RAGPipeline rag(std::move(engine), &indexManager, &config, &memory);
    Thoth::BenchmarkRunner runner(rag);

    std::cout << "Running arms A (wt=0), B (wt=" << wtBc << "), C (wt=" << wtBc << ", empty T)...\n";

    const Thoth::BenchmarkResult result_a =
        runner.run(Thoth::trajectoryAblationArmConfig(Thoth::TrajectoryAblationArm::A, wtBc), cases);
    const Thoth::BenchmarkResult result_b =
        runner.run(Thoth::trajectoryAblationArmConfig(Thoth::TrajectoryAblationArm::B, wtBc), cases);
    const Thoth::BenchmarkResult result_c =
        runner.run(Thoth::trajectoryAblationArmConfig(Thoth::TrajectoryAblationArm::C, wtBc), cases);

    const Thoth::TrajectoryAblationSummary summary =
        Thoth::computeTrajectoryAblationSummary(cases, result_a, result_b, result_c);

    const std::string logPath = ablationLogPath();
    const std::int64_t ts = nowMs();

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const float ndcg_a = result_a.cases[i].ndcg_at_k;
        const float ndcg_b = result_b.cases[i].ndcg_at_k;
        const float ndcg_c = result_c.cases[i].ndcg_at_k;
        const std::string winner = Thoth::computeTrajectoryAblationWinner(ndcg_a, ndcg_b, ndcg_c);

        std::cout << "  " << cases[i].case_id << " winner=" << winner << " ndcg A/B/C=" << ndcg_a
                  << '/' << ndcg_b << '/' << ndcg_c << '\n';

        nlohmann::json caseRow = {
            {"event", "TRAJECTORY_ABLATION_CASE"},
            {"timestamp_ms", ts},
            {"run_id", suiteAttribution.run_id},
            {"env_hash", suiteAttribution.env_hash},
            {"backend", backendLabel},
            {"case_id", cases[i].case_id},
            {"case_type", cases[i].case_type},
            {"winner", winner},
            {"ndcg_a", ndcg_a},
            {"ndcg_b", ndcg_b},
            {"ndcg_c", ndcg_c},
            {"precision_a", result_a.cases[i].precision_at_k},
            {"precision_b", result_b.cases[i].precision_at_k},
            {"precision_c", result_c.cases[i].precision_at_k},
        };
        if (tuneWt.has_value()) {
            caseRow["tune_wt"] = *tuneWt;
        }
        if (!forkLabel.empty()) {
            caseRow["fork"] = forkLabel;
        }
        appendJsonLine(logPath, caseRow);
    }

    nlohmann::json summaryRow = {
        {"event", "TRAJECTORY_ABLATION_SUMMARY"},
        {"timestamp_ms", ts},
        {"run_id", suiteAttribution.run_id},
        {"env_hash", suiteAttribution.env_hash},
        {"backend", backendLabel},
        {"cases_run", summary.cases_run},
        {"sample_mode", useSample},
        {"a_wins", summary.a_wins},
        {"b_wins", summary.b_wins},
        {"c_wins", summary.c_wins},
        {"ties", summary.ties},
        {"b_wins_vs_a", summary.b_wins_vs_a},
        {"a_wins_vs_b", summary.a_wins_vs_b},
        {"mean_ndcg_a", summary.mean_ndcg_a},
        {"mean_ndcg_b", summary.mean_ndcg_b},
        {"mean_ndcg_c", summary.mean_ndcg_c},
        {"mean_ndcg_delta_b_vs_a", summary.mean_ndcg_delta_b_vs_a},
        {"decision", Thoth::g1dDecisionToString(summary.decision)},
        {"decision_rationale", summary.decision_rationale},
        {"wt_bc", wtBc},
    };
    if (tuneWt.has_value()) {
        summaryRow["tune_wt"] = *tuneWt;
    }
    if (!forkLabel.empty()) {
        summaryRow["fork"] = forkLabel;
    }
    appendJsonLine(logPath, summaryRow);

    std::cout << "\nSummary\n";
    std::cout << "  A wins: " << summary.a_wins << "  B wins: " << summary.b_wins
              << "  C wins: " << summary.c_wins << "  Ties: " << summary.ties << '\n';
    std::cout << "  B vs A: " << summary.b_wins_vs_a << " wins / " << summary.a_wins_vs_b
              << " losses (non-tie)\n";
    std::cout << "  mean nDCG A/B/C: " << summary.mean_ndcg_a << " / " << summary.mean_ndcg_b
              << " / " << summary.mean_ndcg_c << '\n';
    std::cout << "  mean delta (B-A): " << summary.mean_ndcg_delta_b_vs_a << '\n';
    std::cout << "  decision: " << Thoth::g1dDecisionToString(summary.decision) << " — "
              << summary.decision_rationale << '\n';
    std::cout << "  backend: " << backendLabel << '\n';
    if (tuneWt.has_value()) {
        std::cout << "  tune_wt: " << *tuneWt << '\n';
    }
    if (!forkLabel.empty()) {
        std::cout << "  fork: " << forkLabel << '\n';
    }
    std::cout << "  log: " << logPath << '\n';

    if (!tfidfMode && summary.decision == Thoth::G1dDecision::PENDING) {
        std::cerr << "[WARN] Decision still PENDING — unexpected.\n";
    }

    suiteRecorder.complete(summary, useSample, tuneWt, forkLabel);

    if (!tfidfMode && useSample) {
        std::cout << "[NOTE] Sample run — decision is indicative only; full ~30-case "
                     "authoritative run required for close-out evidence.\n";
    }
    if (g1eFork && !useSample && !tfidfMode) {
        std::cout << "[NOTE] G1e polarity run — KEEP / no-scalar-rescue awaits owner review "
                     "(production config unchanged; see G1E_POLARITY_PROTOCOL.md).\n";
    } else if (g1dTune && !useSample && !tfidfMode) {
        std::cout << "[NOTE] Phase C TUNE micro-run — terminal KEEP/DROP awaits Checkpoint "
                     "C-TUNE owner review (no production config change yet).\n";
    }

    return 0;
}
