/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — run_grag_benchmark tool
 * Standalone utility to execute the GRAG vs RAG comparison.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/benchmark_case_registry.h"
#include "../include/benchmark_context.h"
#include "../include/benchmark_reporter.h"
#include "../include/benchmark_runner.h"
#include "../include/config.h"
#include "../include/embedding_engine.h"
#include "../include/index_manager.h"
#include "../include/memory.h"
#include "../include/ollama_snapshot.h"
#include "../include/rag.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

Thoth::BenchmarkEnvironmentInputs makeGragBenchmarkInputs(
    EmbeddingEngine* engine,
    IndexManager* idx,
    const std::vector<std::string>& indexedCorpusPaths,
    const Config& config) {
    Thoth::BenchmarkEnvironmentInputs inputs;
    inputs.harness = "grag_benchmark";
    inputs.tier = Thoth::BenchmarkTier::OLLAMA;
    inputs.model.llm_model = config.llm_model;
    inputs.model.embedding_model = config.embedding_model;
    if (engine) {
        inputs.model.embedding_method = "External";
        inputs.model.embedding_dimension = engine->getDimension();
        inputs.model.embedding_internal_version = engine->getInternalVersion();
    }
    inputs.corpus_paths = indexedCorpusPaths;
    inputs.corpus_mode = Thoth::CorpusFingerprintMode::FAST;
    inputs.corpus_chunk_count = idx ? static_cast<int>(idx->getChunks().size()) : 0;
    inputs.thoth_env_flags = Thoth::collectThothEnvFlags();
    inputs.ollama_reachable = Thoth::isOllamaReachable();
    if (inputs.ollama_reachable) {
        if (auto snap = Thoth::fetchOllamaSnapshot()) {
            inputs.ollama = *snap;
        }
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

/** RAII: emit GRAG_BENCHMARK_COMPLETE on normal exit, GRAG_BENCHMARK_ABORTED if scope exits early. */
class GragBenchmarkRunRecorder {
public:
    explicit GragBenchmarkRunRecorder(Thoth::BenchmarkRun& run) : run_(run) {}

    ~GragBenchmarkRunRecorder() {
        if (!finished_) {
            run_.emit("GRAG_BENCHMARK_ABORTED", payload());
        }
    }

    void complete(const Thoth::ComparisonResult& result,
                  std::size_t casesRun,
                  bool sampleMode) {
        cases_run_ = casesRun;
        sample_mode_ = sampleMode;
        rag_mean_ndcg_ = result.rag_mean_ndcg;
        grag_mean_ndcg_ = result.grag_mean_ndcg;

        int grag_wins = 0;
        int rag_wins = 0;
        for (const auto& delta : result.deltas) {
            if (delta.ndcg_delta > 0.001f) {
                ++grag_wins;
            } else if (delta.ndcg_delta < -0.001f) {
                ++rag_wins;
            }
        }
        grag_wins_ndcg_ = grag_wins;
        rag_wins_ndcg_ = rag_wins;

        run_.emit("GRAG_BENCHMARK_COMPLETE", payload());
        finished_ = true;
    }

private:
    nlohmann::json payload() const {
        return {{"cases_run", cases_run_},
                {"sample_mode", sample_mode_},
                {"rag_mean_ndcg_at_5", rag_mean_ndcg_},
                {"grag_mean_ndcg_at_5", grag_mean_ndcg_},
                {"grag_wins_ndcg", grag_wins_ndcg_},
                {"rag_wins_ndcg", rag_wins_ndcg_}};
    }

    Thoth::BenchmarkRun& run_;
    std::size_t cases_run_ = 0;
    bool sample_mode_ = false;
    float rag_mean_ndcg_ = 0.0f;
    float grag_mean_ndcg_ = 0.0f;
    int grag_wins_ndcg_ = 0;
    int rag_wins_ndcg_ = 0;
    bool finished_ = false;
};

} // namespace

int main(int argc, char** argv) {
    bool useSample = false;
    if (argc > 1 && std::string(argv[1]) == "--sample") {
        useSample = true;
    }

    std::cout << "Initializing Research Paper Benchmark Environment...\n";

    if (!Thoth::isOllamaReachable()) {
        std::cerr << "[FAIL] Ollama not reachable at localhost:11434 (embeddings required).\n";
        return 1;
    }

    Config config;
    Memory memory(config);

    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::External, &config);
    IndexManager indexManager(engine.get());

    std::cout << "Indexing research paper corpus only...\n";
    indexManager.clear();

    const std::vector<std::string> corpusFiles = {
        "agent_workspace/docs/2210.03629v3.txt",
        "agent_workspace/docs/2005.11401v4.txt",
        "agent_workspace/docs/2304.03442v2.txt",
        "agent_workspace/docs/2310.08560v2.txt",
        "agent_workspace/docs/2201.11903v6.txt",
    };

    std::vector<std::string> indexedCorpusPaths;
    for (const auto& f : corpusFiles) {
        if (fs::exists(f)) {
            const std::string absPath = fs::absolute(f).lexically_normal().string();
            if (absPath.find("/home/steve/Thoth/agent_workspace/") == std::string::npos) {
                std::cerr << "[SECURITY ALERT] REJECTED path outside sandbox: " << absPath << std::endl;
                continue;
            }

            std::cout << "[Benchmark] Indexing: " << f << "...\n";
            indexManager.indexFile(f);
            indexedCorpusPaths.push_back(absPath);
            std::cout << "[Benchmark] Total chunks so far: " << indexManager.getChunks().size() << std::endl;
        } else {
            std::cerr << "[WARN] Benchmark corpus file missing: " << f << std::endl;
        }
    }

    const auto chunks = indexManager.getChunks();
    std::cout << "Corpus size: " << chunks.size() << " chunks (STRICTLY SANDBOXED).\n";

    Thoth::BenchmarkRun benchmarkRun = Thoth::BenchmarkRun::create(
        makeGragBenchmarkInputs(engine.get(), &indexManager, indexedCorpusPaths, config));
    benchmarkRun.bindIndex(indexEnvironmentFrom(engine.get(), &indexManager));
    const Thoth::BenchmarkRunIdentity suiteIdentity{
        benchmarkRun.run_id(),
        benchmarkRun.environment_hash(),
    };

    std::cout << "BENCHMARK_ENV run_id=" << benchmarkRun.run_id()
              << " env_hash=" << benchmarkRun.environment_hash()
              << " index_hash=" << benchmarkRun.index_hash() << " tier=ollama\n";

    GragBenchmarkRunRecorder suiteRecorder(benchmarkRun);

    if (const char* abortSmoke = std::getenv("THOTH_GRAG_BENCHMARK_ABORT_SMOKE");
        abortSmoke && (std::string(abortSmoke) == "1" || std::string(abortSmoke) == "true")) {
        std::cerr << "GRAG: benchmark abort smoke — exiting before complete()\n";
        return 2;
    }

    RAGPipeline rag(std::move(engine), &indexManager, &config, &memory);

    std::cout << "Loading 100 Hardened Research Paper Test Cases...\n";
    auto cases = Thoth::BenchmarkCaseRegistry::getCases();

    if (useSample && cases.size() > 10) {
        std::cout << "[INFO] Sampling active: running 10 out of " << cases.size() << " cases.\n";
        cases.resize(10);
    }

    Thoth::BenchmarkRunner runner(rag);

    std::cout << "Executing RAG vs GRAG Comparison (Research Corpus)...\n";
    const auto result = runner.runComparison(cases);

    std::cout << "Saving results to grag_benchmark.jsonl...\n";
    Thoth::BenchmarkReporter::reportToFile(result, static_cast<int>(chunks.size()), suiteIdentity);

    std::cout << "Printing Summary:\n";
    Thoth::BenchmarkReporter::reportToStdout(result);

    suiteRecorder.complete(result, cases.size(), useSample);
    return 0;
}
