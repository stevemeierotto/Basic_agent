/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — run_grag_benchmark tool
 * Standalone utility to execute the GRAG vs RAG comparison.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/config.h"
#include "../include/memory.h"
#include "../include/rag.h"
#include "../include/index_manager.h"
#include "../include/embedding_engine.h"
#include "../include/benchmark_runner.h"
#include "../include/benchmark_reporter.h"
#include "../include/benchmark_case_registry.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    bool useSample = false;
    if (argc > 1 && std::string(argv[1]) == "--sample") {
        useSample = true;
    }

    std::cout << "Initializing Research Paper Benchmark Environment...\n";

    Config config;
    Memory memory(config);
    
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::External, &config);
    IndexManager indexManager(engine.get());
    
    // STRICT SANDBOX BOUNDARY: The benchmark MUST only read from agent_workspace/docs/
    std::cout << "Indexing research paper corpus only...\n";
    indexManager.clear();
    
    std::vector<std::string> corpusFiles = {
        "agent_workspace/docs/2210.03629v3.txt",
        "agent_workspace/docs/2005.11401v4.txt",
        "agent_workspace/docs/2304.03442v2.txt",
        "agent_workspace/docs/2310.08560v2.txt",
        "agent_workspace/docs/2201.11903v6.txt"
    };

    for (const auto& f : corpusFiles) {
        if (fs::exists(f)) {
            // Hard reject prefix check
            std::string absPath = fs::absolute(f).lexically_normal().string();
            if (absPath.find("/home/steve/Thoth/agent_workspace/") == std::string::npos) {
                std::cerr << "[SECURITY ALERT] REJECTED path outside sandbox: " << absPath << "\n";
                continue;
            }
            
            std::cout << "[Benchmark] Indexing: " << f << "...\n";
            indexManager.indexFile(f);
            std::cout << "[Benchmark] Total chunks so far: " << indexManager.getChunks().size() << "\n";
        } else {
            std::cerr << "[WARN] Benchmark corpus file missing: " << f << "\n";
        }
    }

    auto chunks = indexManager.getChunks();
    std::cout << "Corpus size: " << chunks.size() << " chunks (STRICTLY SANDBOXED).\n";

    RAGPipeline rag(std::move(engine), &indexManager, &config, &memory);
    
    std::cout << "Loading 100 Hardened Research Paper Test Cases...\n";
    auto cases = Thoth::BenchmarkCaseRegistry::getCases();

    if (useSample && cases.size() > 10) {
        std::cout << "[INFO] Sampling active: running 10 out of " << cases.size() << " cases.\n";
        cases.resize(10);
    }

    Thoth::BenchmarkRunner runner(rag);

    std::cout << "Executing RAG vs GRAG Comparison (Research Corpus)...\n";
    auto result = runner.runComparison(cases);

    std::cout << "Saving results to grag_benchmark.jsonl...\n";
    Thoth::BenchmarkReporter::reportToFile(result, static_cast<int>(chunks.size()));

    std::cout << "Printing Summary:\n";
    Thoth::BenchmarkReporter::reportToStdout(result);

    return 0;
}
