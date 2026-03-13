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
#include "../include/file_handler.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::cout << "Initializing Sandbox Benchmark Environment...\n";

    Config config;
    Memory memory(config);
    FileHandler fh;
    
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::External, &config);
    IndexManager indexManager(engine.get());
    
    // STRICT SANDBOX BOUNDARY: The benchmark MUST only read from agent_workspace/rag/docs/
    std::cout << "Indexing sandboxed benchmark corpus only...\n";
    indexManager.clear();
    
    // We only index the docs inside the sandbox for the benchmark.
    // The test cases now exclusively point to these files.
    std::vector<std::string> corpusFiles = {
        "agent_workspace/rag/docs/GRAG.md",
        "agent_workspace/rag/docs/PLAN.md",
        "agent_workspace/rag/docs/cognate.md",
        "agent_workspace/rag/docs/improvements.md",
        "agent_workspace/rag/docs/NODE.md",
        "agent_workspace/rag/docs/architectural_facts.md",
        "agent_workspace/rag/docs/completed_improvements_log.md",
        "agent_workspace/rag/docs/AGENTS.md"
    };

    for (const auto& f : corpusFiles) {
        if (fs::exists(f)) {
            // Hard reject: double check absolute path before indexing
            std::string absPath = fs::absolute(f).lexically_normal().string();
            if (absPath.find("/home/steve/Thoth/agent_workspace/") == std::string::npos) {
                std::cerr << "[SECURITY ALERT] REJECTED path outside sandbox: " << absPath << "\n";
                continue;
            }
            indexManager.indexFile(f);
        } else {
            std::cerr << "[WARN] Benchmark corpus file missing: " << f << "\n";
        }
    }

    auto chunks = indexManager.getChunks();
    std::cout << "Corpus size: " << chunks.size() << " chunks (STRICTLY SANDBOXED).\n";

    RAGPipeline rag(std::move(engine), &indexManager, &config, &memory);
    
    std::cout << "Loading 15 Rewritten Sandboxed Test Cases...\n";
    auto cases = Thoth::BenchmarkCaseRegistry::getCases();

    Thoth::BenchmarkRunner runner(rag);

    std::cout << "Executing RAG vs GRAG Comparison...\n";
    auto result = runner.runComparison(cases);

    std::cout << "Saving results to grag_benchmark.jsonl...\n";
    Thoth::BenchmarkReporter::reportToFile(result, static_cast<int>(chunks.size()));

    std::cout << "Printing Summary:\n";
    Thoth::BenchmarkReporter::reportToStdout(result);

    return 0;
}
