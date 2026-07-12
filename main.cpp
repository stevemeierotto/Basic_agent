#include "command_processor.h"
#include "memory.h"
#include "rag.h"
#include "llm_interface.h"
#include "runtime_bootstrap.h"
#include "embedding_engine.h"
#include "config.h"
#include "similarity.h"
#include "file_handler.h"

#include <iostream>
int main() {
    Thoth::bootstrapRuntimeEnvironment();

    FileHandler fileHandler;

    // 1. Load config from JSON
    Config agentConfig;
    const std::string configPath = fileHandler.getConfigPath();
    agentConfig.loadFromJson(configPath);

    Thoth::logResolvedRuntimeConfig(&agentConfig);

    // 2. Core objects
    Memory memory;
    LLMInterface llm(LLMBackend::Ollama, &agentConfig);

    // 4. Embedding engine and index manager
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    IndexManager indexManager(engine.get());
    
    // 4a. Initialize similarity metric from config
    auto similarity = createSimilarity(agentConfig.similarity_metric);
    indexManager.store.setSimilarity(std::move(similarity));
    std::cout << "Similarity metric set to: " << agentConfig.similarity_metric << "\n";

    // 5. RAG pipeline with ownership of engine
    RAGPipeline rag(std::move(engine), &indexManager, &agentConfig);

    // 6. Command processor
    CommandProcessor cp(memory, rag, llm, &agentConfig);
    cp.runLoop();

    return 0;
}

