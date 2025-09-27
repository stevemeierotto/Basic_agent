#include "command_processor.h"
#include "memory.h"
#include "rag.h"
#include "llm_interface.h"
#include "env_loader.h"
#include "embedding_engine.h"
#include "config.h"

#include <iostream>
int main() {
    // 1. Load config from JSON
    Config agentConfig;
    agentConfig.loadFromJson("config.json");

    // 2. Load .env file early so environment is set for all components
    if (!EnvLoader::loadEnvFile("../.env")) {
        std::cerr << "Warning: .env file not found. Using system environment variables.\n";
    }

    // 3. Core objects
    Memory memory;
    LLMInterface llm(LLMBackend::Ollama, &agentConfig);

    // 4. Embedding engine and index manager
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    IndexManager indexManager(engine.get());

    // 5. RAG pipeline with ownership of engine
    RAGPipeline rag(std::move(engine), &indexManager, &agentConfig);

    // 6. Command processor
    CommandProcessor cp(memory, rag, llm, &agentConfig);
    cp.runLoop();

    return 0;
}

