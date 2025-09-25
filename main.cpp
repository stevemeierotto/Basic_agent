#include "command_processor.h"
#include "memory.h"
#include "rag.h"
#include "llm_interface.h"
#include "env_loader.h"
#include "embedding_engine.h"
#include "config.h"

#include <iostream>
int main() {
    Memory memory;
    LLMInterface llm;
    Config agentConfig;
    agentConfig.loadFromJson("config.json");

    if (!EnvLoader::loadEnvFile("../.env")) {
        std::cerr << "Warning: .env file not found. Using system environment variables.\n";
    }

    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);

    // Create IndexManager with a non-owning pointer to engine
    IndexManager indexManager(engine.get());

    // Pass both engine (ownership) and indexManager pointer to RAGPipeline
    RAGPipeline rag(std::move(engine), &indexManager, &agentConfig);

    // Now use rag and indexManager as needed
    //CommandProcessor cp(memory, rag, llm);
    CommandProcessor cp(memory, rag, llm, &agentConfig);
    cp.runLoop();
    return 0;
}
