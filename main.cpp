#include "include/command_processor.h"
#include "include/memory.h"
#include "include/rag.h"
#include "include/llm_interface.h"
#include "include/env_loader.h"

#include <filesystem>
#include <iostream>

int main() {
    //std::string absPath = (std::filesystem::current_path() / "memory.json").string();
    Memory memory;
    RAGPipeline rag;
    LLMInterface llm;

    if (!EnvLoader::loadEnvFile("../.env")) {
        std::cerr << "Warning: .env file not found. Using system environment variables.\n";
    }

    CommandProcessor cp(memory, rag, llm);
    cp.runLoop();
    return 0;
}


