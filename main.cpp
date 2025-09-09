#include "include/command_processor.h"
#include "include/memory.h"
#include "include/rag.h"
#include "include/llm_interface.h"

#include <filesystem>

int main() {
    std::string absPath = (std::filesystem::current_path() / "memory.json").string();
    Memory memory(absPath);
    RAGPipeline rag;
    LLMInterface llm;

    CommandProcessor cp(memory, rag, llm);
    cp.runLoop();
    return 0;
}


