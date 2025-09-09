#include "include/command_processor.h"
#include "include/memory.h"
#include "include/rag.h"
#include "include/llm_interface.h"

int main() {
    Memory memory;
    RAGPipeline rag;
    LLMInterface llm;

    CommandProcessor cp(memory, rag, llm);
    cp.runLoop();
    return 0;
}
