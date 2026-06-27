/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Golden chat-RAG case registry
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_rag_golden_cases.h"
#include "file_handler.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace Thoth {

std::vector<ChatRagGoldenCase> getChatRagGoldenCases() {
    return {
        {"C2-01", "Explain GRAG.", "GRAG.md"},
        {"C2-02", "What is Cognate?", "cognate.md"},
        {"C2-03", "How do I use Thoth?", "HOWTO.md"},
        {"C2-04", "What are agent conventions?", "AGENTS.md"},
        {"C2-05", "Quote the first sentence of GRAG.md.", "GRAG.md"},
    };
}

std::vector<std::string> getChatRagGoldenCorpusPaths() {
    FileHandler fh;
    const fs::path ragDir = fs::path(fh.getRagDirectory());
    const std::vector<const char*> files = {
        "GRAG.md",
        "cognate.md",
        "HOWTO.md",
        "AGENTS.md",
    };

    std::vector<std::string> paths;
    paths.reserve(files.size());
    for (const char* name : files) {
        const fs::path path = ragDir / name;
        if (fs::exists(path)) {
            paths.push_back(fs::absolute(path).lexically_normal().string());
        }
    }
    return paths;
}

} // namespace Thoth
