/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 1 golden chat-RAG retrieval cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_RAG_GOLDEN_CASES_H
#define THOTH_CHAT_RAG_GOLDEN_CASES_H

#include <string>
#include <vector>

namespace Thoth {

struct ChatRagGoldenCase {
    std::string id;
    std::string query;
    /** Basename of the document expected at rank #1 (e.g. GRAG.md). */
    std::string expected_top_file;
};

/** Golden queries for document-level retrieval evaluation (C2 Phase 1). */
std::vector<ChatRagGoldenCase> getChatRagGoldenCases();

/** Markdown corpus files indexed before running golden queries (sandbox paths only). */
std::vector<std::string> getChatRagGoldenCorpusPaths();

} // namespace Thoth

#endif // THOTH_CHAT_RAG_GOLDEN_CASES_H
