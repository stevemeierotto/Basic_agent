/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 2 conversational retrieval boosts
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_RETRIEVAL_BOOST_H
#define THOTH_CHAT_RETRIEVAL_BOOST_H

#include "chunkers/code_chunk.h"
#include "grag_diagnostics.h"
#include <string>
#include <utility>
#include <vector>

class IndexManager;

namespace Thoth {
namespace ChatRetrieval {

bool isDefinitionalQuery(const std::string& query);

bool isQuoteQuery(const std::string& query);

bool isUsageQuery(const std::string& query);

/** Salient tokens for filename matching (e.g. GRAG, cognate). */
std::vector<std::string> extractFilenameTokens(const std::string& query);

bool filenameMatchesToken(const std::string& filePath, const std::string& token);

/** Inject substantive chunks from filename-matched docs missing from recall. */
void ensureFilenameCoverage(IndexManager* indexManager,
                            const std::vector<std::string>& tokens,
                            const std::string& query,
                            std::vector<std::pair<CodeChunk, float>>& ragResults,
                            int minPerFile = 2);

/** Re-score and re-sort conversational retrieval candidates (no goal embedding). */
void applyConversationalBoosts(std::vector<std::pair<CodeChunk, float>>& ranked,
                               const std::string& query,
                               GragDiagnostics& diagnostics);

/** Take top-K chunks meeting minimum size; pull deeper if top ranks are tiny headers. */
std::vector<std::pair<CodeChunk, float>> selectTopKForInjection(
    const std::vector<std::pair<CodeChunk, float>>& ranked,
    int topK,
    std::size_t minChunkChars,
    GragDiagnostics& diagnostics);

/** Format a chunk with document metadata for LLM context (injection-time only). */
std::string formatChunkForPrompt(const CodeChunk& chunk);

} // namespace ChatRetrieval
} // namespace Thoth

#endif // THOTH_CHAT_RETRIEVAL_BOOST_H
