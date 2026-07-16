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

/** Plan M G1 (R1) — stats from applying the fail-closed grounding floor. */
struct GroundingFloorStats {
    int candidates_found = 0;       // pre-floor candidate count
    int candidates_passed_gate = 0; // post-floor (injected) count
    bool has_candidates = false;    // true when at least one candidate had a finite score
    float max_score = 0.0f;         // max finite candidate score (valid iff has_candidates)
    float min_injected_score = 0.0f;// min injected score (valid iff candidates_passed_gate > 0)
};

/** Plan M G1 (R1) — result of applying the grounding floor. */
struct GroundingFloorResult {
    std::vector<CodeChunk> injectable;   // chunks that passed the floor, original order
    GragDiagnostics diagnostics;         // filtered + aligned with `injectable`
    GroundingFloorStats stats;
};

/**
 * Fail-closed relevance floor (Plan M G1 / R1).
 *
 * Keeps chunks whose aligned post-boost final_score is finite and >= minFinalScore.
 * Missing or NaN scores are rejected (fail closed). The returned diagnostics are a
 * copy of `diagnostics` with breakdowns/final_scores filtered to match `injectable`,
 * so downstream metric builders stay index-aligned.
 *
 * This is a floor to keep zero / near-zero / broken scores from grounding — not a
 * calibrated meaningful-relevance gate. Greeting handling and any stronger threshold
 * are deferred to later Plan M checkpoints.
 */
GroundingFloorResult applyGroundingFloor(
    const std::vector<CodeChunk>& chunks,
    const GragDiagnostics& diagnostics,
    float minFinalScore);

} // namespace ChatRetrieval
} // namespace Thoth

#endif // THOTH_CHAT_RETRIEVAL_BOOST_H
