/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 STRICT deterministic retrieval (evaluation kernel only)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/e2_strict_retrieval.h"

#include "../include/index_manager.h"
#include "../include/embedding_engine.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_set>

namespace Thoth {

namespace {

std::vector<std::string> strictTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            current += static_cast<char>(std::tolower(c));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

float strictEpisodeRelevanceScore(const std::string& query, const std::string& text) {
    const auto queryTokens = strictTokenize(query);
    const auto textTokens = strictTokenize(text);
    if (queryTokens.empty() || textTokens.empty()) {
        return 0.f;
    }

    std::unordered_set<std::string> querySet(queryTokens.begin(), queryTokens.end());
    int shared = 0;
    for (const auto& token : textTokens) {
        if (querySet.count(token) > 0) {
            ++shared;
        }
    }
    if (shared == 0) {
        return 0.f;
    }

    return static_cast<float>(shared) /
           std::sqrt(static_cast<float>(querySet.size()) *
                     static_cast<float>(textTokens.size()));
}

struct RankedChunk {
    RetrievedChunkRecord record;
    float score = 0.f;
};

bool rankedBefore(const RankedChunk& a, const RankedChunk& b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    return a.record.chunk_id < b.record.chunk_id;
}

/** Episodes below this overlap score are excluded (E2-03 disambiguation).
 *  PROVISIONAL — see docs/E2_PROTOCOL.md § STRICT kernel scoring (A3).
 *  Calibrated on E2-01–E2-03 only (~0.17 vs ~0.83); not B1-validated. */
constexpr float kStrictEpisodeInclusionMinScore = 0.25f;

} // namespace

E2StrictRetrievalResult e2StrictRetrieve(const E2StrictRetrievalInput& input) {
    E2StrictRetrievalResult result;

    if (input.config.tier != E2EvalTier::STRICT) {
        result.status = E2ArmScoringStatus::FAILED_STRICT_BOUNDARY;
        result.error_message = "e2StrictRetrieve requires STRICT tier";
        return result;
    }

    if (!input.episode_log || !input.episode_log->isSealed()) {
        result.status = E2ArmScoringStatus::FAILED_STRICT_BOUNDARY;
        result.error_message = "episode log not sealed";
        return result;
    }

    if (!input.index || !input.engine) {
        result.status = E2ArmScoringStatus::FAILED_RETRIEVAL;
        result.error_message = "index or engine unavailable";
        return result;
    }

    if (input.top_k <= 0) {
        result.status = E2ArmScoringStatus::FAILED_RETRIEVAL;
        result.error_message = "invalid top_k";
        return result;
    }

    try {
        std::vector<RankedChunk> ranked;
        const int recallK = std::max(input.top_k * 4, input.top_k);
        const auto corpusHits = input.index->retrieveChunks(input.query, recallK);

        for (const auto& [code, score] : corpusHits) {
            const auto* chunk = input.index->getChunkByCode(code);
            if (!chunk) {
                continue;
            }
            RankedChunk entry;
            entry.score = score;
            entry.record.chunk_id = chunk->fileName;
            entry.record.source = RetrievedChunkSource::CORPUS;
            entry.record.source_id = chunk->fileName;
            entry.record.content = chunk->code;
            entry.record.validation_status = ProvenanceValidationStatus::VALID;
            ranked.push_back(std::move(entry));
        }

        for (const auto& episode : input.episode_log->entries()) {
            const float episodeScore =
                strictEpisodeRelevanceScore(input.query, episode.content);
            if (episodeScore < kStrictEpisodeInclusionMinScore) {
                continue;
            }
            RankedChunk entry;
            entry.score = episodeScore;
            entry.record.chunk_id = "episode:" + episode.episode_id;
            entry.record.source = RetrievedChunkSource::EVALUATION;
            entry.record.source_id = episode.episode_id;
            entry.record.content = episode.content;
            entry.record.validation_status =
                episode.content_hash.empty() ? ProvenanceValidationStatus::UNTRACED
                                             : ProvenanceValidationStatus::VALID;
            ranked.push_back(std::move(entry));
        }

        std::sort(ranked.begin(), ranked.end(), rankedBefore);

        const std::size_t limit = static_cast<std::size_t>(input.top_k);
        result.chunks.reserve(std::min(limit, ranked.size()));
        for (std::size_t i = 0; i < ranked.size() && i < limit; ++i) {
            result.chunks.push_back(std::move(ranked[i].record));
        }
    } catch (const std::exception& e) {
        result.status = E2ArmScoringStatus::FAILED_RETRIEVAL;
        result.error_message = e.what();
        result.chunks.clear();
        return result;
    }

    return result;
}

} // namespace Thoth
