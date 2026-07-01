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

namespace Thoth {

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

    try {
        const auto raw = input.index->retrieveChunks(input.query, input.top_k * 4);
        for (const auto& [code, score] : raw) {
            if (result.chunks.size() >= static_cast<std::size_t>(input.top_k)) {
                break;
            }
            const auto* chunk = input.index->getChunkByCode(code);
            if (!chunk) {
                continue;
            }
            RetrievedChunkRecord rec;
            rec.chunk_id = chunk->fileName;
            rec.source = RetrievedChunkSource::CORPUS;
            rec.source_id = chunk->fileName;
            rec.content = chunk->code;
            rec.validation_status = ProvenanceValidationStatus::VALID;
            result.chunks.push_back(std::move(rec));
        }

        for (const auto& episode : input.episode_log->entries()) {
            if (result.chunks.size() >= static_cast<std::size_t>(input.top_k)) {
                break;
            }
            RetrievedChunkRecord rec;
            rec.chunk_id = "episode:" + episode.episode_id;
            rec.source = RetrievedChunkSource::EVALUATION;
            rec.source_id = episode.episode_id;
            rec.content = episode.content;
            rec.validation_status =
                episode.content_hash.empty() ? ProvenanceValidationStatus::UNTRACED
                                             : ProvenanceValidationStatus::VALID;
            result.chunks.push_back(std::move(rec));
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
