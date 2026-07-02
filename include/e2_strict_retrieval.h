/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 STRICT deterministic retrieval (evaluation kernel only)
 *
 * Protocol: docs/E2_PROTOCOL.md v1.2 § Link isolation
 *
 * This header is the ONLY retrieval entry point permitted in E2-EVAL-STRICT.
 * Runtime heuristics (rag.cpp warm merge, token overlap, cross-session) are
 * NOT linked into targets that compile e2_strict_retrieval.cpp.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_E2_STRICT_RETRIEVAL_H
#define THOTH_E2_STRICT_RETRIEVAL_H

#include "e2_strict_enforcement.h"
#include "episodic_learning_eval.h"

#include <string>
#include <vector>

class IndexManager;
class EmbeddingEngine;

namespace Thoth {

struct E2StrictRetrievalInput {
    std::string query;
    const SealedEpisodeInjectionLog* episode_log = nullptr;
    E2EvalConfig config;
    IndexManager* index = nullptr;
    EmbeddingEngine* engine = nullptr;
    int top_k = 5;
};

struct E2StrictRetrievalResult {
    std::vector<RetrievedChunkRecord> chunks;
    E2ArmScoringStatus status = E2ArmScoringStatus::OK;
    std::string error_message;
};

/**
 * Deterministic STRICT retrieval: f(query, corpus_snapshot, frozen_episode_log).
 * Fail closed on error — no partial chunks.
 *
 * Pure function invariant (A3+): no writes, global state, caches, SQLite, Executive, or RAG.
 * Harness wires this at the evaluation boundary; executive diagnostics are non-authoritative until A4.
 */
E2StrictRetrievalResult e2StrictRetrieve(const E2StrictRetrievalInput& input);

} // namespace Thoth

#endif // THOTH_E2_STRICT_RETRIEVAL_H
