/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — MemoryPruner (memory consolidation orchestrator)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_PRUNER_H
#define THOTH_MEMORY_PRUNER_H

#include "consolidation_policy.h"
#include "memory_repository.h"
#include "memory_pruning_config.h"
#include "summary_generator.h"
#include <memory>
#include <string>
#include <vector>

class Config;
class EmbeddingEngine;
class LLMInterface;

namespace Thoth {

class Clock;

struct PruningPolicy {
    size_t max_hot_messages = MemoryPruning::kMaxHotMessages;
    int max_hot_age_days = 30;
    size_t max_hot_tokens = 0;
    float memory_pressure_threshold = 0.0f;
    bool summarize_before_pruning = true;
    size_t prune_batch_size = MemoryPruning::kPruneBatchSize;
    size_t max_batches_per_invocation = MemoryPruning::kMaxBatchesPerInvocation;

    static PruningPolicy fromConfig(const Config& config);
};

struct ConsolidationRunResult {
    int total_archived = 0;
    int batches_completed = 0;
    bool deferred = false;
    ConsolidationDecision final_decision;
};

class MemoryPruner {
public:
    MemoryPruner(MemoryRepository& repo,
                 const PruningPolicy& policy = PruningPolicy(),
                 LLMInterface* llm = nullptr,
                 EmbeddingEngine* embeddingEngine = nullptr,
                 std::shared_ptr<Clock> clock = nullptr);

    /** Evaluate policy without side effects (no LLM / DB writes). */
    ConsolidationDecision evaluatePolicy(const std::string& sessionId) const;

    /** Consolidate one batch if policy allows. Returns turns removed from hot. */
    int consolidateOneBatch(const std::string& sessionId);

    /** Loop batches until policy clears, no progress, or batch cap. */
    ConsolidationRunResult consolidateIfNeeded(const std::string& sessionId);

    /** Back-compat alias for consolidateIfNeeded (returns total archived). */
    int prune(const std::string& sessionId);

    std::vector<MemoryRepository::ArchivedTurnRecord> restore(const std::string& sessionId);

private:
    int consolidateOneBatchInternal(const std::string& sessionId, const ConsolidationDecision& decision);

    MemoryRepository& repo_;
    PruningPolicy policy_;
    SummaryGenerator summaryGenerator_;
    EmbeddingEngine* embeddingEngine_;
    std::shared_ptr<Clock> clock_;
};

} // namespace Thoth

#endif // THOTH_MEMORY_PRUNER_H
