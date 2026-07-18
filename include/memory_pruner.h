/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — MemoryPruner (memory consolidation orchestrator)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_PRUNER_H
#define THOTH_MEMORY_PRUNER_H

#include "consolidation_api.h"
#include "consolidation_policy.h"
#include "memory_repository.h"
#include "memory_pruning_config.h"
#include "restore_api.h"
#include "summary_generator.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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

/** @deprecated Prefer ConsolidationResult — retained for M2 call sites. */
using ConsolidationRunResult = ConsolidationResult;

struct BatchConsolidationOutcome {
    int archived = 0;
    int warm_created = 0;
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

    /** Immutable status snapshot including configured thresholds. */
    ConsolidationStatus buildStatus(const std::string& sessionId,
                                    bool marked_stale,
                                    bool goal_active) const;

    /** Unified consolidation entry (automatic + manual). */
    ConsolidationResult runConsolidation(const std::string& sessionId,
                                         const ConsolidationRequest& request);

    /** M4 ranged restore (replay or rehydrate). Emits DecisionTrace on this path. */
    RestoreResult restore(const std::string& sessionId, const RestoreRequest& request);

    /** Consolidate one batch if policy allows. Returns turns removed from hot. */
    int consolidateOneBatch(const std::string& sessionId);

    /** Loop batches until policy clears, no progress, or batch cap. */
    ConsolidationResult consolidateIfNeeded(const std::string& sessionId);

    /** Back-compat alias for consolidateIfNeeded (returns total archived). */
    int prune(const std::string& sessionId);

    /**
     * Legacy full-session REPLAY alias (M4).
     * Equivalent to restore(sessionId, {REPLAY, empty range}) but silent (no DecisionTrace).
     */
    std::vector<MemoryRepository::ArchivedTurnRecord> restore(const std::string& sessionId);

    bool isEmbedReady() const { return embeddingEngine_ != nullptr; }

    /**
     * Clear the no-progress circuit breaker for a session so automatic
     * consolidation will be attempted again. Called on session (re)activation
     * and whenever the failure condition may have been resolved.
     */
    void resetConsolidationBackoff(const std::string& sessionId);

private:
    /** True if the session's automatic consolidation is currently backed off. */
    bool isBackedOff(const std::string& sessionId) const;

    /** Record the outcome of an automatic run to drive the circuit breaker. */
    void recordProgress(const std::string& sessionId, bool madeProgress);

    BatchConsolidationOutcome consolidateOneBatchInternal(
        const std::string& sessionId,
        const ConsolidationDecision& decision,
        ConsolidationSource source,
        const std::string& requested_by);

    bool shouldEnterConsolidation(const ConsolidationDecision& decision,
                                  const ConsolidationRequest& request) const;

    void finalizeResultCompat(ConsolidationResult& result) const;

    MemoryRepository& repo_;
    PruningPolicy policy_;
    SummaryGenerator summaryGenerator_;
    EmbeddingEngine* embeddingEngine_;
    std::shared_ptr<Clock> clock_;

    mutable std::mutex backoff_mtx_;
    std::unordered_map<std::string, int> no_progress_counts_;
};

} // namespace Thoth

#endif // THOTH_MEMORY_PRUNER_H
