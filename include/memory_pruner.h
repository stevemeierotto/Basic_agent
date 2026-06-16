/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — MemoryPruner for managing tiered memory
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_PRUNER_H
#define THOTH_MEMORY_PRUNER_H

#include "memory_repository.h"
#include "memory_pruning_config.h"
#include <string>
#include <vector>

namespace Thoth {

/**
 * @struct PruningPolicy
 * @brief Configuration for the MemoryPruner logic.
 */
struct PruningPolicy {
    size_t max_hot_messages = Thoth::MemoryPruning::kMaxHotMessages;
    int max_hot_age_days = 30;         // Retention period for the Hot Tier (not yet implemented)
    bool summarize_before_pruning = true;
    size_t prune_batch_size = Thoth::MemoryPruning::kPruneBatchSize;
};

/**
 * @class MemoryPruner
 * @brief Handles moving messages between tiered storage layers.
 */
class MemoryPruner {
public:
    explicit MemoryPruner(MemoryRepository& repo, const PruningPolicy& policy = PruningPolicy());

    /**
     * @brief Checks and performs pruning for a session if thresholds are exceeded.
     * @return Number of turns archived.
     */
    int prune(const std::string& sessionId);

    /**
     * @brief Restores archived turns for a session.
     * In Phase 4.2, this is a query method rather than a data-mover.
     */
    std::vector<MemoryRepository::ArchivedTurnRecord> restore(const std::string& sessionId);

private:
    MemoryRepository& repo_;
    PruningPolicy policy_;
};

} // namespace Thoth

#endif // THOTH_MEMORY_PRUNER_H
