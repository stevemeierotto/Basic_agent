/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — MemoryPruner implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/memory_pruner.h"
#include "../include/decision_trace.h"
#include <iostream>

namespace Thoth {

MemoryPruner::MemoryPruner(MemoryRepository& repo, const PruningPolicy& policy)
    : repo_(repo), policy_(policy) {}

int MemoryPruner::prune(const std::string& sessionId) {
    int hot_count = repo_.getHotMessageCount(sessionId);
    
    if (hot_count <= static_cast<int>(policy_.max_hot_messages)) {
        return 0;
    }

    int to_archive = static_cast<int>(policy_.prune_batch_size);
    
    // Ensure we don't try to archive more than we have
    if (to_archive > hot_count) to_archive = hot_count;

    DecisionTraceLogger logger;
    DecisionTrace trace = logger.startTrace("memory_pruning", hot_count);
    
    bool success = repo_.archiveMessages(sessionId, to_archive, 1); // Version 1
    
    if (success) {
        logger.addStage(trace, "pruning_executed", true, "Archived " + std::to_string(to_archive) + " turns to Cold Tier", {
            {"session_id", sessionId},
            {"turns_archived", to_archive},
            {"remaining_hot", hot_count - to_archive}
        });
        logger.finishTrace(trace, true, "Memory pruning completed successfully");
    } else {
        logger.addStage(trace, "pruning_failed", false, "Failed to archive turns", {{"session_id", sessionId}});
        logger.finishTrace(trace, false, "Memory pruning failed");
    }
    
    logger.writeTrace(trace);

    return success ? to_archive : 0;
}

std::vector<MemoryRepository::ArchivedTurnRecord> MemoryPruner::restore(const std::string& sessionId) {
    return repo_.getArchivedMessages(sessionId);
}

} // namespace Thoth
