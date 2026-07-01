/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — memory consolidation public API (M3)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CONSOLIDATION_API_H
#define THOTH_CONSOLIDATION_API_H

#include "consolidation_policy.h"
#include <string>

namespace Thoth {

enum class ConsolidationSource : uint8_t {
    AUTOMATIC,
    MANUAL,
};

/** Immutable snapshot of consolidation state + configured thresholds. */
struct ConsolidationStatus {
    std::string session_id;
    ConsolidationDecision decision;

    int max_hot_messages = 0;
    int max_hot_age_days = 0;
    int prune_batch_size = 0;
    int max_batches_per_invocation = 0;

    bool marked_stale = false;
    bool goal_active = false;
    bool embed_ready = false;
};

/** Input for Memory::runConsolidation / MemoryPruner::runConsolidation. */
struct ConsolidationRequest {
    ConsolidationSource source = ConsolidationSource::MANUAL;
    bool ignore_thresholds = false;
    bool single_batch = false;
    bool allow_during_goal = false;
    std::string requested_by;
};

/** Output from a consolidation run. */
struct ConsolidationResult {
    ConsolidationSource source = ConsolidationSource::AUTOMATIC;
    ConsolidationDecision decision;

    int archived = 0;
    int warm_created = 0;
    int batches = 0;
    int remaining_hot = 0;
    bool deferred = false;
    bool blocked = false;
    std::string block_reason;

    /** M2 compatibility fields (mirrored from primary fields). */
    int total_archived = 0;
    int batches_completed = 0;
    ConsolidationDecision final_decision;
};

std::string consolidationSourceToString(ConsolidationSource source);
std::string explainConsolidationStatus(const ConsolidationStatus& status);

} // namespace Thoth

#endif // THOTH_CONSOLIDATION_API_H
