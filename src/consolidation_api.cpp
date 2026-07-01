/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — memory consolidation public API helpers (M3)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/consolidation_api.h"
#include <sstream>

namespace Thoth {

namespace {

std::string triggerLine(const char* label, bool active) {
    return std::string("  ") + (active ? "✓" : "✗") + ' ' + label + (active ? " triggered" : " not triggered");
}

} // namespace

std::string consolidationSourceToString(ConsolidationSource source) {
    switch (source) {
        case ConsolidationSource::AUTOMATIC: return "AUTOMATIC";
        case ConsolidationSource::MANUAL: return "MANUAL";
        default: return "UNKNOWN";
    }
}

std::string explainConsolidationStatus(const ConsolidationStatus& status) {
    std::ostringstream oss;
    oss << "Session: " << status.session_id << '\n';
    oss << "Hot messages: " << status.decision.hot_count
        << " / " << status.max_hot_messages << '\n';

    if (status.max_hot_age_days > 0) {
        oss << "Oldest message age: " << status.decision.oldest_message_age_days
            << " days (threshold: " << status.max_hot_age_days << ")\n";
        oss << "Session inactive: " << status.decision.session_age_days
            << " days\n";
    }

    oss << "Policy triggers:\n";
    oss << triggerLine("HOT_COUNT", hasConsolidationReason(
        status.decision.reasons, ConsolidationReason::HOT_COUNT)) << '\n';
    oss << triggerLine("SESSION_INACTIVE", hasConsolidationReason(
        status.decision.reasons, ConsolidationReason::SESSION_INACTIVE)) << '\n';
    oss << triggerLine("OLDEST_MESSAGE", hasConsolidationReason(
        status.decision.reasons, ConsolidationReason::OLDEST_MESSAGE)) << '\n';

    oss << "Should consolidate: "
        << (status.decision.shouldConsolidate() ? "yes" : "no") << '\n';
    oss << "Marked stale: " << (status.marked_stale ? "yes" : "no") << '\n';
    oss << "Embed engine: " << (status.embed_ready ? "ready" : "unavailable") << '\n';
    oss << "Goal active: " << (status.goal_active ? "yes" : "no") << '\n';
    oss << "Batch size: " << status.prune_batch_size
        << "  Max batches/run: " << status.max_batches_per_invocation << '\n';

    return oss.str();
}

} // namespace Thoth
