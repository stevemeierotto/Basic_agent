/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — consolidation policy helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/consolidation_policy.h"

namespace Thoth {

namespace {

void appendIfSet(std::vector<std::string>& out, ConsolidationReason flags, ConsolidationReason bit) {
    if (hasConsolidationReason(flags, bit)) {
        out.push_back(consolidationReasonToString(bit));
    }
}

} // namespace

std::string consolidationReasonToString(ConsolidationReason reason) {
    switch (reason) {
        case ConsolidationReason::NONE: return "NONE";
        case ConsolidationReason::HOT_COUNT: return "HOT_COUNT";
        case ConsolidationReason::SESSION_INACTIVE: return "SESSION_INACTIVE";
        case ConsolidationReason::OLDEST_MESSAGE: return "OLDEST_MESSAGE";
        case ConsolidationReason::TOKEN_LIMIT: return "TOKEN_LIMIT";
        case ConsolidationReason::MEMORY_PRESSURE: return "MEMORY_PRESSURE";
        default: return "UNKNOWN";
    }
}

std::vector<std::string> consolidationReasonsToStrings(ConsolidationReason reasons) {
    std::vector<std::string> labels;
    appendIfSet(labels, reasons, ConsolidationReason::HOT_COUNT);
    appendIfSet(labels, reasons, ConsolidationReason::SESSION_INACTIVE);
    appendIfSet(labels, reasons, ConsolidationReason::OLDEST_MESSAGE);
    appendIfSet(labels, reasons, ConsolidationReason::TOKEN_LIMIT);
    appendIfSet(labels, reasons, ConsolidationReason::MEMORY_PRESSURE);
    if (labels.empty()) {
        labels.push_back("NONE");
    }
    return labels;
}

nlohmann::json consolidationReasonsToJson(ConsolidationReason reasons) {
    return nlohmann::json(consolidationReasonsToStrings(reasons));
}

nlohmann::json consolidationDecisionToJson(const ConsolidationDecision& decision) {
    return {
        {"reasons", consolidationReasonsToJson(decision.reasons)},
        {"hot_count", decision.hot_count},
        {"session_age_days", decision.session_age_days},
        {"oldest_message_age_days", decision.oldest_message_age_days},
        {"should_consolidate", decision.shouldConsolidate()}
    };
}

} // namespace Thoth
