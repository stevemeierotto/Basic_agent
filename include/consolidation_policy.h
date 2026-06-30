/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — memory consolidation policy types (M2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CONSOLIDATION_POLICY_H
#define THOTH_CONSOLIDATION_POLICY_H

#include <../include/json.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace Thoth {

enum class ConsolidationReason : uint8_t {
    NONE = 0,
    HOT_COUNT = 1 << 0,
    SESSION_INACTIVE = 1 << 1,
    OLDEST_MESSAGE = 1 << 2,
    TOKEN_LIMIT = 1 << 3,
    MEMORY_PRESSURE = 1 << 4,
};

inline ConsolidationReason operator|(ConsolidationReason a, ConsolidationReason b) {
    return static_cast<ConsolidationReason>(
        static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline ConsolidationReason& operator|=(ConsolidationReason& a, ConsolidationReason b) {
    a = a | b;
    return a;
}

inline bool hasConsolidationReason(ConsolidationReason flags, ConsolidationReason bit) {
    return (static_cast<uint8_t>(flags) & static_cast<uint8_t>(bit)) != 0;
}

struct ConsolidationDecision {
    ConsolidationReason reasons = ConsolidationReason::NONE;
    int hot_count = 0;
    int session_age_days = 0;
    int oldest_message_age_days = 0;

    bool shouldConsolidate() const {
        return reasons != ConsolidationReason::NONE;
    }
};

std::string consolidationReasonToString(ConsolidationReason reason);
std::vector<std::string> consolidationReasonsToStrings(ConsolidationReason reasons);
nlohmann::json consolidationReasonsToJson(ConsolidationReason reasons);
nlohmann::json consolidationDecisionToJson(const ConsolidationDecision& decision);

} // namespace Thoth

#endif // THOTH_CONSOLIDATION_POLICY_H
