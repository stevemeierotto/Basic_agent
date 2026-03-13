/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 8.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/strategy.h"

namespace Thoth {

nlohmann::json Strategy::to_json() const {
    nlohmann::json j;
    j["strategy_id"] = strategy_id;
    j["description"] = description;
    j["step_pattern"] = step_pattern;
    j["success_rate"] = success_rate;
    j["occurrence_count"] = occurrence_count;
    j["created_at"] = created_at;
    return j;
}

Strategy Strategy::from_json(const nlohmann::json& j) {
    Strategy s;
    s.strategy_id = j.value("strategy_id", "");
    s.description = j.value("description", "");
    s.step_pattern = j.value("step_pattern", std::vector<std::string>());
    s.success_rate = j.value("success_rate", 0.0f);
    s.occurrence_count = j.value("occurrence_count", 0);
    s.created_at = j.value("created_at", 0);
    return s;
}

} // namespace Thoth
