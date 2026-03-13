/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 8.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "json.hpp"

namespace Thoth {

/**
 * @brief Represents a reusable pattern of execution steps.
 */
struct Strategy {
    std::string strategy_id;
    std::string description;
    std::vector<std::string> step_pattern; // Sequence of step descriptions or types
    float success_rate = 0.0f;
    int occurrence_count = 0;
    int64_t created_at = 0;

    nlohmann::json to_json() const;
    static Strategy from_json(const nlohmann::json& j);
};

} // namespace Thoth
