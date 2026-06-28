/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C5 robustness golden cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ROBUSTNESS_CASES_H
#define THOTH_ROBUSTNESS_CASES_H

#include "json.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace Thoth {

enum class RobustnessCategory {
    Planning,
    Retrieval,
    Execution,
    Reflection,
    Lifecycle,
};

struct RobustnessCaseOutcome {
    std::string case_id;
    std::string category;
    std::string scenario;

    std::string terminal_state;
    std::string failure_reason;
    int reflection_cycles = 0;
    bool fallback_used = false;
    int planner_calls = 0;
    bool structurally_valid_plan = false;
    bool valid_dependencies = false;
    bool synthesis_prompt_ok = false;

    std::int64_t duration_ms = 0;
    bool pass = false;
    std::string pass_reason;
    nlohmann::json details = nlohmann::json::object();
};

struct RobustnessCaseSpec {
    std::string id;
    RobustnessCategory category;
    std::string scenario;
};

std::vector<RobustnessCaseSpec> getRobustnessCases();
RobustnessCaseOutcome runRobustnessCase(const RobustnessCaseSpec& spec);

const char* categoryName(RobustnessCategory category);

} // namespace Thoth

#endif // THOTH_ROBUSTNESS_CASES_H
