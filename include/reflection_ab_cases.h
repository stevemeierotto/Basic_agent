/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C3 reflection A/B golden cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_REFLECTION_AB_CASES_H
#define THOTH_REFLECTION_AB_CASES_H

#include "iplanner.h"
#include "plan.h"
#include "json.hpp"
#include <memory>
#include <string>
#include <vector>

namespace Thoth {

enum class ReflectionAbFixture {
    RecoverableStepFailure,
    TimeoutStepFailure,
};

struct ReflectionAbCase {
    std::string id;
    std::string description;
    std::string goal;
    ReflectionAbFixture fixture = ReflectionAbFixture::RecoverableStepFailure;
    /** Expected terminal ControllerState name with max_reflections=0. */
    std::string expected_outcome_off;
    /** Expected terminal state with max_reflections=2. */
    std::string expected_outcome_on;
    int expected_planner_calls_off = 1;
    int expected_planner_calls_on = 2;
};

std::vector<ReflectionAbCase> getReflectionAbCases();

/** Mock planner used by the reflection A/B harness (deterministic, no Ollama). */
class ReflectionAbMockPlanner : public IPlanner {
public:
    explicit ReflectionAbMockPlanner(ReflectionAbFixture fixture);

    int call_count() const { return call_count_; }
    const std::vector<std::string>& goals_requested() const { return goals_requested_; }

    Plan create_plan(const std::string& goal) override;
    Plan revise_plan(const Plan& plan, const nlohmann::json& failed_step_result) override;

private:
    ReflectionAbFixture fixture_;
    int call_count_ = 0;
    std::vector<std::string> goals_requested_;
};

} // namespace Thoth

#endif // THOTH_REFLECTION_AB_CASES_H
