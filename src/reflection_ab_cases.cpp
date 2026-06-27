/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C3 reflection A/B golden cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/reflection_ab_cases.h"

namespace Thoth {

std::vector<ReflectionAbCase> getReflectionAbCases() {
    return {
        {"C3-01",
         "Recoverable step failure — reflection replan should rescue completion",
         "Reflection AB recoverable failure",
         ReflectionAbFixture::RecoverableStepFailure,
         "FAILED",
         "COMPLETED",
         1,
         2},
        {"C3-02",
         "Timeout step failure — reflection must not replan",
         "Reflection AB timeout failure",
         ReflectionAbFixture::TimeoutStepFailure,
         "FAILED",
         "FAILED",
         1,
         1},
    };
}

ReflectionAbMockPlanner::ReflectionAbMockPlanner(ReflectionAbFixture fixture) : fixture_(fixture) {}

Plan ReflectionAbMockPlanner::create_plan(const std::string& goal) {
    ++call_count_;
    goals_requested_.push_back(goal);

    Plan plan;
    plan.plan_id = "ab-plan-" + std::to_string(call_count_);
    plan.goal = goal;
    plan.status = PlanStatus::ACTIVE;

    PlanStep step;
    step.step_id = "step-1";
    step.description = "AB harness step";

    if (fixture_ == ReflectionAbFixture::TimeoutStepFailure) {
        step.type = StepType::LLM;
        step.step_id = "timeout-step";
        step.payload = {{"prompt", "timeout probe"}};
        step.failure_policy.timeout_ms = 1;
    } else if (call_count_ == 1) {
        step.type = StepType::NODE;
        step.payload = {{"node_id", "ab-recoverable-node"}};
    } else {
        step.type = StepType::LLM;
        step.payload = {{"prompt", "recover via mock LLM"}};
    }

    plan.steps.push_back(step);
    return plan;
}

Plan ReflectionAbMockPlanner::revise_plan(const Plan& plan, const nlohmann::json&) {
    return plan;
}

} // namespace Thoth
