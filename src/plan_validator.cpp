/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — PlanValidator implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/plan_validator.h"
#include "../include/goal_text_utils.h"
#include "../include/json.hpp"
#include <algorithm>

namespace Thoth {

namespace {

bool stepHasToolPayload(const PlanStep& step) {
    if (step.type == StepType::TOOL) {
        return true;
    }
    if (!step.payload.is_object()) {
        return false;
    }
    return step.payload.contains("tool") || step.payload.contains("tool_name");
}

void wireRetrievalLlmDependencies(Plan& plan, bool& repaired) {
    std::string retrievalId;
    for (auto& step : plan.steps) {
        if (step.type == StepType::RETRIEVAL && retrievalId.empty()) {
            retrievalId = step.step_id;
        }
        if (step.type == StepType::LLM && step.depends_on.empty() && !retrievalId.empty()) {
            step.depends_on = {retrievalId};
            repaired = true;
        }
    }
}

} // namespace

Plan PlanValidator::createFallbackPlan(const std::string& plan_id, const std::string& goal) {
    const std::string cleanGoal = cleanGoalForStorage(goal);

    Plan plan;
    plan.plan_id = plan_id;
    plan.goal = cleanGoal;
    plan.status = PlanStatus::ACTIVE;

    PlanStep retrieval;
    retrieval.step_id = "retrieve-context";
    retrieval.description = "Retrieve relevant corpus context";
    retrieval.type = StepType::RETRIEVAL;
    retrieval.payload = {{"query", cleanGoal}, {"top_k", 5}};

    PlanStep synthesis;
    synthesis.step_id = "synthesize";
    synthesis.description = "Summarize findings from retrieved context";
    synthesis.type = StepType::LLM;
    synthesis.depends_on = {"retrieve-context"};
    synthesis.payload = nlohmann::json::object();
    // Phase A: disable synthesis retries until Phase B cancellation semantics.
    synthesis.failure_policy.max_retries = 0;

    plan.steps = {retrieval, synthesis};
    return plan;
}

PlanValidationResult PlanValidator::validateAndRepair(Plan& plan, bool allow_tool_steps) {
    PlanValidationResult result;

    if (plan.steps.empty()) {
        result.reason = "Plan has no steps";
        return result;
    }

    if (!allow_tool_steps) {
        for (const auto& step : plan.steps) {
            if (step.type == StepType::TOOL || stepHasToolPayload(step)) {
                result.reason = "TOOL steps are not allowed for corpus Q&A plans";
                return result;
            }
        }
    }

    bool hasRetrieval = false;
    bool hasLlm = false;
    for (const auto& step : plan.steps) {
        if (step.type == StepType::RETRIEVAL) {
            hasRetrieval = true;
        }
        if (step.type == StepType::LLM) {
            hasLlm = true;
        }
    }

    if (!allow_tool_steps) {
        if (!hasRetrieval) {
            result.reason = "Corpus Q&A plan requires a RETRIEVAL step";
            return result;
        }
        if (!hasLlm) {
            result.reason = "Corpus Q&A plan requires an LLM step";
            return result;
        }

        const auto firstNonTool = std::find_if(plan.steps.begin(), plan.steps.end(), [](const PlanStep& s) {
            return s.type != StepType::TOOL;
        });
        if (firstNonTool != plan.steps.end() && firstNonTool->type != StepType::RETRIEVAL) {
            result.reason = "First executable step must be RETRIEVAL";
            return result;
        }
    }

    if (plan.steps.size() < 2) {
        result.reason = "Plan must contain at least 2 steps";
        return result;
    }

    wireRetrievalLlmDependencies(plan, result.depends_on_repaired);

    // Phase A: LLM/synthesis steps do not retry under the shared step deadline.
    for (auto& step : plan.steps) {
        if (step.type == StepType::LLM) {
            step.failure_policy.max_retries = 0;
        }
    }

    result.valid = true;
    return result;
}

} // namespace Thoth
