/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/default_planner.h"
#include "../include/logger.h"
#include <uuid/uuid.h> // Assuming available, if not use stub

static std::string generate_uuid() {
    // Stub for now to avoid dependency issues if libuuid is not linked
    static int counter = 0;
    return "plan-" + std::to_string(++counter);
}

Plan DefaultPlanner::create_plan(const std::string& goal) {
    Plan plan;
    plan.plan_id = generate_uuid();
    plan.goal = goal;
    
    // Step 1: Retrieval
    PlanStep s1;
    s1.step_id = generate_uuid();
    s1.description = "Retrieve relevant code context for: " + goal;
    s1.type = StepType::RETRIEVAL;
    s1.payload = {{"query", goal}, {"top_k", 5}};
    plan.steps.push_back(s1);

    // Step 2: LLM Analysis
    PlanStep s2;
    s2.step_id = generate_uuid();
    s2.description = "Summarize the findings for: " + goal;
    s2.type = StepType::LLM;
    s2.payload = {{"query", goal}, {"context_from_step", s1.step_id}};
    plan.steps.push_back(s2);

    // Phase 1 Stabilization: Log the plan
    StructuredLogger::instance().log(
        LogLevel::Info,
        "planner",
        "planner_plan_dump",
        "New plan generated",
        {
            {"plan_id", plan.plan_id},
            {"goal", plan.goal},
            {"step_count", plan.steps.size()},
            {"plan_json", plan.to_json()}
        });
    
    return plan;
}

Plan DefaultPlanner::revise_plan(const Plan& /*current_plan*/,
                                 const nlohmann::json& /*step_result*/) {
    // Stub
    return Plan(); // Or return current_plan with modifications
}
