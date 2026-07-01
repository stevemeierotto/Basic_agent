/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic memory learning golden cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/episodic_learning_cases.h"

namespace Thoth {

namespace {

EpisodicLearningExpectations positiveExpectations(const std::string& match_token) {
    EpisodicLearningExpectations exp;
    exp.expect_warm_retrieval_hit = true;
    exp.lift_constraint = EpisodicLiftConstraint::GTE;
    exp.lift_threshold = kEpisodicLearningLiftMargin;
    exp.allow_binary_pass = true;
    exp.include_in_mean_episodic_lift = true;
    exp.retrieval_match_token = match_token;
    return exp;
}

EpisodicLearningExpectations negativeExpectations() {
    EpisodicLearningExpectations exp;
    exp.expect_warm_retrieval_hit = false;
    exp.lift_constraint = EpisodicLiftConstraint::ABS_LT;
    exp.lift_threshold = kEpisodicLearningLiftMargin;
    exp.allow_binary_pass = false;
    exp.include_in_mean_episodic_lift = false;
    exp.forbidden_retrieval_tokens = {"Apollo"};
    exp.retrieval_match_token = "Apollo";
    return exp;
}

} // namespace

std::vector<EpisodicLearningCase> getEpisodicLearningCases() {
    return {
        {"E2-01",
         "Apollo fact — warm retrieval should improve goal outcome",
         "My dog's name is Apollo.",
         "e2-01-plant",
         "What is my dog's name?",
         "Apollo",
         false,
         "",
         positiveExpectations("Apollo")},
        {"E2-02",
         "Zephyrx7 preference — warm retrieval should improve goal outcome",
         "My assistant's codename is Zephyrx7.",
         "e2-02-plant",
         "What is my assistant's codename?",
         "Zephyrx7",
         false,
         "",
         positiveExpectations("Zephyrx7")},
        {"E2-03",
         "Unrelated goal — Apollo must not pollute retrieval; no spurious lift",
         "My dog's name is Apollo.",
         "e2-03-plant",
         "What is the capital of France?",
         "Paris",
         true,
         "Paris is the capital of France and a major European city.",
         negativeExpectations()},
    };
}

EpisodicLearningMockPlanner::EpisodicLearningMockPlanner(std::string validation_token)
    : validation_token_(std::move(validation_token)) {}

Plan EpisodicLearningMockPlanner::create_plan(const std::string& goal) {
    Plan plan;
    plan.plan_id = "e2-plan";
    plan.goal = goal;
    plan.status = PlanStatus::ACTIVE;

    PlanStep retrieve;
    retrieve.step_id = "retrieve";
    retrieve.description = "Retrieve context for goal";
    retrieve.type = StepType::RETRIEVAL;
    retrieve.payload = {{"query", goal}, {"top_k", 5}};

    PlanStep synthesize;
    synthesize.step_id = "synthesize";
    synthesize.description = "Synthesize answer from retrieved context";
    synthesize.type = StepType::LLM;
    synthesize.depends_on = {"retrieve"};
    synthesize.payload = {{"prompt", "Answer the goal using retrieved context only."}};
    if (!validation_token_.empty()) {
        synthesize.payload["required_token"] = validation_token_;
    }

    plan.steps.push_back(retrieve);
    plan.steps.push_back(synthesize);
    return plan;
}

Plan EpisodicLearningMockPlanner::revise_plan(const Plan& plan, const nlohmann::json&) {
    return plan;
}

} // namespace Thoth
