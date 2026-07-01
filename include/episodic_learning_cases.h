/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic memory learning golden cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODIC_LEARNING_CASES_H
#define THOTH_EPISODIC_LEARNING_CASES_H

#include "episodic_learning_eval.h"
#include "iplanner.h"
#include "plan.h"

#include <memory>
#include <string>
#include <vector>

namespace Thoth {

/** Protocol v1.1 — see docs/E2_PROTOCOL.md */
struct EpisodicLearningCase {
    std::string id;
    std::string description;
    /** User message planted before consolidation on the warm arm (empty = no plant). */
    std::string plant_message;
    std::string plant_session_id;
    std::string goal;
    /** Token the mock LLM step requires in prior RETRIEVAL chunks (empty = no requirement). */
    std::string validation_token;
    /** When true, cold arm also receives consolidated memory (E2-03 negative control). */
    bool cold_arm_pre_consolidated = false;
    /** Optional RAG corpus snippet indexed for unrelated goals (E2-03). */
    std::string index_distractor_text;
    EpisodicLearningExpectations expectations;
};

std::vector<EpisodicLearningCase> getEpisodicLearningCases();

/** Deterministic RETRIEVAL → LLM plan for E2 mock tier. */
class EpisodicLearningMockPlanner : public IPlanner {
public:
    explicit EpisodicLearningMockPlanner(std::string validation_token);

    Plan create_plan(const std::string& goal) override;
    Plan revise_plan(const Plan& plan, const nlohmann::json& failed_step_result) override;

private:
    std::string validation_token_;
};

} // namespace Thoth

#endif // THOTH_EPISODIC_LEARNING_CASES_H
