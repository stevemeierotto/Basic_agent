/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C4 developer/CI test-suite fast path helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/test_suite_dev.h"
#include <cstdlib>

namespace Thoth {

namespace {

bool envTruthy(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    const std::string flag(value);
    return flag == "1" || flag == "true" || flag == "TRUE" || flag == "yes";
}

bool looksLikePlannerPrompt(const std::string& prompt) {
    return prompt.find("Schema:\n") != std::string::npos &&
           prompt.find("step_type") != std::string::npos &&
           prompt.find("RETRIEVAL") != std::string::npos;
}

std::string mockPlanJson() {
    return R"({
  "plan": [
    {
      "step_id": "retrieve-context",
      "step_type": "RETRIEVAL",
      "description": "Retrieve relevant corpus context",
      "payload": {"query": "GRAG ExecutiveController retrieval", "top_k": 5}
    },
    {
      "step_id": "synthesize",
      "step_type": "LLM",
      "description": "Summarize findings from retrieved context",
      "depends_on": ["retrieve-context"]
    }
  ]
})";
}

std::string mockChatResponse(const std::string& prompt) {
    if (prompt.find("indexes") != std::string::npos ||
        prompt.find("Indexes") != std::string::npos) {
        return "GRAG uses multi-index routing: PLAN_AWARE, GOAL_ONLY, and CONVERSATIONAL modes. "
               "PLAN_AWARE scans the codebase index when a goal is active.";
    }
    if (prompt.find("scientific") != std::string::npos) {
        return "Scientific execution mode runs hypothesis-driven steps with controlled iteration limits.";
    }
    if (prompt.find("alpha") != std::string::npos) {
        return "GRAG adaptive alpha blends query similarity with directional scoring D = G - C.";
    }
    if (prompt.find("What did you find") != std::string::npos ||
        prompt.find("what did you find") != std::string::npos) {
        return "Directional scoring aligns retrieved chunks with the goal-relative vector D = G - C.";
    }
    if (prompt.find("controller") != std::string::npos ||
        prompt.find("Controller") != std::string::npos) {
        return "ExecutiveController states include IDLE, PLANNING, EXECUTING_STEP, OBSERVING_RESULT, "
               "REVISING_PLAN, and COMPLETED.";
    }
    return "GRAG (Goal-Relative Adaptive Graph Retrieval) uses goal embedding G and current state C. "
           "Direction D = G - C steers retrieval with adaptive alpha blending.";
}

} // namespace

bool testSuiteDevTierEnabled() {
    return envTruthy("THOTH_TEST_SUITE_DEV");
}

std::string mockTestSuiteLlmResponse(const std::string& prompt) {
    if (looksLikePlannerPrompt(prompt)) {
        return mockPlanJson();
    }
    return mockChatResponse(prompt);
}

} // namespace Thoth
