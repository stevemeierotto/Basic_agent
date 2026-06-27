/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Semantic validation and limited structural repair for LLM plans (C1)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_PLAN_VALIDATOR_H
#define THOTH_PLAN_VALIDATOR_H

#include <string>
#include "plan.h"

namespace Thoth {

struct PlanValidationResult {
    bool valid = false;
    bool depends_on_repaired = false;
    std::string reason;
};

class PlanValidator {
public:
    /**
     * Validate a plan for corpus Q&A execution.
     * Only structural repair allowed: wire missing LLM depends_on to prior RETRIEVAL.
     * Missing steps or wrong step types → invalid (caller should reject or fallback).
     */
    static PlanValidationResult validateAndRepair(Plan& plan, bool allow_tool_steps = false);

    /** Canonical RETRIEVAL → LLM fallback — constructed in C++, no LLM call. */
    static Plan createFallbackPlan(const std::string& plan_id, const std::string& goal);
};

} // namespace Thoth

#endif // THOTH_PLAN_VALIDATOR_H
