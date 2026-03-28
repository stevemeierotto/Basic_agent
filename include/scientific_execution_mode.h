/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ScientificExecutionMode Phase 1.6
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "iexecution_mode.h"
#include "problem_state.h"
#include <vector>
#include <string>

namespace Thoth {

/**
 * @brief Scientific reasoning mode for Thoth.
 * 
 * Implements a prototype of the structured scientific reasoning loop:
 * Hypothesis -> Constraints -> Evaluation -> Alternatives -> Selection.
 */
class ScientificExecutionMode : public IExecutionMode {
public:
    ScientificExecutionMode() = default;
    ~ScientificExecutionMode() override = default;

    void execute_step(ExecutiveController& controller) override;
    std::string name() const override { return "Scientific"; }

private:
    void generate_hypotheses(ExecutiveController& controller);
    void extract_constraints(ExecutiveController& controller);
    void evaluate_feasibility(ExecutiveController& controller);
    void finalize_selection(ExecutiveController& controller);
    bool is_converged(ExecutiveController& controller, const ProblemState& state) const;
    float calculate_jaccard(const std::vector<std::string>& a, const std::vector<std::string>& b) const;

    int reasoning_stage_ = 0; // 0: Hypothesis, 1: Constraints, 2: Evaluation, 3: Selection
    float current_confidence_ = 0.0f;
    std::vector<std::string> last_hypothesis_set_;
};

} // namespace Thoth
