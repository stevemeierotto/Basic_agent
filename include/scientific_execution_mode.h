/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ScientificExecutionMode Phase 1.6
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "iexecution_mode.h"
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
    struct Hypothesis {
        std::string description;
        float confidence = 0.0f;
        std::vector<std::string> evidence;
    };

    std::vector<Hypothesis> hypotheses_;
    int reasoning_stage_ = 0; // 0: Hypothesis, 1: Constraints, 2: Evaluation...
};

} // namespace Thoth
