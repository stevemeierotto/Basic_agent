/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — StandardExecutionMode Phase 1.5
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "iexecution_mode.h"

namespace Thoth {

/**
 * @brief Standard execution mode for Thoth.
 * 
 * Wraps the current baseline execution path:
 * 1. Poll for active step completion.
 * 2. If finished, handle results.
 * 3. Else, decide and perform the next transition (start new step).
 */
class StandardExecutionMode : public IExecutionMode {
public:
    StandardExecutionMode() = default;
    ~StandardExecutionMode() override = default;

    void execute_step(ExecutiveController& controller) override;
    std::string name() const override { return "Standard"; }
};

} // namespace Thoth
