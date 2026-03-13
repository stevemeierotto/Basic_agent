/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveMode Interface Phase 1.5
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>

namespace Thoth {

class ExecutiveController;

/**
 * @brief Strategy Pattern interface for execution modes.
 * 
 * Pluggable modes allow swapping the execution logic (e.g., Standard vs Scientific)
 * without modifying the core state machine loop.
 */
class IExecutionMode {
public:
    virtual ~IExecutionMode() = default;

    /**
     * @brief Performs a single execution step.
     * 
     * Called by ExecutiveController::run_loop() on each iteration.
     * 
     * @param controller Reference to the driving controller.
     */
    virtual void execute_step(ExecutiveController& controller) = 0;

    /**
     * @brief Returns the human-readable name of the mode.
     */
    virtual std::string name() const = 0;
};

} // namespace Thoth
