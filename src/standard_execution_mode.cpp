/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — StandardExecutionMode Implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/standard_execution_mode.h"
#include "../include/executive_controller.h"

namespace Thoth {

void StandardExecutionMode::execute_step(ExecutiveController& controller) {
    // Parallel engine: always try to transition
    controller.decide_transition();
}

} // namespace Thoth
