/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 Checkpoint C wiring contract
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_BENCHMARK_EXECUTION_CONTRACT_H
#define THOTH_BENCHMARK_EXECUTION_CONTRACT_H

#include "benchmark_environment.h"

#include <string>

namespace Thoth {

/**
 * Implemented on ExecutiveController (Checkpoint C):
 *
 *   std::string execute_goal(const std::string& goal,
 *                            const BenchmarkAttribution& benchmark = {});
 *
 * Harness: controller.execute_goal(goal, run.attribution());
 * BasicAgentPlugin::executeGoal(goal, run.attribution());
 * GUI / CommandProcessor: execute_goal(goal) — attribution omitted.
 */

} // namespace Thoth

#endif // THOTH_BENCHMARK_EXECUTION_CONTRACT_H
