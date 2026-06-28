/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C4 developer/CI test-suite fast path helpers
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_TEST_SUITE_DEV_H
#define THOTH_TEST_SUITE_DEV_H

#include <string>

namespace Thoth {

/** True when THOTH_TEST_SUITE_DEV=1|true (fast mock path for run_test_suite). */
bool testSuiteDevTierEnabled();

/** Deterministic LLM responses for TEST_SUITE dev tier (planner + chat). */
std::string mockTestSuiteLlmResponse(const std::string& prompt);

} // namespace Thoth

#endif // THOTH_TEST_SUITE_DEV_H
