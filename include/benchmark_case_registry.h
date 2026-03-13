/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkCaseRegistry for defining retrieval test cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_BENCHMARK_CASE_REGISTRY_H
#define THOTH_BENCHMARK_CASE_REGISTRY_H

#include "benchmark_runner.h"
#include <vector>

namespace Thoth {

/**
 * @class BenchmarkCaseRegistry
 * @brief Provides access to the 15 predefined benchmark test cases.
 */
class BenchmarkCaseRegistry {
public:
    static std::vector<BenchmarkCase> getCases();
};

} // namespace Thoth

#endif // THOTH_BENCHMARK_CASE_REGISTRY_H
