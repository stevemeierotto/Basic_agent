/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkReporter for serializing and displaying results
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_BENCHMARK_REPORTER_H
#define THOTH_BENCHMARK_REPORTER_H

#include "benchmark_runner.h"
#include <string>

namespace Thoth {

/** Minimal E1 identity for grag_benchmark.jsonl rows (avoids coupling to BenchmarkAttribution). */
struct BenchmarkRunIdentity {
    std::string run_id;
    std::string env_hash;

    bool empty() const { return run_id.empty() && env_hash.empty(); }
};

/**
 * @class BenchmarkReporter
 * @brief Handles reporting of benchmark results to file and stdout.
 */
class BenchmarkReporter {
public:
    /**
     * @brief Writes the comparison result to grag_benchmark.jsonl.
     * When identity is empty, uses legacy benchmark-{timestamp} run_id (permanent fallback).
     */
    static bool reportToFile(const ComparisonResult& result,
                             int corpus_chunk_count,
                             const BenchmarkRunIdentity& identity = {});

    /**
     * @brief Prints a human-readable summary of the comparison to stdout.
     */
    static void reportToStdout(const ComparisonResult& result);

    /**
     * @brief Appends a human-readable summary to docs/benchmark_results.md.
     */
    static bool archiveResults(const ComparisonResult& result, int corpus_chunk_count);
};

} // namespace Thoth

#endif // THOTH_BENCHMARK_REPORTER_H
