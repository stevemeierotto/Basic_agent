/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — BenchmarkRunner for GRAG vs RAG comparison
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_BENCHMARK_RUNNER_H
#define THOTH_BENCHMARK_RUNNER_H

#include "rag.h"
#include <string>
#include <vector>
#include <map>

namespace Thoth {

/**
 * @struct BenchmarkCase
 * @brief Represents a single test case for retrieval benchmarking.
 */
struct BenchmarkCase {
    std::string case_id;
    std::string goal;
    std::string query;
    std::string trajectory;
    std::vector<std::string> expected_source;
    std::string case_type; // UNAMBIGUOUS | GOAL_DISAMBIGUATES | TRAJECTORY_DISAMBIGUATES
};

/**
 * @struct BenchmarkConfig
 * @brief Parameters for a single benchmark run.
 */
struct BenchmarkConfig {
    float wq = 1.0f;
    float wd = 0.0f;
    float wt = 0.0f;
    float keyword_weight = 0.0f;
    int top_k = 5;
};

/**
 * @struct CaseResult
 * @brief Outcome of a single test case execution.
 */
struct CaseResult {
    std::string case_id;
    std::string case_type;
    std::vector<std::string> top_k_chunks; // file names
    std::vector<std::string> hits;
    float precision_at_k;
    float reciprocal_rank;
    float ndcg_at_k;
};

/**
 * @struct BenchmarkResult
 * @brief Aggregated outcome of a benchmark run.
 */
struct BenchmarkResult {
    std::vector<CaseResult> cases;
    float mean_precision;
    float mean_reciprocal_rank;
    float mean_ndcg;
};

/**
 * @struct CaseDelta
 * @brief Represents the performance difference between RAG and GRAG for a single case.
 */
struct CaseDelta {
    std::string case_id;
    std::string case_type;
    float rag_precision;
    float grag_precision;
    float precision_delta;
    float rag_reciprocal_rank;
    float grag_reciprocal_rank;
    float rag_ndcg;
    float grag_ndcg;
    float ndcg_delta;
    float directional_lift; // rank_RAG - rank_GRAG
    std::vector<std::string> rag_top_chunks;
    std::vector<std::string> grag_top_chunks;
};

/**
 * @struct ComparisonResult
 * @brief Aggregated comparison metrics between RAG and GRAG modes.
 */
struct ComparisonResult {
    std::vector<CaseDelta> deltas;
    float rag_mean_precision;
    float grag_mean_precision;
    float rag_mean_reciprocal_rank;
    float grag_mean_reciprocal_rank;
    float rag_mean_ndcg;
    float grag_mean_ndcg;
    std::map<std::string, float> rag_precision_by_type;
    std::map<std::string, float> grag_precision_by_type;
    std::map<std::string, float> rag_ndcg_by_type;
    std::map<std::string, float> grag_ndcg_by_type;
};

/**
 * @class BenchmarkRunner
 * @brief Executes retrieval benchmarks against a fixed corpus.
 */
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(RAGPipeline& rag);

    /**
     * @brief Runs a benchmark with the given configuration.
     */
    BenchmarkResult run(const BenchmarkConfig& config, const std::vector<BenchmarkCase>& cases);

    /**
     * @brief Runs both RAG and GRAG modes and returns a comparison.
     */
    ComparisonResult runComparison(const std::vector<BenchmarkCase>& cases);

private:
    RAGPipeline& rag_;
};

} // namespace Thoth

#endif // THOTH_BENCHMARK_RUNNER_H
