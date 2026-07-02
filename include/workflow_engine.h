/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — WorkflowEngine Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <future>
#include <mutex>
#include <memory>
#include "plan.h"
#include "json.hpp"

// Forward declarations
class ToolRegistry;
class RAGPipeline;
class Memory;
class LLMInterface;
class Config;
class IndexManager;
class EmbeddingEngine;

namespace Thoth {

class StepMetricsRepository;
struct E2EvalConfig;
class SealedEpisodeInjectionLog;

/**
 * @brief Represents the outcome of a single PlanStep execution.
 */
struct StepResult {
    std::string step_id;            // ID of the step that produced this result
    bool success = false;
    nlohmann::json data;            // The structured output of the step
    std::string error_message;      // Human-readable error if failed
    int final_retry_count = 0;      // How many retries were actually performed
    int64_t latency_ms = 0;         // Execution duration
};

/**
 * @brief Prior completed steps passed into LLM synthesis.
 */
struct PriorStepContext {
    std::string step_id;
    int step_type = 0;
    std::string description;
    nlohmann::json result;
};

struct StepExecutionContext {
    std::string goal;
    std::vector<PriorStepContext> prior_steps;
    /** C7 Phase 3: snapshot G/C/T at dispatch — safe for parallel RETRIEVAL. */
    std::vector<float> goal_embedding;
    std::vector<float> current_embedding;
    std::vector<float> trajectory_embedding;
    /** E2 A4: evaluation-only STRICT kernel context (injected by harness; null in production). */
    const SealedEpisodeInjectionLog* e2_strict_episode_log = nullptr;
    const E2EvalConfig* e2_eval_config = nullptr;
};

/**
 * @brief The WorkflowEngine is the execution harness for Thoth.
 */
class WorkflowEngine {
public:
    WorkflowEngine(
        std::shared_ptr<ToolRegistry> toolRegistry,
        std::shared_ptr<RAGPipeline> ragPipeline,
        std::shared_ptr<Memory> memory,
        std::shared_ptr<StepMetricsRepository> metricsRepo,
        LLMInterface* llm = nullptr);

    void setLLMInterface(LLMInterface* llm) { llm_ = llm; }
    void setConfig(Config* cfg) { config_ = cfg; }

    virtual ~WorkflowEngine() = default;

    /**
     * @brief Executes a single PlanStep synchronously.
     */
    virtual StepResult executeStep(const PlanStep& step,
                                   const std::string& planId = "",
                                   const StepExecutionContext& context = {});

    /**
     * @brief Executes a single PlanStep asynchronously with timeout.
     */
    virtual std::future<StepResult> executeStepAsync(const PlanStep& step,
                                                     const std::string& planId = "",
                                                     const StepExecutionContext& context = {});

private:
    StepResult executeTool(const PlanStep& step);
    StepResult executeRetrieval(const PlanStep& step,
                                const std::string& planId,
                                const StepExecutionContext& context);
    StepResult executeNode(const PlanStep& step);
    StepResult executeLLM(const PlanStep& step,
                          const std::string& planId,
                          const StepExecutionContext& context);

    bool validateInput(const nlohmann::json& input, const nlohmann::json& schema, std::string& error);

    std::shared_ptr<ToolRegistry> toolRegistry_;
    std::shared_ptr<RAGPipeline> ragPipeline_;
    std::shared_ptr<Memory> memory_;
    std::shared_ptr<StepMetricsRepository> metricsRepo_;
    LLMInterface* llm_ = nullptr;
    Config* config_ = nullptr;
};

} // namespace Thoth
