/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — WorkflowEngine Implementation Phase 1.7 (Error Handling Audit)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/workflow_engine.h"
#include "../include/tools.h"
#include "../include/rag.h"
#include "../include/decision_trace.h"
#include "../include/memory.h"
#include "../include/step_metrics_repository.h"
#include "../include/llm_interface.h"
#include "../include/goal_text_utils.h"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace Thoth {

namespace {

std::string extractToolName(const nlohmann::json& payload) {
    std::string name = payload.value("tool", "");
    if (name.empty()) {
        name = payload.value("tool_name", "");
    }
    return name;
}

nlohmann::json extractToolArgs(const nlohmann::json& payload) {
    nlohmann::json args = nlohmann::json::object();
    if (payload.contains("args") && payload["args"].is_object()) {
        args = payload["args"];
    } else {
        static const char* skipKeys[] = {"tool", "tool_name", "query", "top_k", "plan_id", "args"};
        for (auto it = payload.begin(); it != payload.end(); ++it) {
            bool skip = false;
            for (const char* key : skipKeys) {
                if (it.key() == key) {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                args[it.key()] = it.value();
            }
        }
    }
    if (payload.contains("operation") && payload["operation"].is_string()) {
        args["operation"] = payload["operation"];
    }
    return args;
}

bool mockLLMEnabled() {
    const char* mock = std::getenv("THOTH_MOCK_LLM");
    return mock && (std::string(mock) == "1" || std::string(mock) == "true");
}

std::string buildRetrievedContext(const std::vector<PriorStepContext>& priorSteps) {
    std::ostringstream oss;
    for (const auto& prior : priorSteps) {
        if (static_cast<StepType>(prior.step_type) != StepType::RETRIEVAL) {
            continue;
        }
        if (!prior.result.contains("data") || !prior.result["data"].is_object()) {
            continue;
        }
        const auto& data = prior.result["data"];
        if (!data.contains("chunks") || !data["chunks"].is_array()) {
            continue;
        }
        for (const auto& chunk : data["chunks"]) {
            if (!chunk.is_object()) {
                continue;
            }
            const std::string file = chunk.value("file", "");
            const std::string content = chunk.value("content", "");
            if (content.empty()) {
                continue;
            }
            if (!file.empty()) {
                oss << "--- " << file << " ---\n";
            }
            oss << content << "\n\n";
        }
    }
    return oss.str();
}

std::string buildLLMSynthesisPrompt(const PlanStep& step, const StepExecutionContext& context) {
    std::ostringstream prompt;

    std::string goal = context.goal;
    if (!goal.empty()) {
        auto [cleanGoal, _] = Thoth::splitPlanReuseInjection(goal);
        goal = cleanGoal;
    }

    std::string instruction = step.payload.value("prompt", "");
    if (instruction.empty()) {
        instruction = step.description;
    }
    if (instruction.empty()) {
        instruction = "Synthesize a concise answer for the goal using the retrieved context.";
    }

    if (!goal.empty()) {
        prompt << "Goal: " << goal << "\n\n";
    }

    const std::string retrieved = buildRetrievedContext(context.prior_steps);
    if (!retrieved.empty()) {
        prompt << "[Retrieved Context]\n" << retrieved;
    } else {
        prompt << "[Retrieved Context]\n(none — answer from the goal and task only)\n\n";
    }

    prompt << "[Task]\n" << instruction << "\n\n";
    prompt << "Respond in natural language. Do not emit JSON or tool calls.";
    return prompt.str();
}

} // namespace

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

WorkflowEngine::WorkflowEngine(
    std::shared_ptr<ToolRegistry> toolRegistry,
    std::shared_ptr<RAGPipeline> ragPipeline,
    std::shared_ptr<Memory> memory,
    std::shared_ptr<StepMetricsRepository> metricsRepo,
    LLMInterface* llm)
    : toolRegistry_(toolRegistry),
      ragPipeline_(ragPipeline),
      memory_(memory),
      metricsRepo_(metricsRepo),
      llm_(llm) {}

StepResult WorkflowEngine::executeStep(const PlanStep& step,
                                       const std::string& planId,
                                       const StepExecutionContext& context) {
    StepResult result;
    result.step_id = step.step_id;
    int64_t startTime = nowMs();
    
    DecisionTraceLogger traceLogger;
    DecisionTrace trace = traceLogger.startTrace("workflow_step", step.description.size());
    traceLogger.addStage(trace, "dispatch", true, "Executing step: " + step.description, 
        {{"plan_id", planId}, {"step_id", step.step_id}, {"type", static_cast<int>(step.type)}});

    int maxAttempts = 1 + step.failure_policy.max_retries;
    bool success = false;

    for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
        result.final_retry_count = attempt - 1;

        StepResult currentAttempt;
        currentAttempt.step_id = step.step_id;
        try {
            switch (step.type) {
                case StepType::TOOL:
                    currentAttempt = executeTool(step);
                    break;
                case StepType::RETRIEVAL:
                    currentAttempt = executeRetrieval(step, planId);
                    break;
                case StepType::LLM:
                    currentAttempt = executeLLM(step, planId, context);
                    break;
                case StepType::NODE:
                    currentAttempt = executeNode(step);
                    break;
                default:
                    currentAttempt.success = false;
                    currentAttempt.error_message = "Unknown step type";
                    break;
            }
        } catch (const std::exception& e) {
            currentAttempt.success = false;
            currentAttempt.error_message = std::string("Internal dispatcher exception: ") + e.what();
        } catch (...) {
            currentAttempt.success = false;
            currentAttempt.error_message = "Internal dispatcher unknown exception";
        }

        if (currentAttempt.success) {
            result.success = true;
            result.data = currentAttempt.data;
            success = true;
            break;
        } else {
            result.success = false;
            result.error_message = currentAttempt.error_message;
            result.data = currentAttempt.data;
            traceLogger.addStage(trace, "attempt_failed", false, "Attempt " + std::to_string(attempt) + " failed", 
                {{"error", result.error_message}});
            
            if (attempt == maxAttempts) {
                break;
            }
        }
    }

    result.latency_ms = nowMs() - startTime;

    if (success) {
        traceLogger.finishTrace(trace, true, "Step completed successfully");
    } else {
        traceLogger.finishTrace(trace, false, "Step failed after " + std::to_string(maxAttempts) + " attempts: " + result.error_message);
    }

    traceLogger.writeTrace(trace);

    if (metricsRepo_) {
        try {
            StepMetricsRepository::StepMetricRecord metric;
            metric.step_id = step.step_id;
            metric.plan_id = "unknown";
            if (step.payload.contains("plan_id")) {
                metric.plan_id = step.payload["plan_id"];
            }
            metric.tool_name = (step.type == StepType::TOOL) ? extractToolName(step.payload) : "";
            if (metric.tool_name.empty()) {
                metric.tool_name = "unknown";
            }
            metric.latency_ms = result.latency_ms;
            metric.retry_count = result.final_retry_count;
            metric.status = success ? "success" : "failed";
            metric.timestamp_ms = nowMs();
            metricsRepo_->storeMetric(metric);
        } catch (...) {} 
    }

    return result;
}

std::future<StepResult> WorkflowEngine::executeStepAsync(const PlanStep& step,
                                                       const std::string& planId,
                                                       const StepExecutionContext& context) {
    return std::async(std::launch::async, [this, step, planId, context]() -> StepResult {
        try {
            auto executionFuture = std::async(std::launch::async, [this, step, planId, context]() {
                return this->executeStep(step, planId, context);
            });

            int timeoutMs = step.failure_policy.timeout_ms;
            if (timeoutMs <= 0) {
                timeoutMs = (step.type == StepType::LLM) ? 180000 : 30000;
            } else if (step.type == StepType::LLM && timeoutMs < 120000) {
                timeoutMs = 120000;
            }

            auto status = executionFuture.wait_for(std::chrono::milliseconds(timeoutMs));

            if (status == std::future_status::ready) {
                return executionFuture.get();
            } else {
                StepResult timeoutResult;
                timeoutResult.step_id = step.step_id;
                timeoutResult.success = false;
                timeoutResult.error_message = "Step execution timed out after " + std::to_string(timeoutMs) + "ms";
                timeoutResult.latency_ms = timeoutMs;
                timeoutResult.data = {{"status", "failed"}, {"error_message", timeoutResult.error_message}};
                return timeoutResult;
            }
        } catch (const std::exception& e) {
            StepResult res;
            res.step_id = step.step_id;
            res.success = false;
            res.error_message = std::string("Async execution exception: ") + e.what();
            return res;
        } catch (...) {
            StepResult res;
            res.step_id = step.step_id;
            res.success = false;
            res.error_message = "Async execution unknown exception";
            return res;
        }
    });
}

StepResult WorkflowEngine::executeTool(const PlanStep& step) {
    StepResult result;
    try {
        if (!toolRegistry_) {
            result.success = false;
            result.error_message = "ToolRegistry not available";
            return result;
        }

        std::string toolName = extractToolName(step.payload);
        nlohmann::json toolArgs = extractToolArgs(step.payload);
        if (!toolArgs.contains("confirmed")) {
            toolArgs["confirmed"] = true;
        }

        auto tools = toolRegistry_->getAvailableTools();
        const ITool* targetTool = nullptr;
        for (auto t : tools) {
            if (t->name() == toolName) {
                targetTool = t;
                break;
            }
        }

        if (!targetTool) {
            result.success = false;
            result.error_message = "Tool not found: " + toolName;
            return result;
        }

        std::string validationError;
        if (!validateInput(toolArgs, targetTool->input_schema(), validationError)) {
            result.success = false;
            result.error_message = "Input validation failed: " + validationError;
            return result;
        }

        nlohmann::json toolOutput = toolRegistry_->executeTool(toolName, toolArgs);

        if (toolOutput.contains("status") && toolOutput["status"].is_string()) {
            std::string status = toolOutput["status"];
            if (status == "success") {
                result.success = true;
                result.data = toolOutput;
            } else {
                result.success = false;
                result.error_message = toolOutput.value("error_message", "Tool reported failure");
                result.data = toolOutput;
            }
        } else {
            result.success = false;
            result.error_message = "Tool response missing status field";
            result.data = toolOutput;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Tool execution exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.error_message = "Tool execution unknown exception";
    }

    return result;
}

StepResult WorkflowEngine::executeRetrieval(const PlanStep& step, const std::string& planId) {
    StepResult result;
    try {
        if (!ragPipeline_) {
            result.success = false;
            result.error_message = "RAGPipeline not available";
            return result;
        }

        if (!extractToolName(step.payload).empty()) {
            return executeTool(step);
        }

        std::string query = step.payload.value("query", "");
        if (query.empty()) {
            query = step.description;
        }
        int topK = step.payload.value("top_k", 5);

        if (query.empty()) {
            result.success = false;
            result.error_message = "Empty retrieval query";
            return result;
        }

        auto chunks = ragPipeline_->retrieveRelevant(query, {}, topK, "", planId, step.step_id);
        
        nlohmann::json chunksJson = nlohmann::json::array();
        for (const auto& chunk : chunks) {
            nlohmann::json chunkObj = {
                {"file", chunk.fileName},
                {"content", chunk.code}
            };
            chunksJson.push_back(chunkObj);
        }

        if (chunks.empty()) {
            result.success = false;
            result.error_message = "No relevant chunks found for query: " + query;
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
        } else {
            result.success = true;
            result.data = {{"status", "success"}, {"data", {{"chunks", chunksJson}}}};
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Retrieval exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.error_message = "Retrieval unknown exception";
    }
    return result;
}

StepResult WorkflowEngine::executeLLM(const PlanStep& step,
                                      const std::string& planId,
                                      const StepExecutionContext& context) {
    (void)planId;
    StepResult result;

    try {
        if (mockLLMEnabled()) {
            result.success = true;
            result.data = {
                {"status", "success"},
                {"data", {{"response", "Mock LLM synthesis for tests"}}}};
            return result;
        }

        if (!llm_) {
            result.success = false;
            result.error_message = "LLMInterface not available for LLM step";
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
            return result;
        }

        const std::string prompt = buildLLMSynthesisPrompt(step, context);
        if (prompt.empty()) {
            result.success = false;
            result.error_message = "Empty LLM synthesis prompt";
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
            return result;
        }

        const std::string response = llm_->query(prompt);
        if (response.empty()) {
            result.success = false;
            result.error_message = "LLM returned an empty response";
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
            return result;
        }

        result.success = true;
        result.data = {
            {"status", "success"},
            {"data", {{"response", response}, {"prompt_chars", prompt.size()}}}};
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("LLM step exception: ") + e.what();
        result.data = {{"status", "error"}, {"error_message", result.error_message}};
    } catch (...) {
        result.success = false;
        result.error_message = "LLM step unknown exception";
        result.data = {{"status", "error"}, {"error_message", result.error_message}};
    }

    return result;
}

StepResult WorkflowEngine::executeNode(const PlanStep& step) {
    (void)step;
    StepResult result;
    result.success = false;
    result.error_message = "NODE execution not yet implemented";
    return result;
}

bool WorkflowEngine::validateInput(const nlohmann::json& input, const nlohmann::json& schema, std::string& error) {
    try {
        if (schema.contains("required") && schema["required"].is_array()) {
            for (const auto& field : schema["required"]) {
                if (!input.contains(field.get<std::string>())) {
                    error = "Missing required field: " + field.get<std::string>();
                    return false;
                }
            }
        }
        
        if (schema.contains("properties") && schema["properties"].is_object()) {
            for (auto it = input.begin(); it != input.end(); ++it) {
                if (schema["properties"].contains(it.key())) {
                    const auto& propSchema = schema["properties"][it.key()];
                    if (propSchema.contains("type")) {
                        std::string expectedType = propSchema["type"];
                        if (expectedType == "string" && !it.value().is_string()) {
                            error = "Field " + it.key() + " must be a string";
                            return false;
                        }
                        if (expectedType == "integer" && !it.value().is_number_integer()) {
                            error = "Field " + it.key() + " must be an integer";
                            return false;
                        }
                        if (expectedType == "object" && !it.value().is_object()) {
                            error = "Field " + it.key() + " must be an object";
                            return false;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        error = std::string("Validation exception: ") + e.what();
        return false;
    }
    return true;
}

} // namespace Thoth
