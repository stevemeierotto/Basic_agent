/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — WorkflowEngine Implementation Phase 1.7 (Error Handling Audit)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/workflow_engine.h"
#include "../include/llm_timeout_policy.h"
#include "../include/agent_context_retrieval.h"
#include "../include/tools.h"
#include "../include/rag.h"
#include "../include/e2_strict_retrieval.h"
#include "../include/e2_strict_enforcement.h"
#include "../include/episodic_learning_eval.h"
#include "../include/decision_trace.h"
#include "../include/config.h"
#include "../include/runtime_latency_config.h"
#include "../include/generation_budget.h"
#include "../include/generation_call_log.h"
#include "../include/goal_text_utils.h"
#include "../include/grag_diagnostics.h"
#include "../include/memory.h"
#include "../include/step_metrics_repository.h"
#include "../include/llm_interface.h"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <thread>

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

bool mockLLMDelayEnabled(int& delayMsOut) {
    const char* mock = std::getenv("THOTH_MOCK_LLM_DELAY_MS");
    if (!mock || !*mock) {
        return false;
    }
    try {
        delayMsOut = std::max(0, std::stoi(mock));
        return delayMsOut > 0;
    } catch (...) {
        return false;
    }
}

static bool llmResponseIsError(const std::string& response) {
    return response.find("[Error]") != std::string::npos;
}

bool mockStepTimeoutEnabled() {
    const char* mock = std::getenv("THOTH_MOCK_STEP_TIMEOUT");
    return mock && (std::string(mock) == "1" || std::string(mock) == "true");
}

bool mockEpisodicEnabled() {
    const char* mock = std::getenv("THOTH_MOCK_EPISODIC");
    return mock && (std::string(mock) == "1" || std::string(mock) == "true");
}

bool priorRetrievalContainsToken(const std::vector<PriorStepContext>& priorSteps,
                                 const std::string& token) {
    if (token.empty()) {
        return true;
    }
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
            if (content.find(token) != std::string::npos) {
                return true;
            }
            if (!file.empty() && file.find(token) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

std::string buildRetrievedContext(const std::vector<PriorStepContext>& priorSteps,
                                  std::size_t maxChars,
                                  bool* truncatedOut) {
    if (truncatedOut) {
        *truncatedOut = false;
    }
    std::ostringstream oss;
    std::size_t used = 0;
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
            std::ostringstream block;
            if (!file.empty()) {
                block << "--- " << file << " ---\n";
            }
            block << content << "\n\n";
            const std::string piece = block.str();
            if (maxChars > 0 && used + piece.size() > maxChars) {
                if (truncatedOut) {
                    *truncatedOut = true;
                }
                return oss.str();
            }
            oss << piece;
            used += piece.size();
        }
    }
    return oss.str();
}

std::string buildRetrievedContext(const std::vector<PriorStepContext>& priorSteps) {
    return buildRetrievedContext(priorSteps, 0, nullptr);
}

std::string buildLLMSynthesisPrompt(const PlanStep& step,
                                    const StepExecutionContext& context,
                                    Config* config,
                                    bool* contextTruncatedOut) {
    std::ostringstream prompt;

    std::string goal = context.goal;
    if (!goal.empty()) {
        auto [cleanGoal, _] = Thoth::splitPlanReuseInjection(goal);
        goal = cleanGoal;
    }

    std::string instruction;
    if (step.payload.is_object()) {
        instruction = step.payload.value("prompt", "");
    }
    if (instruction.empty()) {
        instruction = step.description;
    }
    if (instruction.empty()) {
        instruction = "Synthesize a concise answer for the goal using the retrieved context.";
    }

    if (!goal.empty()) {
        prompt << "Goal: " << goal << "\n\n";
    }

    std::size_t maxContext = Thoth::RuntimeLatency::kDefaultSynthesisMaxContextChars;
    if (config && config->synthesis_max_context_chars > 0) {
        maxContext = static_cast<std::size_t>(config->synthesis_max_context_chars);
    }
    bool truncated = false;
    const std::string retrieved = buildRetrievedContext(context.prior_steps, maxContext, &truncated);
    if (contextTruncatedOut) {
        *contextTruncatedOut = truncated;
    }
    if (!retrieved.empty()) {
        prompt << "[Retrieved Context]\n" << retrieved;
    } else {
        prompt << "[Retrieved Context]\nNo relevant documents found.\n\n";
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

    if (mockStepTimeoutEnabled() && step.step_id == "timeout-step") {
        result.success = false;
        result.error_message = "Step execution timed out after 1ms";
        result.latency_ms = nowMs() - startTime;
        result.data = {{"status", "failed"}, {"error_message", result.error_message}};
        traceLogger.finishTrace(trace, false, result.error_message);
        traceLogger.writeTrace(trace);
        return result;
    }

    // Phase A: synthesis/LLM steps do not retry (avoids stacked attempts under one soft deadline).
    int maxAttempts = (step.type == StepType::LLM)
                          ? 1
                          : (1 + step.failure_policy.max_retries);
    bool success = false;
    StepResult currentAttempt;
    currentAttempt.step_id = step.step_id;

    for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
        result.final_retry_count = attempt - 1;

        currentAttempt.step_id = step.step_id;
        try {
            switch (step.type) {
                case StepType::TOOL:
                    currentAttempt = executeTool(step);
                    break;
                case StepType::RETRIEVAL:
                    currentAttempt = executeRetrieval(step, planId, context);
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

    // B2 — one structural field forward per invocation; no semantic mutation.
    result.run_block_reason = currentAttempt.run_block_reason;

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

            const int timeoutMs = LlmTimeoutPolicy::stepTimeoutMs(
                step.type, step.failure_policy.timeout_ms);

            auto status = executionFuture.wait_for(std::chrono::milliseconds(timeoutMs));

            if (status == std::future_status::ready) {
                return executionFuture.get();
            } else {
                StepResult timeoutResult;
                timeoutResult.step_id = step.step_id;
                timeoutResult.success = false;
                timeoutResult.error_message = "Step execution timed out after " + std::to_string(timeoutMs) + "ms";
                timeoutResult.latency_ms = timeoutMs;
                timeoutResult.data = {{"status", "failed"},
                                     {"error_message", timeoutResult.error_message},
                                     {"timeout_ms", timeoutMs}};
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

StepResult WorkflowEngine::executeRetrieval(const PlanStep& step,
                                            const std::string& planId,
                                            const StepExecutionContext& context) {
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

        std::string query;
        int topK = 5;
        if (step.payload.is_object()) {
            query = step.payload.value("query", "");
            topK = step.payload.value("top_k", 5);
        }
        if (query.empty()) {
            query = step.description;
        }

        if (query.empty()) {
            result.success = false;
            result.error_message = "Empty retrieval query";
            return result;
        }

        // A4.0b — Single dispatch decision point: STRICT → e2StrictRetrieve(); else RAG below.
        if (context.e2_eval_config &&
            context.e2_eval_config->tier == E2EvalTier::STRICT) {
            if (!context.e2_strict_episode_log) {
                result.success = false;
                result.error_message = "STRICT retrieval missing sealed episode log";
                result.data = {{"status", "error"},
                               {"strict_e2_retrieval", true},
                               {"strict_retrieval_status",
                                e2ArmScoringStatusToString(
                                    E2ArmScoringStatus::FAILED_STRICT_BOUNDARY)},
                               {"error_message", result.error_message},
                               {"data", {{"chunks", nlohmann::json::array()}}},
                               {"retrieved_chunk_count", 0}};
                return result;
            }

            if (!ragPipeline_->indexManager || !ragPipeline_->engine) {
                result.success = false;
                result.error_message = "STRICT retrieval index or engine unavailable";
                result.data = {{"status", "error"},
                               {"strict_e2_retrieval", true},
                               {"strict_retrieval_status",
                                e2ArmScoringStatusToString(
                                    E2ArmScoringStatus::FAILED_RETRIEVAL)},
                               {"error_message", result.error_message},
                               {"data", {{"chunks", nlohmann::json::array()}}},
                               {"retrieved_chunk_count", 0}};
                return result;
            }

            E2StrictRetrievalInput strictInput;
            strictInput.query = query;
            strictInput.episode_log = context.e2_strict_episode_log;
            strictInput.config = *context.e2_eval_config;
            strictInput.index = ragPipeline_->indexManager;
            strictInput.engine = ragPipeline_->engine.get();
            strictInput.top_k = topK;

            const E2StrictRetrievalResult strictResult = e2StrictRetrieve(strictInput);
            const std::string statusStr = e2ArmScoringStatusToString(strictResult.status);

            nlohmann::json chunksJson = nlohmann::json::array();
            for (const auto& chunk : strictResult.chunks) {
                nlohmann::json chunkObj = retrievedChunkToJson(chunk);
                chunkObj["content"] = chunk.content;
                chunkObj["file"] = chunk.chunk_id;
                chunksJson.push_back(chunkObj);
            }

            if (strictResult.status != E2ArmScoringStatus::OK) {
                result.success = false;
                result.error_message =
                    strictResult.error_message.empty()
                        ? ("STRICT retrieval: " + statusStr)
                        : strictResult.error_message;
                result.data = {{"status", "error"},
                               {"strict_e2_retrieval", true},
                               {"strict_retrieval_status", statusStr},
                               {"error_message", result.error_message},
                               {"data", {{"chunks", nlohmann::json::array()}}},
                               {"retrieved_chunk_count", 0}};
                return result;
            }

            result.success = true;
            result.data = {{"status", "success"},
                           {"strict_e2_retrieval", true},
                           {"strict_retrieval_status", statusStr},
                           {"data", {{"chunks", chunksJson}}},
                           {"retrieved_chunk_count",
                            static_cast<int>(strictResult.chunks.size())}};
            return result;
        }

        GragDiagnostics diagnostics;
        Thoth::RetrievalScope retrievalScope = Thoth::resolveAgentContextRetrievalScope(
            memory_ ? memory_->getActiveSessionId() : std::string{}, ragPipeline_->indexManager);
        auto chunks = ragPipeline_->retrieveRelevant(
            query, {}, topK, "", planId, step.step_id,
            context.goal_embedding, context.current_embedding, context.trajectory_embedding,
            &diagnostics, &retrievalScope, nullptr);
        
        nlohmann::json chunksJson = nlohmann::json::array();
        for (const auto& chunk : chunks) {
            nlohmann::json chunkObj = {
                {"file", chunk.fileName},
                {"content", chunk.code}
            };
            chunksJson.push_back(chunkObj);
        }

        if (chunks.empty()) {
            const bool indexPopulated =
                ragPipeline_->indexManager &&
                !ragPipeline_->indexManager->getChunks().empty();

            if (!indexPopulated) {
                result.success = false;
                result.error_message = "No relevant chunks found for query: " + query;
                result.data = {{"status", "error"},
                               {"error_message", result.error_message},
                               {"retrieval_empty", true},
                               {"index_populated", false},
                               {"grag_alpha", diagnostics.alpha},
                               {"grag_routing_mode", diagnostics.routing_mode},
                               {"retrieved_chunk_count", 0}};
            } else {
                result.success = true;
                result.data = {{"status", "success"},
                               {"data", {{"chunks", nlohmann::json::array()}}},
                               {"retrieval_empty", true},
                               {"index_populated", true},
                               {"grag_alpha", diagnostics.alpha},
                               {"grag_routing_mode", diagnostics.routing_mode},
                               {"retrieved_chunk_count", 0}};
            }
        } else {
            result.success = true;
            result.data = {{"status", "success"},
                           {"data", {{"chunks", chunksJson}}},
                           {"grag_alpha", diagnostics.alpha},
                           {"grag_routing_mode", diagnostics.routing_mode},
                           {"retrieved_chunk_count", static_cast<int>(chunks.size())}};
        }
    } catch (const E2RuntimeHeuristicGuardViolation& e) {
        // B2 — sole semantic write site for run_block_reason (typed event → enum).
        result.success = false;
        result.run_block_reason = E2RunBlockReason::RUNTIME_HEURISTIC_GUARD;
        result.error_message = e.what();
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
        bool contextTruncated = false;
        const std::string prompt =
            buildLLMSynthesisPrompt(step, context, config_, &contextTruncated);
        if (prompt.empty()) {
            result.success = false;
            result.error_message = "Empty LLM synthesis prompt";
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
            return result;
        }

        if (mockLLMEnabled()) {
            int delayMs = 0;
            if (mockLLMDelayEnabled(delayMs)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            }

            if (mockEpisodicEnabled() && step.payload.is_object() &&
                step.payload.contains("required_token") &&
                step.payload["required_token"].is_string()) {
                const std::string required_token = step.payload["required_token"].get<std::string>();
                if (!required_token.empty() &&
                    !priorRetrievalContainsToken(context.prior_steps, required_token)) {
                    result.success = false;
                    result.error_message =
                        "E2 mock: required token not found in prior RETRIEVAL context";
                    result.data = {{"status", "error"}, {"error_message", result.error_message}};
                    return result;
                }
            }

            result.success = true;
            result.data = {
                {"status", "success"},
                {"synthesis_prompt_chars", static_cast<int>(prompt.size())},
                {"synthesis_context_truncated", contextTruncated},
                {"data", {{"response", "Mock LLM synthesis for tests"},
                          {"prompt_chars", prompt.size()},
                          {"synthesis_prompt", prompt}}}};
            return result;
        }

        if (!llm_) {
            result.success = false;
            result.error_message = "LLMInterface not available for LLM step";
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
            return result;
        }

        int numPredict = Thoth::RuntimeLatency::kDefaultSynthesisNumPredict;
        if (config_ && config_->synthesis_num_predict > 0) {
            numPredict = config_->synthesis_num_predict;
        }
        if (Thoth::GenerationBudget::hasCeiling()) {
            numPredict = Thoth::GenerationBudget::ceiling();
        }
        Thoth::GenerationCallContext call;
        call.task_id = context.task_id;
        call.session_id = context.session_id;
        call.plan_id = planId;
        call.call_type = "synthesis";
        const Thoth::GenerationOutcome generated = llm_->generateCall(prompt, numPredict, {}, call);
        Thoth::GenerationRecordFields fields;
        fields.associated_generation_id = generated.generation_id;
        fields.context_overflow = Thoth::reportedContextOverflow(generated);
        fields.synthesis_observed_empty =
            generated.ok && generated.provider_usage_reported && generated.text.empty();
        Thoth::GenerationCallLog::append(generated, fields);
        const std::string response = generated.ok ? generated.text
                                                   : std::string("Assistant: [Error] ") + generated.error;
        if (response.empty() || llmResponseIsError(response)) {
            result.success = false;
            result.error_message = response.empty() ? "LLM returned an empty response" : response;
            result.data = {{"status", "error"}, {"error_message", result.error_message}};
            return result;
        }

        result.success = true;
        result.data = {
            {"status", "success"},
            {"synthesis_prompt_chars", static_cast<int>(prompt.size())},
            {"synthesis_context_truncated", contextTruncated},
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
