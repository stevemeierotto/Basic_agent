#include "command_processor.h"
#include "file_handler.h"
#include "index_manager.h"
#include "logger.h"
#include "tools.h"
#include "standard_execution_mode.h"
#include "scientific_execution_mode.h"
#include "decision_trace.h"
#include "chat_rag_observability.h"
#include "chat_retrieval_boost.h"
#include "chat_retrieval_config.h"
#include "chat_query_utils.h"
#include "chat_prompt_config.h"
#include "chat_generation_safety.h"
#include "generation_budget.h"
#include "generation_call.h"
#include "chat_turn_timing.h"
#include "agent_context_retrieval.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <iostream>
#include <mutex>
#include <sstream>
#include <regex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <json.hpp>

namespace fs = std::filesystem;

namespace {
std::atomic<int> g_processToolCallProbe{0};

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::int64_t elapsedMs(std::int64_t startMs) {
    return nowMs() - startMs;
}

std::string benchmarkHistoryPath() {
    FileHandler fh;
    return fh.getAgentWorkspacePath("benchmark_history.jsonl");
}

std::vector<nlohmann::json> readBenchmarkHistory(std::size_t limit) {
    std::ifstream in(benchmarkHistoryPath());
    std::vector<nlohmann::json> entries;
    if (!in.is_open()) {
        return entries;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            entries.push_back(nlohmann::json::parse(line));
        } catch (...) {
        }
    }

    if (entries.size() > limit) {
        entries.erase(entries.begin(), entries.end() - static_cast<std::ptrdiff_t>(limit));
    }
    return entries;
}

std::string fileBasename(const std::string& path) {
    try {
        return fs::path(path).filename().string();
    } catch (...) {
        return path;
    }
}

float safeRatio(std::size_t numerator, std::size_t denominator) {
    if (denominator == 0) {
        return 0.0f;
    }
    return static_cast<float>(numerator) / static_cast<float>(denominator);
}

std::vector<Thoth::ChatRagDocumentMetric> buildDocumentMetrics(
    const std::vector<CodeChunk>& chunks,
    const GragDiagnostics& diagnostics) {
    std::vector<Thoth::ChatRagDocumentMetric> documents;
    documents.reserve(chunks.size());

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        Thoth::ChatRagDocumentMetric doc;
        doc.rank = static_cast<int>(i) + 1;
        doc.file = fileBasename(chunks[i].fileName);
        doc.chunk_id = doc.rank;
        doc.start_line = chunks[i].startLine;
        doc.end_line = chunks[i].endLine;
        doc.chars = chunks[i].code.size();
        if (i < diagnostics.breakdowns.size()) {
            doc.score = diagnostics.breakdowns[i].final_score;
            if (doc.file.empty()) {
                doc.file = fileBasename(diagnostics.breakdowns[i].file_name);
            }
        }
        documents.push_back(doc);
    }
    return documents;
}

int countUniqueDocuments(const std::vector<Thoth::ChatRagDocumentMetric>& documents) {
    std::unordered_map<std::string, bool> seen;
    for (const auto& doc : documents) {
        seen[doc.file] = true;
    }
    return static_cast<int>(seen.size());
}

struct ChatTurnPhaseTiming {
    std::int64_t queue_wait_ms = 0;
    std::int64_t session_setup_ms = 0;
    std::int64_t retrieval_latency_ms = 0;
    std::int64_t prompt_build_latency_ms = 0;
    std::int64_t post_processing_latency_ms = 0;
};

void copyGenerationAttempts(std::vector<Thoth::ChatGenerationAttemptRecord>& dest,
                            const std::vector<Thoth::ChatGeneration::GenerationAttemptTelemetry>& src) {
    dest.clear();
    dest.reserve(src.size());
    for (const auto& attempt : src) {
        Thoth::ChatGenerationAttemptRecord row;
        row.attempt = attempt.attempt;
        row.latency_ms = attempt.latency_ms;
        row.prompt_tokens = attempt.prompt_tokens;
        row.completion_tokens = attempt.completion_tokens;
        row.finish_reason = attempt.finish_reason;
        row.provider_ok = attempt.provider_ok;
        row.raw_answer_chars = attempt.raw_answer_chars;
        row.sanitize_reason = attempt.sanitize_reason;
        row.sanitized_answer_chars = attempt.sanitized_answer_chars;
        row.empty_after_sanitize = attempt.empty_after_sanitize;
        row.response_valid = attempt.response_valid;
        row.invalid_reason = attempt.invalid_reason;
        row.transcript_user_marker_count = attempt.transcript_user_marker_count;
        row.transcript_agent_marker_count = attempt.transcript_agent_marker_count;
        row.max_tokens_requested = attempt.max_tokens_requested;
        row.raw_sample_first = attempt.raw_sample_first;
        row.raw_sample_last = attempt.raw_sample_last;
        row.raw_completion = attempt.raw_completion;
        row.sanitized_completion = attempt.sanitized_completion;
        dest.push_back(std::move(row));
    }
}

Thoth::ChatRagContextRecord buildChatRagContextRecord(
    const std::string& requestId,
    const std::string& query,
    int topK,
    const std::string& ragContext,
    const std::string& finalPrompt,
    const std::vector<CodeChunk>& chunks,
    const GragDiagnostics& diagnostics,
    const Thoth::ConversationPromptMetrics& promptMetrics,
    const std::string& llmModel,
    const std::string& groundingMode) {
    Thoth::ChatRagContextRecord record;
    record.request_id = requestId;
    record.query = query;
    record.top_k = topK;
    record.documents = buildDocumentMetrics(chunks, diagnostics);
    record.retrieved_chars = ragContext.size();
    record.conversation_history_chars = promptMetrics.conversation_history_chars;
    record.tool_schema_chars = promptMetrics.tool_schema_chars_in_final;
    record.memory_context_chars = promptMetrics.memory_context_chars;
    record.system_prompt_chars = promptMetrics.system_prompt_chars;
    record.assembled_conversation_chars = promptMetrics.assembled_prompt_chars;
    record.final_prompt_chars = finalPrompt.size();
    record.prompt_before_truncation_chars = promptMetrics.assembled_prompt_chars + promptMetrics.rag_context_chars;
    record.prompt_after_truncation_chars = promptMetrics.final_prompt_chars;
    record.truncated = promptMetrics.truncated;
    record.truncated_section = promptMetrics.truncated_section;
    record.llm_model = llmModel;
    record.grounding_mode = groundingMode;

    if (!ragContext.empty()) {
        record.rag_wrapper_chars = promptMetrics.rag_context_chars +
                                   std::char_traits<char>::length("[RAG Context]\n") +
                                   std::char_traits<char>::length("[User Query]\n") + 1;
    }

    const std::size_t groundingChars = promptMetrics.rag_context_chars + promptMetrics.grounding_rules_chars;
    record.grounding_ratio = safeRatio(groundingChars, record.final_prompt_chars);
    record.tool_ratio = safeRatio(record.tool_schema_chars, record.final_prompt_chars);
    record.history_ratio = safeRatio(record.conversation_history_chars, record.final_prompt_chars);
    record.memory_ratio = safeRatio(record.memory_context_chars, record.final_prompt_chars);
    record.generation_max_tokens = Thoth::GenerationBudget::resolvedOr(Thoth::ChatPrompt::kChatMaxTokens);
    record.chat_stop_sequence_count = static_cast<int>(
        Thoth::ChatPrompt::chatStopSequences(Thoth::ChatPrompt::chatInferenceModeFromEnv()).size());
    if (Thoth::ChatGeneration::chatPromptLoggingEnabled()) {
        record.final_prompt = finalPrompt;
    }
    return record;
}

struct ConversationalInferencePrompt {
    std::string final_prompt;
    std::optional<Thoth::InferenceChatRequest> chat_request;
    Thoth::ChatPrompt::ChatInferenceMode inference_mode;
};

ConversationalInferencePrompt buildConversationalInferencePrompt(
    PromptFactory& promptFactory,
    const std::string& input,
    const std::string& ragContext,
    bool useExtendedSummary,
    const PromptFactory::ConversationBuildOptions& options,
    Thoth::ConversationPromptMetrics* metrics) {
    ConversationalInferencePrompt out;
    out.inference_mode = Thoth::ChatPrompt::chatInferenceModeFromEnv();

    if (out.inference_mode == Thoth::ChatPrompt::ChatInferenceMode::Chat) {
        PromptFactory::ConversationBuildOptions chatOptions = options;
        chatOptions.includeConversationHistory = false;
        const auto rolePrompt = promptFactory.buildChatRolePrompt(
            input, ragContext, useExtendedSummary, chatOptions, metrics);
        const auto priorTurns = promptFactory.getPriorChatTurnMessages();
        std::ostringstream promptTelemetry;
        promptTelemetry << rolePrompt.system_content << "\n---CHAT-ROLE-BOUNDARY---\n";
        Thoth::InferenceChatRequest request;
        request.messages.push_back({"system", rolePrompt.system_content});
        for (const auto& prior : priorTurns) {
            promptTelemetry << "[" << prior.first << "-turn]\n" << prior.second << "\n";
            request.messages.push_back({prior.first, prior.second});
        }
        promptTelemetry << rolePrompt.user_content;
        out.final_prompt = promptTelemetry.str();
        request.messages.push_back({"user", rolePrompt.user_content});
        if (metrics) {
            std::size_t historyChars = 0;
            for (const auto& prior : priorTurns) {
                historyChars += prior.second.size();
            }
            metrics->conversation_history_chars = historyChars;
        }
        out.chat_request = std::move(request);
    } else {
        out.final_prompt =
            promptFactory.buildChatPrompt(input, ragContext, useExtendedSummary, options, metrics);
    }
    return out;
}

void attachChatInferenceMetadata(Thoth::ChatRagContextRecord& record,
                                 const ConversationalInferencePrompt& bundle) {
    record.inference_mode = Thoth::ChatPrompt::chatInferenceModeLabel(bundle.inference_mode);
    if (bundle.chat_request.has_value()) {
        nlohmann::json messages = nlohmann::json::array();
        for (const auto& message : bundle.chat_request->messages) {
            messages.push_back({{"role", message.role}, {"content", message.content}});
        }
        record.chat_messages = std::move(messages);
    }
}

nlohmann::json buildTraceGroundingJson(
    bool grounded,
    const std::string& groundingMode,
    const std::string& groundingReason,
    const std::vector<Thoth::ChatRagDocumentMetric>& documents) {
    nlohmann::json docNames = nlohmann::json::array();
    for (const auto& doc : documents) {
        if (!doc.file.empty()) {
            docNames.push_back(doc.file);
        }
    }
    return {{"grounded", grounded},
            {"grounding_mode", groundingMode},
            {"grounding_decision_reason", groundingReason},
            {"documents", docNames}};
}

void emitChatRagContext(const Thoth::ChatRagContextRecord& record) {
    const nlohmann::json payload = Thoth::ChatRagLogger::contextToJson(record);
    Thoth::ChatRagLogger::instance().logContext(record);
    StructuredLogger::instance().log(
        LogLevel::Info,
        "command_processor",
        "CHAT_RAG_CONTEXT",
        "Chat RAG context assembled",
        payload,
        record.request_id);
}

void emitChatRagResponse(const Thoth::ChatRagResponseRecord& record) {
    const nlohmann::json payload = Thoth::ChatRagLogger::responseToJson(record);
    Thoth::ChatRagLogger::instance().logResponse(record);
    StructuredLogger::instance().log(
        LogLevel::Info,
        "command_processor",
        "CHAT_RAG_RESPONSE",
        "Chat RAG response recorded",
        payload,
        record.request_id);
}

} // namespace

static std::string backendToString(LLMBackend backend) {
    switch (backend) {
        case LLMBackend::Ollama:
            return "ollama";
        case LLMBackend::OpenAI:
            return "openai";
        default:
            return "unknown";
    }
}

CommandProcessor::CommandProcessor(Memory& mem, 
                                   RAGPipeline& ragPipeline, 
                                   LLMInterface& llmInterface,
                                   Config* cfg,
                                   std::shared_ptr<Thoth::ExecutiveController> ctrl)
    : memory(mem),
      rag(ragPipeline),
      llm(llmInterface),
      controller(ctrl),
      promptFactory(mem, ragPipeline),
    indexManager(ragPipeline.getIndexManager()),
    config(cfg)
{
    initializeCommands();
}

void CommandProcessor::syncPromptConfig() {
    try {
        if (!config) return;
        auto pCfg = promptFactory.getConfig();
        pCfg.enableTools = config->enable_tools;
        pCfg.maxContextLength = static_cast<size_t>(config->max_tokens * 4);
        promptFactory.setConfig(pCfg);
    } catch (...) {}
}

bool CommandProcessor::hasDangerousArgPattern(const std::string& args) const {
    static const std::regex dangerousPattern(R"((\|\||&&|;|`|\$\(|\n|\r))");
    return std::regex_search(args, dangerousPattern);
}

void CommandProcessor::handleSimilarityCommand(const std::string& args) {
    try {
        std::unordered_map<std::string, std::unique_ptr<ISimilarity>> options;
        options["dot"] = std::make_unique<DotProductSimilarity>();
        options["cosine"] = std::make_unique<CosineSimilarity>();
        options["euclidean"] = std::make_unique<EuclideanSimilarity>();
        options["jaccard"] = std::make_unique<JaccardSimilarity>();

        std::string chosen = toLower(trim(args));

        if (chosen.empty()) {
            std::cout << "Available similarity methods:\n";
            for (auto& [name, _] : options) std::cout << "  " << name << "\n";
            std::cout << "Enter choice: ";
            std::getline(std::cin, chosen);
            chosen = toLower(trim(chosen));
        }

        auto it = options.find(chosen);
        if (it == options.end()) {
            std::cout << "Unknown similarity: " << chosen << "\n";
            return;
        }

        rag.getIndexManager()->store.setSimilarity(std::move(it->second));
        std::cout << "Similarity set to " << chosen << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error setting similarity: " << e.what() << "\n";
    }
}


void CommandProcessor::resetProcessToolCallProbeForTest() {
    g_processToolCallProbe.store(0);
}

int CommandProcessor::processToolCallProbeCountForTest() {
    return g_processToolCallProbe.load();
}

void CommandProcessor::applyGenerationDiagnostics(
    Thoth::ChatRagResponseRecord& record,
    const Thoth::ChatGeneration::ChatGenerationResult& gen) {
    record.fallback_used = gen.fallback_used;
    record.raw_answer_chars = gen.raw_text.size();
    record.sanitized_answer_chars = gen.sanitized_text.size();
    record.sanitize_reason = gen.sanitize_reason;
    record.retried_without_stops = gen.retried_without_stops;
    record.retry_due_to_regurgitation = gen.retry_due_to_regurgitation;
    record.regurgitation_retry_reason = gen.regurgitation_retry_reason;
    record.regurgitation_detected = gen.regurgitation_detected;
    record.regurgitation_score = gen.regurgitation_score;
    record.used_stops = gen.used_stops;
    record.provider_ok = gen.provider_ok;
    record.finish_reason = gen.finish_reason;
    record.generation_attempt_count = gen.generation_attempt_count;
    record.generation_latency_ms = gen.generation_latency_ms;
    copyGenerationAttempts(record.generation_attempts, gen.generation_attempts);
    if (!gen.generation_attempts.empty()) {
        const auto& lastAttempt = gen.generation_attempts.back();
        record.prompt_tokens = lastAttempt.prompt_tokens;
        record.completion_tokens = lastAttempt.completion_tokens;
    }
    record.raw_sample_first = gen.raw_sample.first;
    record.raw_sample_last = gen.raw_sample.last;
    record.response_valid = gen.response_valid;
    record.invalid_reason = gen.invalid_reason;
    record.final_answer_chars = gen.sanitized_text.size();
    record.generation_max_tokens = Thoth::GenerationBudget::resolvedOr(Thoth::ChatPrompt::kChatMaxTokens);
}

void CommandProcessor::applyChatTurnTelemetry(
    Thoth::ChatRagResponseRecord& record,
    const std::string& user_query,
    const ConversationalTurnResult& turn,
    const DecisionTrace& trace,
    const std::int64_t queue_wait_ms,
    const std::int64_t session_setup_ms,
    const std::int64_t retrieval_latency_ms,
    const std::int64_t prompt_build_latency_ms,
    const std::int64_t post_processing_latency_ms) {
    (void)user_query;
    applyGenerationDiagnostics(record, turn.gen);

    record.turn_started_at_ms = trace.startedAtMs;
    record.turn_finished_at_ms = nowMs();
    record.turn_total_ms = record.turn_finished_at_ms - record.turn_started_at_ms;
    record.queue_wait_ms = queue_wait_ms;
    record.session_setup_ms = session_setup_ms;
    record.retrieval_latency_ms = retrieval_latency_ms;
    record.prompt_build_latency_ms = prompt_build_latency_ms;
    record.post_processing_latency_ms = post_processing_latency_ms;

    record.telemetry_accounted_ms = session_setup_ms + retrieval_latency_ms + prompt_build_latency_ms
                                    + turn.gen.generation_latency_ms + post_processing_latency_ms;
    if (record.turn_total_ms >= record.telemetry_accounted_ms) {
        record.telemetry_unaccounted_ms = record.turn_total_ms - record.telemetry_accounted_ms;
    } else {
        record.telemetry_unaccounted_ms = 0;
    }
    record.worker_turn_total_ms = queue_wait_ms + record.turn_total_ms;
    if (Thoth::ChatGeneration::chatFullRawLoggingEnabled()) {
        record.final_answer = turn.final_response;
        record.final_answer_chars = turn.final_response.size();
    }
}

CommandProcessor::ConversationalTurnResult CommandProcessor::runConversationalGenerate(
    const std::string& prompt,
    const std::string& user_query,
    bool use_greeting_fallback,
    DecisionTrace& trace,
    const std::string& generation_stage_message,
    const std::optional<Thoth::InferenceChatRequest>& chat_request) {
    ConversationalTurnResult turn;

    Thoth::ChatGeneration::ChatGenerateOptions opts;
    opts.max_tokens = Thoth::GenerationBudget::resolvedOr(Thoth::ChatPrompt::kChatMaxTokens);
    if (chat_request.has_value()) {
        opts.stop_sequences =
            Thoth::ChatPrompt::chatStopSequences(Thoth::ChatPrompt::ChatInferenceMode::Chat);
    } else {
        opts.stop_sequences = Thoth::ChatPrompt::chatStopSequences();
    }
    opts.use_greeting_fallback = use_greeting_fallback;
    opts.user_query = user_query;
    if (chat_request.has_value()) {
        opts.chat_request = *chat_request;
    }

    const auto generationStartMs = nowMs();
    turn.gen = Thoth::ChatGeneration::generateAndSanitizeChat(llm, prompt, opts);
    const auto generationLatencyMs = elapsedMs(generationStartMs);

    if (!turn.gen.provider_ok) {
        turn.final_response = llm.formatProviderError(turn.gen.provider_error);
    } else {
        turn.final_response = processToolCall(turn.gen.sanitized_text, trace);
    }

    traceLogger.addStage(
        trace,
        "generation",
        turn.gen.provider_ok,
        generation_stage_message,
        {{"backend", backendToString(llm.getBackend())},
         {"generation_latency_ms", generationLatencyMs},
         {"provider_ok", turn.gen.provider_ok},
         {"fallback_used", turn.gen.fallback_used},
         {"retried_without_stops", turn.gen.retried_without_stops},
         {"used_stops", turn.gen.used_stops},
         {"sanitize_reason", turn.gen.sanitize_reason},
         {"raw_answer_chars", turn.gen.raw_text.size()},
         {"sanitized_answer_chars", turn.gen.sanitized_text.size()},
         {"finish_reason", turn.gen.finish_reason},
         {"generation_attempt_count", turn.gen.generation_attempt_count},
         {"prompt_tokens", turn.gen.generation_attempts.empty()
                               ? 0
                               : turn.gen.generation_attempts.back().prompt_tokens},
         {"completion_tokens", turn.gen.generation_attempts.empty()
                                   ? 0
                                   : turn.gen.generation_attempts.back().completion_tokens},
         {"response_valid", turn.gen.response_valid},
         {"invalid_reason", turn.gen.invalid_reason}});

    return turn;
}

std::string CommandProcessor::processQuery(const std::string& input,
                                           const std::optional<std::string>& active_goal,
                                           const std::string& task_id,
                                           const std::string& raw_capture_id) {
    ChatTurnPhaseTiming phaseTiming;
    std::int64_t worker_started_at_ms = 0;
    if (const auto ctx = Thoth::ChatTurnTiming::consumeWorkerContext()) {
        worker_started_at_ms = ctx->worker_started_at_ms;
        if (ctx->worker_started_at_ms > 0 && ctx->enqueued_at_ms > 0) {
            phaseTiming.queue_wait_ms = ctx->worker_started_at_ms - ctx->enqueued_at_ms;
        }
    }

    auto trace = traceLogger.startTrace("query", input.size());
    if (worker_started_at_ms > 0 && trace.startedAtMs >= worker_started_at_ms) {
        phaseTiming.session_setup_ms = trace.startedAtMs - worker_started_at_ms;
    }
    
    try {
        StructuredLogger::instance().log(
            LogLevel::Info,
            "command_processor",
            "query_start",
            "Processing query request",
            {{"input_length", input.size()}},
            trace.requestId);

        // Phase 1 Stabilization: Log routing decision
        bool goalActive = (controller && controller->get_state() != Thoth::ControllerState::IDLE);
        std::string mode = goalActive ? "PLAN_AWARE" : "CONVERSATIONAL";
        nlohmann::json indexes = goalActive ? nlohmann::json::array({"codebase_index"}) : nlohmann::json::array({"conversations_index"});
        
        StructuredLogger::instance().log(
            LogLevel::Info,
            "command_processor",
            "routing_decision",
            "Determining retrieval strategy",
            {
                {"goal_active", goalActive},
                {"routing_mode", mode},
                {"indexes_selected", indexes},
                {"reason", goalActive ? "active goal detected" : "no active goal detected"}
            },
            trace.requestId);

        traceLogger.addStage(
            trace,
            "intent",
            true,
            "classified as user_query",
            {{"is_command", false}});

        // --- Goal Detection (Phase 9 Upgrade) ---
        // If the user starts with "Goal: " or specific action keywords, trigger the ExecutiveController
        bool shouldBeGoal = false;
        std::string goalText = input;
        
        if (startsWith(toLower(input), "goal:")) {
            shouldBeGoal = true;
            goalText = trim(input.substr(5));
        } else if (startsWith(toLower(input), "implement") || 
                   startsWith(toLower(input), "fix") || 
                   startsWith(toLower(input), "create")) {
            // Heuristic for action-oriented requests
            shouldBeGoal = true;
        }

        if (shouldBeGoal && controller && controller->get_state() == Thoth::ControllerState::IDLE) {
            StructuredLogger::instance().log(
                LogLevel::Info,
                "command_processor",
                "goal_detected",
                "Transitioning query to ExecutiveController goal",
                {{"goal", goalText}},
                trace.requestId);
                
            controller->execute_goal(goalText);
            return "[Goal Started] " + goalText;
        }

        syncPromptConfig();
        ensureInitialized();

        // --- Sync Goal Context from Controller ---
        if (controller && controller->get_state() != Thoth::ControllerState::IDLE) {
            auto currentPlan = controller->get_current_plan();
            if (!currentPlan.plan_id.empty()) {
                std::string stepId = "";
                if (currentPlan.current_index < currentPlan.steps.size()) {
                    stepId = currentPlan.steps[currentPlan.current_index].step_id;
                }
                rag.setPlanContext(currentPlan.plan_id, stepId);
            }
        }

        if (input.size() > maxQueryLength) {
            traceLogger.addStage(
                trace,
                "input_validation",
                false,
                "query rejected by max length check",
                {{"max_query_length", maxQueryLength}});
            traceLogger.finishTrace(trace, false, "denied_input_too_large");
            traceLogger.writeTrace(trace);
            return "[Denied] input exceeds maximum allowed length.";
        }

        std::string queryDenyReason;
        if (!isQueryAllowed(queryDenyReason)) {
            traceLogger.addStage(
                trace,
                "policy",
                false,
                "query denied by policy",
                {{"reason", queryDenyReason}});
            traceLogger.finishTrace(trace, false, "denied_policy");
            traceLogger.writeTrace(trace);
            return "[Denied] " + queryDenyReason;
        }

        if (indexManager && indexManager->getChunks().empty()) {
            Thoth::ConversationPromptMetrics promptMetrics;
            PromptFactory::ConversationBuildOptions options;
            options.includeTools = config && config->enable_tools && Thoth::looksLikeToolIntent(input);
            const auto promptStartMs = nowMs();
            const auto promptBundle = buildConversationalInferencePrompt(
                promptFactory, input, "", false, options, &promptMetrics);
            const std::string& finalPrompt = promptBundle.final_prompt;
            phaseTiming.prompt_build_latency_ms = elapsedMs(promptStartMs);
            std::string llmModel = llm.getSelectedModel();
            if (llmModel.empty() && config) {
                llmModel = config->llm_model;
            }

            Thoth::ChatRagContextRecord contextRecord = buildChatRagContextRecord(
                trace.requestId,
                input,
                DEFAULT_RAG_TOP_K,
                "",
                finalPrompt,
                {},
                GragDiagnostics{},
                promptMetrics,
                llmModel,
                "no_index");
            contextRecord.retrieval_ran = false;
            contextRecord.retrieval_skip_reason = "no_index";
            contextRecord.candidates_found = 0;
            contextRecord.candidates_passed_gate = 0;
            contextRecord.grounding_decision_reason = "empty_index";
            contextRecord.grounded = false;
            contextRecord.prompt_build_latency_ms = phaseTiming.prompt_build_latency_ms;
            attachChatInferenceMetadata(contextRecord, promptBundle);
            emitChatRagContext(contextRecord);
            traceLogger.addStage(
                trace,
                "chat_rag_context",
                true,
                "Chat RAG context metrics recorded (no index)",
                Thoth::ChatRagLogger::contextToJson(contextRecord));

            // Emit empty diagnostics to clear UI
            if (rag.eventCallback) {
                ControllerEvent ev;
                ev.type = EventType::RETRIEVAL_DIAGNOSTICS;
                ev.session_id = session_id.empty() ? memory.getActiveSessionId() : session_id;
                ev.metadata = {
                    {"scoring_type", "no_index"},
                    {"breakdowns", nlohmann::json::array()},
                    {"alpha", 0.0},
                    {"direction_magnitude", 0.0},
                    {"request_id", trace.requestId},
                };
                rag.eventCallback(ev);
            }

            const auto turn = runConversationalGenerate(
                finalPrompt, input, false, trace, "LLM response generated (no RAG)", promptBundle.chat_request);
            const std::string& finalResponse = turn.final_response;

            Thoth::ChatRagResponseRecord responseRecord;
            responseRecord.task_id = task_id;
            if (!raw_capture_id.empty()) {
                Thoth::RawProviderCapture::store(raw_capture_id, turn.gen.raw_text);
            }
            responseRecord.request_id = trace.requestId;
            responseRecord.answer_chars = finalResponse.size();
            responseRecord.retrieved_doc_count = 0;
            responseRecord.grounding_mode = "no_index";
            const auto postStartMs = nowMs();
            try {
                memory.addMessage("user", input);
                memory.addMessage("assistant", finalResponse);
                memory.save();
                memory.updateSummary(input, finalResponse);
            } catch (...) {}
            phaseTiming.post_processing_latency_ms = elapsedMs(postStartMs);
            applyChatTurnTelemetry(responseRecord,
                                   input,
                                   turn,
                                   trace,
                                   phaseTiming.queue_wait_ms,
                                   phaseTiming.session_setup_ms,
                                   phaseTiming.retrieval_latency_ms,
                                   phaseTiming.prompt_build_latency_ms,
                                   phaseTiming.post_processing_latency_ms);
            emitChatRagResponse(responseRecord);
            traceLogger.addStage(
                trace,
                "chat_rag_response",
                true,
                "Chat RAG response metrics recorded (no index)",
                Thoth::ChatRagLogger::responseToJson(responseRecord));

            traceLogger.finishTrace(trace, true, "query_completed_without_rag");
            traceLogger.writeTrace(trace);
            return finalResponse;
        }

        if (Thoth::isGreetingSkipQuery(input)) {
            Thoth::ConversationPromptMetrics promptMetrics;
            PromptFactory::ConversationBuildOptions options;
            options.grounded = false;
            options.includeTools = config && config->enable_tools && Thoth::looksLikeToolIntent(input);
            const auto promptStartMs = nowMs();
            const auto promptBundle = buildConversationalInferencePrompt(
                promptFactory, input, "", false, options, &promptMetrics);
            const std::string& finalPrompt = promptBundle.final_prompt;
            phaseTiming.prompt_build_latency_ms = elapsedMs(promptStartMs);
            std::string llmModel = llm.getSelectedModel();
            if (llmModel.empty() && config) {
                llmModel = config->llm_model;
            }

            Thoth::ChatRagContextRecord contextRecord = buildChatRagContextRecord(
                trace.requestId,
                input,
                DEFAULT_RAG_TOP_K,
                "",
                finalPrompt,
                {},
                GragDiagnostics{},
                promptMetrics,
                llmModel,
                "no_retrieval_hits");
            contextRecord.retrieval_ran = false;
            contextRecord.retrieval_skip_reason = "greeting";
            contextRecord.candidates_found = 0;
            contextRecord.candidates_passed_gate = 0;
            contextRecord.grounding_decision_reason = "greeting_skip";
            contextRecord.grounded = false;
            contextRecord.prompt_build_latency_ms = phaseTiming.prompt_build_latency_ms;
            attachChatInferenceMetadata(contextRecord, promptBundle);
            emitChatRagContext(contextRecord);
            traceLogger.addStage(
                trace,
                "chat_rag_context",
                true,
                "Chat RAG context metrics recorded (greeting skip)",
                Thoth::ChatRagLogger::contextToJson(contextRecord));

            if (rag.eventCallback) {
                ControllerEvent ev;
                ev.type = EventType::RETRIEVAL_DIAGNOSTICS;
                ev.session_id = session_id.empty() ? memory.getActiveSessionId() : session_id;
                ev.metadata = {
                    {"scoring_type", "greeting_skip"},
                    {"breakdowns", nlohmann::json::array()},
                    {"alpha", 0.0},
                    {"direction_magnitude", 0.0},
                    {"request_id", trace.requestId},
                };
                rag.eventCallback(ev);
            }

            const auto turn = runConversationalGenerate(
                finalPrompt, input, true, trace, "LLM response generated (greeting skip)", promptBundle.chat_request);
            const std::string& finalResponse = turn.final_response;

            Thoth::ChatRagResponseRecord responseRecord;
            responseRecord.task_id = task_id;
            if (!raw_capture_id.empty()) {
                Thoth::RawProviderCapture::store(raw_capture_id, turn.gen.raw_text);
            }
            responseRecord.request_id = trace.requestId;
            responseRecord.answer_chars = finalResponse.size();
            responseRecord.retrieved_doc_count = 0;
            responseRecord.grounding_mode = "no_retrieval_hits";
            const auto postStartMs = nowMs();
            try {
                memory.addMessage("user", input);
                memory.addMessage("assistant", finalResponse);
                memory.save();
                memory.updateSummary(input, finalResponse);
            } catch (...) {}
            phaseTiming.post_processing_latency_ms = elapsedMs(postStartMs);
            applyChatTurnTelemetry(responseRecord,
                                   input,
                                   turn,
                                   trace,
                                   phaseTiming.queue_wait_ms,
                                   phaseTiming.session_setup_ms,
                                   phaseTiming.retrieval_latency_ms,
                                   phaseTiming.prompt_build_latency_ms,
                                   phaseTiming.post_processing_latency_ms);
            emitChatRagResponse(responseRecord);
            traceLogger.addStage(
                trace,
                "chat_rag_response",
                true,
                "Chat RAG response metrics recorded (greeting skip)",
                Thoth::ChatRagLogger::responseToJson(responseRecord));

            traceLogger.finishTrace(trace, true, "query_completed_greeting_skip");
            traceLogger.writeTrace(trace);
            return finalResponse;
        }

        // 1. Retrieve context
        const auto retrievalStartMs = nowMs();
        std::string activePlanId = "";
        std::string activeStepId = "";
        bool hasPlanContext = false;
        Plan currentPlan;
        if (controller) {
            currentPlan = controller->get_current_plan();
            if (!currentPlan.plan_id.empty()) {
                hasPlanContext = true;
            }
        }

        if (hasPlanContext) {
            activePlanId = currentPlan.plan_id;
            if (currentPlan.current_index < currentPlan.steps.size()) {
                activeStepId = currentPlan.steps[currentPlan.current_index].step_id;
            }
        }

        // Sync embeddings for GRAG when we have any plan context (even if idle)
        const std::string activeContextKey =
            session_id.empty() ? memory.getActiveSessionId() : session_id;
        const Thoth::ChatRetrievalGoal chatGoal = Thoth::resolveChatRetrievalGoal(
            controller.get(),
            activeContextKey,
            active_goal,
            session_goal_cache_,
            rag.engine.get());

        if (!chatGoal.embedding.empty()) {
            rag.setGoalEmbedding(chatGoal.embedding);
            if (chatGoal.source == "executive" && controller) {
                rag.setCurrentEmbedding(controller->get_current_embedding());
            } else {
                rag.setCurrentEmbedding({});
            }
        } else if (hasPlanContext) {
            rag.setGoalEmbedding(controller->get_goal_embedding());
            rag.setCurrentEmbedding(controller->get_current_embedding());
        }

        if (!chatGoal.error.empty()) {
            StructuredLogger::instance().log(
                LogLevel::Warn,
                "command_processor",
                "session_goal_embed_failed",
                chatGoal.error,
                {{"goal_source", chatGoal.source}, {"session_id", activeContextKey}},
                trace.requestId);
        }

        GragDiagnostics retrievalDiagnostics;
        retrievalDiagnostics.goal_source = chatGoal.source;
        const int topK = static_cast<int>(DEFAULT_RAG_TOP_K);
        Thoth::RetrievalScope retrievalScope =
            Thoth::resolveAgentContextRetrievalScope(activeContextKey, rag.getIndexManager());
        Thoth::RetrievalTrace retrievalTrace;
        std::vector<CodeChunk> contextChunks = rag.retrieveRelevant(
            input, {}, topK, trace.requestId, activePlanId, activeStepId, {}, {}, {},
            &retrievalDiagnostics, &retrievalScope, &retrievalTrace);
        phaseTiming.retrieval_latency_ms = elapsedMs(retrievalStartMs);
        retrievalDiagnostics.goal_source = chatGoal.source;
        if (!chatGoal.embedding.empty()) {
            retrievalDiagnostics.goal_present = true;
        } else if (!chatGoal.error.empty()) {
            retrievalDiagnostics.goal_present = false;
        }

        // Plan M G1 (R1): fail-closed grounding floor on post-boost final_score.
        // grounded=true must mean chunks survived the floor, not merely that the
        // nearest-neighbor list was non-empty.
        Thoth::ChatRetrieval::GroundingFloorResult grounding =
            Thoth::ChatRetrieval::applyGroundingFloor(
                contextChunks, retrievalDiagnostics,
                Thoth::ChatRetrieval::kMinGroundingFinalScore);

        const Thoth::ChatRetrieval::RagChunkPresentation ragPresentation =
            Thoth::ChatRetrieval::ragChunkPresentationFromEnv();
        std::ostringstream contextStream;
        for (const auto& c : grounding.injectable) {
            contextStream << Thoth::ChatRetrieval::formatChunkForPrompt(c, ragPresentation)
                            << "\n---\n";
        }
        std::string ragContext = contextStream.str();

        const bool grounded = !grounding.injectable.empty();

        PromptFactory::ConversationBuildOptions options;
        options.grounded = grounded;
        options.includeTools =
            config && config->enable_tools && Thoth::looksLikeToolIntent(input);

        Thoth::ConversationPromptMetrics promptMetrics;
        const auto promptStartMs = nowMs();
        const auto promptBundle = buildConversationalInferencePrompt(
            promptFactory, input, ragContext, false, options, &promptMetrics);
        const std::string& finalPrompt = promptBundle.final_prompt;
        phaseTiming.prompt_build_latency_ms = elapsedMs(promptStartMs);

        std::string llmModel = llm.getSelectedModel();
        if (llmModel.empty() && config) {
            llmModel = config->llm_model;
        }
        const std::string groundingMode = grounded ? "retrieved_context" : "no_retrieval_hits";

        std::string groundingReason;
        if (grounding.stats.candidates_found == 0) {
            groundingReason = "no_candidates";
        } else if (!grounded) {
            groundingReason = "below_threshold";
        } else {
            groundingReason = "injected_meaningful_hits";
        }

        Thoth::ChatRagContextRecord contextRecord = buildChatRagContextRecord(
            trace.requestId,
            input,
            topK,
            ragContext,
            finalPrompt,
            grounding.injectable,
            grounding.diagnostics,
            promptMetrics,
            llmModel,
            groundingMode);
        contextRecord.retrieval_ran = true;
        contextRecord.retrieval_skip_reason = "none";
        contextRecord.candidates_found = grounding.stats.candidates_found;
        contextRecord.candidates_passed_gate = grounding.stats.candidates_passed_gate;
        contextRecord.grounding_decision_reason = groundingReason;
        contextRecord.grounded = grounded;
        contextRecord.presentation_mode =
            Thoth::ChatRetrieval::ragChunkPresentationLabel(ragPresentation);
        attachChatInferenceMetadata(contextRecord, promptBundle);
        contextRecord.retrieval_latency_ms = phaseTiming.retrieval_latency_ms;
        contextRecord.prompt_build_latency_ms = phaseTiming.prompt_build_latency_ms;
        contextRecord.has_candidate_scores = grounding.stats.has_candidates;
        contextRecord.max_score = grounding.stats.max_score;
        contextRecord.has_injected_scores = grounding.stats.candidates_passed_gate > 0;
        contextRecord.min_injected_score = grounding.stats.min_injected_score;
        if (!retrievalDiagnostics.retrieval_trace.is_null()) {
            contextRecord.retrieval_trace = retrievalDiagnostics.retrieval_trace;
        } else {
            contextRecord.retrieval_trace = retrievalTrace.toJson();
        }
        const nlohmann::json groundingJson = buildTraceGroundingJson(
            grounded, groundingMode, groundingReason, contextRecord.documents);
        if (contextRecord.retrieval_trace.is_object()) {
            contextRecord.retrieval_trace["grounding"] = groundingJson;
        }
        emitChatRagContext(contextRecord);
        traceLogger.addStage(
            trace,
            "chat_rag_context",
            true,
            "Chat RAG context metrics recorded",
            Thoth::ChatRagLogger::contextToJson(contextRecord));

        if (rag.eventCallback) {
            GragDiagnostics enriched = grounding.diagnostics;
            enriched.retrieval_trace = contextRecord.retrieval_trace;
            ControllerEvent ev;
            ev.type = EventType::RETRIEVAL_DIAGNOSTICS;
            ev.session_id = activeContextKey;
            ev.metadata = enriched.to_json();
            ev.metadata["request_id"] = trace.requestId;
            rag.eventCallback(ev);
        }

        // 3. Query LLM (Plan N N6 — shared conversational boundary)
        const auto turn = runConversationalGenerate(
            finalPrompt, input, false, trace, "LLM response generated", promptBundle.chat_request);
        const std::string& finalResponse = turn.final_response;

        Thoth::ChatRagResponseRecord responseRecord;
        responseRecord.task_id = task_id;
        if (!raw_capture_id.empty()) {
            Thoth::RawProviderCapture::store(raw_capture_id, turn.gen.raw_text);
        }
        responseRecord.request_id = trace.requestId;
        responseRecord.answer_chars = finalResponse.size();
        responseRecord.retrieved_doc_count = countUniqueDocuments(contextRecord.documents);
        responseRecord.grounding_mode = groundingMode;
        const auto postStartMs = nowMs();
        try {
            memory.addMessage("user", input);
            memory.addMessage("assistant", finalResponse);
            memory.save();
            memory.updateSummary(input, finalResponse);
        } catch (...) {}
        phaseTiming.post_processing_latency_ms = elapsedMs(postStartMs);
        applyChatTurnTelemetry(responseRecord,
                               input,
                               turn,
                               trace,
                               phaseTiming.queue_wait_ms,
                               phaseTiming.session_setup_ms,
                               phaseTiming.retrieval_latency_ms,
                               phaseTiming.prompt_build_latency_ms,
                               phaseTiming.post_processing_latency_ms);
        emitChatRagResponse(responseRecord);
        traceLogger.addStage(
            trace,
            "chat_rag_response",
            true,
            "Chat RAG response metrics recorded",
            Thoth::ChatRagLogger::responseToJson(responseRecord));

        traceLogger.finishTrace(trace, true, "query_completed");
        traceLogger.writeTrace(trace);
        return finalResponse;

    } catch (const std::exception& e) {
        traceLogger.finishTrace(trace, false, std::string("Query failed with exception: ") + e.what());
        traceLogger.writeTrace(trace);
        return std::string("[Error] Query failed: ") + e.what();
    } catch (...) {
        traceLogger.finishTrace(trace, false, "Query failed with unknown exception");
        traceLogger.writeTrace(trace);
        return "[Error] Query failed with an unknown error.";
    }
}

std::string CommandProcessor::processToolCall(const std::string& response, DecisionTrace& trace) {
    g_processToolCallProbe.fetch_add(1);
    try {
        nlohmann::json responseJson = nlohmann::json::parse(response);
        if (responseJson.contains("tool_call") && responseJson["tool_call"].is_object()) {
            auto toolCall = responseJson["tool_call"];
            if (toolCall.contains("name") && toolCall["name"].is_string() &&
                toolCall.contains("input") && toolCall["input"].is_object()) {
                
                std::string toolName = toolCall["name"].get<std::string>();
                nlohmann::json toolInput = toolCall["input"];

                // NEW: Global Constraint Check (Standard Chat Loop)
                nlohmann::json checkPayload = toolInput;
                checkPayload["tool_name"] = toolName;
                
                // Map tool names to action types for ConstraintChecker
                std::string actionType = "tool_call";
                if (toolName == "code_modify") {
                    std::string op = toolInput.value("operation", "");
                    if (op == "read") actionType = "file_read";
                    else if (op == "apply_diff") actionType = "file_modify";
                } else if (toolName == "web_scrape") {
                    actionType = "network_request";
                } else if (toolName.find("gmail") == 0) {
                    actionType = "network_request";
                    checkPayload["url"] = "google.com"; // Generic for gmail tools
                }

                auto checkResult = constraint_checker_.check_action(actionType, checkPayload);
                if (!checkResult.allowed) {
                    traceLogger.addStage(
                        trace,
                        "tool_execution",
                        false,
                        "Tool BLOCKED by security policy: " + toolName,
                        {{"reason", checkResult.reason}});
                    
                    return nlohmann::json({
                        {"status", "error"},
                        {"error_message", "Action blocked by security policy: " + checkResult.reason}
                    }).dump();
                }

                const auto toolStartMs = nowMs();
                nlohmann::json toolResult = ToolRegistry::instance().executeTool(toolName, toolInput);
                const auto toolLatencyMs = elapsedMs(toolStartMs);

                std::string status = toolResult.value("status", "error");
                bool toolSuccess = (status == "success");

                traceLogger.addStage(
                    trace,
                    "tool_execution",
                    toolSuccess,
                    "Executed tool: " + toolName,
                    {{"latency_ms", toolLatencyMs}, {"status", status}});

                return toolResult.dump();
            }
        }
    } catch (...) {
    }
    return response;
}

std::string CommandProcessor::trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string CommandProcessor::lstripSlash(const std::string& s) {
    if (!s.empty() && s[0] == '/') return s.substr(1);
    return s;
}

std::string CommandProcessor::toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

bool CommandProcessor::startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

void CommandProcessor::runLoop() {
    std::cout << "Basic Chat Agent. Type /help for commands. Type exit or quit to leave.\n";
    std::string line;

    try {
        ensureInitialized();
        std::cout << "RAG system ready.\n";
    } catch (const std::exception& e) {
        std::cout << "Warning: RAG system initialization failed: " << e.what() << "\n";
    }

    while (true) {
        std::cout << "<USER> " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;
        }

        line = trim(line);
        if (line.empty()) continue;

        std::string low = toLower(line);
        if (low == "exit" || low == "quit" || low == "/exit" || low == "/quit") {
            std::cout << "Goodbye.\n";
            break;
        }
        handleCommand(line);
    }
}

void CommandProcessor::showConfig() const {
    try {
        if(config) config->printConfig();
        else std::cout << "No config connected.\n";
    } catch (...) {}
}

void CommandProcessor::setConfig(const std::string& key, const std::string& value) {
    try {
        if (config) {
            if (config->set(key, value)) {
                std::cout << "Updated " << key << " to " << value << "\n";
            } else {
                std::cout << "Failed to update key: " << key << "\n";
            }
        } else {
            std::cout << "No config connected.\n";
        }
    } catch (...) {}
}


std::string CommandProcessor::handleCommand(const std::string& input) {
    try {
        syncPromptConfig();
        ensureInitialized();
        if (!startsWith(input, "/")) {
            std::string response = processQuery(input);
            try { memory.updateSummary(input, response); } catch (...) {}
            return response;
        }

        auto [cmd, args] = parseCommand(input);
        auto trace = traceLogger.startTrace("command", input.size());

        traceLogger.addStage(
            trace,
            "intent",
            true,
            "classified as command",
            {{"command", cmd}});

        if (args.size() > maxCommandArgsLength) {
            traceLogger.finishTrace(trace, false, "denied_command_args_too_large");
            traceLogger.writeTrace(trace);
            return "[Denied] command arguments exceed maximum allowed length.";
        }

        std::string denyReason;
        if (!isCommandAllowed(cmd, args, denyReason)) {
            traceLogger.finishTrace(trace, false, "denied_policy");
            traceLogger.writeTrace(trace);
            return "[Denied] " + denyReason;
        }

        auto it = commandHandlers.find(cmd);
        if (it != commandHandlers.end()) {
            try {
                std::string result = it->second(args);
                traceLogger.finishTrace(trace, true, "command_completed");
                traceLogger.writeTrace(trace);
                return result;
            } catch (const std::exception& e) {
                traceLogger.finishTrace(trace, false, std::string("Command exception: ") + e.what());
                traceLogger.writeTrace(trace);
                return "Error executing command: " + std::string(e.what());
            }
        } else {
            traceLogger.finishTrace(trace, false, "command_unknown");
            traceLogger.writeTrace(trace);
            return "Unknown command '/" + cmd + "'. Try /help.";
        }
    } catch (const std::exception& e) {
        return "Fatal error in command handler: " + std::string(e.what());
    } catch (...) {
        return "Unknown fatal error in command handler.";
    }
}

bool CommandProcessor::isQueryAllowed(std::string& reason) const {
    if (!config) return true;
    if (llm.getBackend() == LLMBackend::OpenAI && !config->allow_network) {
        reason = "network access disabled by policy (allow_network=false).";
        return false;
    }
    return true;
}

bool CommandProcessor::isCommandAllowed(const std::string& cmd, const std::string& args, std::string& reason) const {
    if (!config) return true;
    if ((cmd == "rag" || cmd == "clear" || cmd == "reset" || cmd == "benchmark" || cmd == "prune")
        && !config->allow_file_io) {
        reason = "file I/O disabled by policy (allow_file_io=false).";
        return false;
    }
    if (cmd == "backend") {
        const std::string backendArg = toLower(trim(args));
        if (backendArg == "openai" && !config->allow_network) {
            reason = "cannot switch to OpenAI: network access disabled by policy (allow_network=false).";
            return false;
        }
    }
    if (hasDangerousArgPattern(args)) {
        reason = "command arguments include blocked shell-like patterns.";
        return false;
    }
    return true;
}


std::pair<std::string, std::string> CommandProcessor::parseCommand(const std::string& input) {
    std::string stripped = input;
    if (!stripped.empty() && stripped[0] == '/')
        stripped.erase(0, 1);

    std::istringstream iss(stripped);
    std::string cmd;
    std::getline(iss, cmd, ' ');
    std::string args;
    std::getline(iss, args);
    return { toLower(trim(cmd)), trim(args) };
}

void CommandProcessor::showHelp() {
    std::cout <<
        "Built-ins:\n"
        "  /help               Show this help\n"
        "  /rag                Query knowledge with RAG\n"
        "  /clear              Clears agent's memory and summaries\n"
        "  /backend ollama     Switch to Ollama\n"
        "  /backend openai     Switch to OpenAI\n"
        "  /mode [std|sci]     Switch execution mode\n"
        "  /similarity         Switch Similarity\n"
        "  /benchmark ...      Run retrieval/index benchmarks\n"
        "  /config             Show config values\n"
        "  /set key value      Update config\n"
        "Also: type 'exit' or 'quit' to leave.\n";
}

std::string CommandProcessor::clearMemory() {
    try {
        memory.clear();
        memory.save();
        return "Memory cleared.";
    } catch (const std::exception& e) {
        return "Failed to clear memory: " + std::string(e.what());
    }
}

void CommandProcessor::ensureInitialized() {
    if (!initialized) {
        try {
            FileHandler fh;
            indexManager->init(fh.getRagPath("rag_index.bin"));
            
            if (indexManager->getChunks().empty()) {
                std::string sandboxRag = fh.getAgentWorkspacePath("rag/");
                std::cerr << "[RAG] Index empty. Bootstrapping from sandbox: " << sandboxRag << "\n";
                indexManager->indexProject(sandboxRag);
                indexManager->saveIndex();
            }
            initialized = true;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] RAG initialization failed: " << e.what() << "\n";
            initialized = true;
        } catch (...) {
            std::cerr << "[ERROR] Unknown error during RAG initialization.\n";
            initialized = true;
        }
    }
}

std::string CommandProcessor::formatConsolidationStatusLine(
    const Thoth::ConsolidationStatus& status) const {
    std::ostringstream oss;
    oss << "[Prune status] session=" << status.session_id
        << " hot=" << status.decision.hot_count << '/' << status.max_hot_messages
        << " should_consolidate=" << (status.decision.shouldConsolidate() ? "yes" : "no")
        << " stale=" << (status.marked_stale ? "yes" : "no")
        << " goal_active=" << (status.goal_active ? "yes" : "no");
    const auto reasons = Thoth::consolidationReasonsToStrings(status.decision.reasons);
    if (!reasons.empty() && reasons[0] != "NONE") {
        oss << " reasons=[";
        for (size_t i = 0; i < reasons.size(); ++i) {
            if (reasons[i] == "NONE") continue;
            if (i > 0) oss << ',';
            oss << reasons[i];
        }
        oss << ']';
    }
    return oss.str();
}

std::string CommandProcessor::formatConsolidationResultLine(
    const Thoth::ConsolidationResult& result,
    const std::string& sessionId) const {
    if (result.blocked) {
        return std::string("[Prune blocked] ") + result.block_reason;
    }
    std::ostringstream oss;
    oss << "[Prune] session=" << sessionId
        << " archived=" << result.archived
        << " warm_created=" << result.warm_created
        << " batches=" << result.batches
        << " remaining_hot=" << result.remaining_hot
        << " deferred=" << (result.deferred ? "yes" : "no");
    if (result.archived == 0 && !result.deferred
        && result.decision.hot_count > 0 && !result.decision.shouldConsolidate()) {
        oss << "\nNothing consolidated — policy clear (use --ignore-thresholds to consolidate anyway).";
    }
    return oss.str();
}

namespace {

std::string formatRestoreBound(const std::optional<int64_t>& bound) {
    if (!bound.has_value()) {
        return "*";
    }
    return std::to_string(*bound);
}

std::string truncateRestoreContent(const std::string& content, std::size_t maxChars) {
    std::string oneLine;
    oneLine.reserve(content.size());
    for (char c : content) {
        if (c == '\n' || c == '\r') {
            oneLine.push_back(' ');
        } else {
            oneLine.push_back(c);
        }
    }
    if (oneLine.size() <= maxChars) {
        return oneLine;
    }
    return oneLine.substr(0, maxChars) + "…";
}

} // namespace

std::string CommandProcessor::formatRestoreResultLine(
    const Thoth::RestoreResult& result,
    const std::string& sessionId,
    const Thoth::RestoreRange& range) const {
    if (result.blocked) {
        return std::string("[Restore blocked] ") + result.block_reason;
    }

    const std::string startStr = formatRestoreBound(range.start_ms);
    const std::string endStr = formatRestoreBound(range.end_ms);

    if (result.mode == Thoth::RestoreMode::REHYDRATE) {
        if (!result.block_reason.empty() && result.restored == 0
            && result.block_reason.find("failed") != std::string::npos) {
            return "[Restore rehydrate failed] session=" + sessionId
                + " reason=" + result.block_reason + " (hot unchanged)";
        }
        std::ostringstream oss;
        oss << "[Restore rehydrate] session=" << sessionId
            << " matched=" << result.matched
            << " restored=" << result.restored
            << " skipped_dup=" << result.skipped_dup
            << " start=" << startStr
            << " end=" << endStr;
        return oss.str();
    }

    // REPLAY
    std::ostringstream oss;
    oss << "[Restore replay] session=" << sessionId
        << " matched=" << result.matched
        << " start=" << startStr
        << " end=" << endStr;
    constexpr int kPreviewMax = 20;
    constexpr std::size_t kContentMax = 120;
    const int preview = std::min(result.matched, kPreviewMax);
    for (int i = 0; i < preview; ++i) {
        const auto& turn = result.turns[static_cast<std::size_t>(i)];
        oss << '\n' << "  " << turn.original_timestamp_ms << ' ' << turn.role << ' '
            << truncateRestoreContent(turn.content, kContentMax);
    }
    if (result.matched > kPreviewMax) {
        oss << '\n' << "… and " << (result.matched - kPreviewMax) << " more";
    }
    return oss.str();
}

std::string CommandProcessor::handlePrune(const std::string& args) {
    std::string subcommand = "status";
    bool ignore_thresholds = false;
    bool allow_during_goal = false;
    bool rehydrate = false;
    bool have_start = false;
    bool have_end = false;
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    std::string session_id;

    std::istringstream iss(args);
    std::string token;
    while (iss >> token) {
        const std::string lower = toLower(trim(token));
        if (lower == "--ignore-thresholds") {
            ignore_thresholds = true;
        } else if (lower == "--unsafe") {
            allow_during_goal = true;
        } else if (lower == "--rehydrate") {
            rehydrate = true;
        } else if (lower == "--start") {
            std::string value;
            if (!(iss >> value)) {
                return "Usage: /prune restore [--rehydrate] [--unsafe] [--start <ms>] [--end <ms>] [session]";
            }
            try {
                start_ms = std::stoll(value);
                have_start = true;
            } catch (...) {
                return "[Restore blocked] Invalid --start value.";
            }
        } else if (lower == "--end") {
            std::string value;
            if (!(iss >> value)) {
                return "Usage: /prune restore [--rehydrate] [--unsafe] [--start <ms>] [--end <ms>] [session]";
            }
            try {
                end_ms = std::stoll(value);
                have_end = true;
            } catch (...) {
                return "[Restore blocked] Invalid --end value.";
            }
        } else if (lower == "status" || lower == "explain" || lower == "batch"
                   || lower == "run" || lower == "restore") {
            subcommand = lower;
        } else if (!lower.empty() && lower[0] != '-') {
            session_id = token;
        }
    }

    if (session_id.empty()) {
        session_id = memory.getActiveSessionId();
    }

    auto trace = traceLogger.startTrace("admin_command", args.size());
    traceLogger.addStage(trace, "prune_requested", true, "Manual prune command", {
        {"subcommand", subcommand},
        {"session_id", session_id},
        {"ignore_thresholds", ignore_thresholds},
        {"allow_during_goal", allow_during_goal},
        {"rehydrate", rehydrate},
        {"requested_by", "CLI"}
    });

    std::string response;
    if (subcommand == "restore") {
        Thoth::RestoreRequest request;
        request.mode = rehydrate ? Thoth::RestoreMode::REHYDRATE : Thoth::RestoreMode::REPLAY;
        request.allow_during_goal = allow_during_goal;
        request.requested_by = "CLI";
        if (have_start) {
            request.range.start_ms = start_ms;
        }
        if (have_end) {
            request.range.end_ms = end_ms;
        }

        const auto result = memory.runRestore(session_id, request);
        response = formatRestoreResultLine(result, session_id, request.range);

        const bool ok = !result.blocked
            && (result.mode == Thoth::RestoreMode::REPLAY
                || result.block_reason.empty()
                || result.restored > 0
                || (result.matched == 0 && result.restored == 0));
        // Treat explicit rehydrate failure string as failure for admin_command stage.
        const bool rehydrate_failed = result.mode == Thoth::RestoreMode::REHYDRATE
            && !result.blocked
            && !result.block_reason.empty()
            && result.restored == 0
            && result.matched > 0;

        traceLogger.addStage(trace, "prune_completed", ok && !rehydrate_failed, response, {
            {"session_id", session_id},
            {"subcommand", "restore"},
            {"mode", Thoth::restoreModeToString(result.mode)},
            {"matched", result.matched},
            {"restored", result.restored},
            {"skipped_dup", result.skipped_dup},
            {"blocked", result.blocked}
        });
        traceLogger.finishTrace(trace, ok && !rehydrate_failed, "prune restore completed");
        traceLogger.writeTrace(trace);
        return response;
    }

    if (subcommand == "status") {
        const auto status = memory.getConsolidationStatus(session_id);
        response = formatConsolidationStatusLine(status);
    } else if (subcommand == "explain") {
        const auto status = memory.getConsolidationStatus(session_id);
        response = Thoth::explainConsolidationStatus(status);
    } else if (subcommand == "batch" || subcommand == "run") {
        Thoth::ConsolidationRequest request;
        request.source = Thoth::ConsolidationSource::MANUAL;
        request.ignore_thresholds = ignore_thresholds;
        request.single_batch = (subcommand == "batch");
        request.allow_during_goal = allow_during_goal;
        request.requested_by = "CLI";

        const auto result = memory.runConsolidation(session_id, request);
        response = formatConsolidationResultLine(result, session_id);

        traceLogger.addStage(trace, "prune_completed", !result.blocked, response, {
            {"session_id", session_id},
            {"archived", result.archived},
            {"warm_created", result.warm_created},
            {"batches", result.batches},
            {"remaining_hot", result.remaining_hot},
            {"deferred", result.deferred},
            {"blocked", result.blocked},
            {"source", Thoth::consolidationSourceToString(result.source)},
            {"decision", Thoth::consolidationDecisionToJson(result.decision)}
        });
    } else {
        response = "Unknown prune subcommand. Use: status, explain, batch, run, restore";
        traceLogger.finishTrace(trace, false, response);
        traceLogger.writeTrace(trace);
        return response;
    }

    if (subcommand == "status" || subcommand == "explain") {
        traceLogger.addStage(trace, "prune_completed", true, response, {
            {"session_id", session_id},
            {"subcommand", subcommand}
        });
    }

    traceLogger.finishTrace(trace, true, "prune command completed");
    traceLogger.writeTrace(trace);
    return response;
}

std::string CommandProcessor::handleMemorySubcommand(const std::string& sub,
                                                     const std::string& args) {
    if (sub == "prune") {
        return handlePrune(args);
    }
    return "Unknown memory subcommand '/" + sub + "'.";
}

void CommandProcessor::initializeCommands() {
    commandHandlers["help"] = [this](const std::string&) { 
        return "Built-ins:\n"
               "  /help               Show this help\n"
               "  /rag                Query knowledge with RAG\n"
               "  /clear              Clears agent's memory and summaries\n"
               "  /backend ollama     Switch to Ollama\n"
               "  /backend openai     Switch to OpenAI\n"
               "  /mode [std|sci]     Switch execution mode\n"
               "  /similarity         Switch Similarity\n"
               "  /benchmark ...      Run retrieval/index benchmarks\n"
               "  /config             Show config values\n"
               "  /set key value      Update config\n"
               "  /prune [status|explain|batch|run|restore] [--ignore-thresholds] [--unsafe]\n"
               "                      [--rehydrate] [--start <ms>] [--end <ms>] [session]\n"
               "                      Memory consolidation / range restore (default: status)\n"
               "Also: type 'exit' or 'quit' to leave.\n";
    };
    commandHandlers["h"] = commandHandlers["help"];
    commandHandlers["?"] = commandHandlers["help"];
    
    commandHandlers["clear"] = [this](const std::string&) { return clearMemory(); };
    commandHandlers["reset"] = commandHandlers["clear"];
    commandHandlers["rag"] = [this](const std::string& args) { 
        if (args.empty()) return std::string("Usage: /rag <query>");
        auto chunks = rag.retrieveRelevant(args, {}, 5);
        std::ostringstream oss;
        oss << "Retrieved " << chunks.size() << " chunks for query: " << args << "\n";
        return oss.str();
    };
    commandHandlers["backend"] = [this](const std::string& args) { 
        if (args == "ollama") {
            llm.setBackend(LLMBackend::Ollama);
            return std::string("Backend switched to Ollama");
        } else if (args == "openai") {
            llm.setBackend(LLMBackend::OpenAI);
            return std::string("Backend switched to OpenAI");
        }
        return std::string("Unknown backend. Use 'ollama' or 'openai'");
    };
    commandHandlers["mode"] = [this](const std::string& args) {
        if (!controller) return std::string("Controller not available");
        std::string modeStr = toLower(trim(args));
        if (modeStr == "standard" || modeStr == "std") {
            controller->set_execution_mode(std::make_unique<Thoth::StandardExecutionMode>());
            return std::string("Switched to Standard execution mode");
        } else if (modeStr == "scientific" || modeStr == "sci") {
            controller->set_execution_mode(std::make_unique<Thoth::ScientificExecutionMode>());
            return std::string("Switched to Scientific execution mode");
        }
        return std::string("Unknown mode. Use 'standard' or 'scientific'");
    };
    commandHandlers["goal"] = [this](const std::string& args) {
        if (!controller) return std::string("Controller not available");
        if (args.empty()) return std::string("Usage: /goal <your objective>");
        controller->execute_goal(args);
        return "GOAL ACCEPTED: " + args;
    };
    commandHandlers["aging"] = [this](const std::string&) {
        memory.processMemoryAging();
        return std::string("Memory aging and graph decay triggered");
    };
    commandHandlers["graph"] = [this](const std::string& args) {
        if (args == "stats") {
            auto stats = memory.getGraphStatistics();
            std::ostringstream oss;
            oss << "Graph Stats: Nodes=" << stats.total_nodes << ", Edges=" << stats.total_edges;
            return oss.str();
        } else if (args == "aging") {
            memory.processMemoryAging();
            return std::string("Graph decay triggered");
        }
        return std::string("Usage: /graph <stats|aging>");
    };
    commandHandlers["set"] = [this](const std::string& args) {
        std::istringstream iss(args);
        std::string k, v;
        iss >> k >> v;
        if (!k.empty() && !v.empty()) {
            setConfig(k, v);
            return "Config updated: " + k + "=" + v;
        }
        return std::string("Usage: /set <key> <value>");
    };
    commandHandlers["prune"] = [this](const std::string& args) {
        return handleMemorySubcommand("prune", args);
    };
}

void CommandProcessor::handleBenchmark(const std::string& args) {
    try {
        ensureInitialized();
        std::istringstream iss(args);
        std::string mode;
        iss >> mode;
        mode = toLower(trim(mode));

        if (mode == "index") {
            FileHandler fh;
            const auto start = nowMs();
            indexManager->indexProject(fh.getRagDirectory());
            indexManager->saveIndex();
            std::cout << "[Benchmark] Indexing took " << elapsedMs(start) << "ms\n";
        } else if (mode == "retrieve") {
            std::string q; std::getline(iss, q);
            const auto start = nowMs();
            rag.retrieveRelevant(q, {}, 5);
            std::cout << "[Benchmark] Retrieval took " << elapsedMs(start) << "ms\n";
        } else if (mode == "history") {
            auto entries = readBenchmarkHistory(10);
            if (entries.empty()) {
                std::cout << "[Benchmark] No history found.\n";
                return;
            }
            for (const auto& e : entries) {
                std::cout << " - " << e.dump() << "\n";
            }
        }
    } catch (...) {}
}

void CommandProcessor::handleRag(const std::string& args) {
    try {
        if (args.empty()) return;
        auto chunks = rag.retrieveRelevant(args, {}, 5);
        for (const auto& c : chunks) {
            std::cout << "File: " << c.fileName << "\n" << c.code << "\n---\n";
        }
    } catch (...) {}
}

void CommandProcessor::handleBackend(const std::string& args) {
    try {
        if (args == "ollama") llm.setBackend(LLMBackend::Ollama);
        else if (args == "openai") llm.setBackend(LLMBackend::OpenAI);
    } catch (...) {}
}
