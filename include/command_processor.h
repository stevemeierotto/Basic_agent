/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * uses either Ollama lacal models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include "memory.h"
#include "rag.h"
#include "prompt_factory.h"
#include "llm_interface.h"
#include "index_manager.h"
#include "config.h"
#include "decision_trace.h"
#include "executive_controller.h"
#include "constraint_checker.h"
#include "consolidation_api.h"
#include "restore_api.h"
#include "chat_generation_safety.h"
#include "chat_rag_observability.h"
#include "chat_retrieval_goal.h"
#include "inference_types.h"

#include <optional>

#include <cstdint>
#include <optional>

class CommandProcessor {
public:
    CommandProcessor(Memory& mem, 
                                   RAGPipeline& ragPipeline, 
                                   LLMInterface& llmInterface,
                                   Config* cfg,
                                   std::shared_ptr<Thoth::ExecutiveController> controller = nullptr);

    // Starts a REPL loop
    void runLoop();

    std::string handleCommand(const std::string& input);
    void handleSimilarityCommand(const std::string& args);

    // NEW: Send query through Memory + RAG + LLM
    std::string processQuery(const std::string& input,
                             const std::optional<std::string>& active_goal = std::nullopt);
    void ensureInitialized();
    void setInitialized(bool value) { initialized = value; }
    void setController(std::shared_ptr<Thoth::ExecutiveController> ctrl) { controller = ctrl; }
    void set_session_id(const std::string& id) { session_id = id; }
    void syncPromptConfig();

    std::string processToolCall(const std::string& response, DecisionTrace& trace);

    /** Plan N N6 — test probe: processToolCall invocations since last reset. */
    static void resetProcessToolCallProbeForTest();
    static int processToolCallProbeCountForTest();

private:
    std::string session_id;
    static constexpr size_t DEFAULT_MAX_QUERY_LENGTH = 10000;
    static constexpr size_t DEFAULT_MAX_COMMAND_ARGS_LENGTH = 2048;
    static constexpr int DEFAULT_RAG_TOP_K = 5;
    
    size_t maxQueryLength = DEFAULT_MAX_QUERY_LENGTH;
    size_t maxCommandArgsLength = DEFAULT_MAX_COMMAND_ARGS_LENGTH;
    bool initialized = false;

    Memory& memory;
    RAGPipeline& rag;
    LLMInterface& llm;
    std::shared_ptr<Thoth::ExecutiveController> controller;
    PromptFactory promptFactory;   
    IndexManager * indexManager;
    Config* config;
    DecisionTraceLogger traceLogger;
    Thoth::ConstraintChecker constraint_checker_;
    Thoth::SessionGoalEmbedCache session_goal_cache_;

    void showConfig() const;
    void setConfig(const std::string& key, const std::string& value);

    std::pair<std::string, std::string> parseCommand(const std::string& input);
    void showHelp();
    std::string clearMemory();
    std::string handleMemorySubcommand(const std::string& sub, const std::string& args);
    std::string handlePrune(const std::string& args);
    std::string formatConsolidationStatusLine(const Thoth::ConsolidationStatus& status) const;
    std::string formatConsolidationResultLine(const Thoth::ConsolidationResult& result,
                                              const std::string& sessionId) const;
    std::string formatRestoreResultLine(const Thoth::RestoreResult& result,
                                        const std::string& sessionId,
                                        const Thoth::RestoreRange& range) const;

    using CommandHandler = std::function<std::string(const std::string&)>;
    std::unordered_map<std::string, CommandHandler> commandHandlers;
    void initializeCommands();

    void handleRag(const std::string& args);
    void handleBackend(const std::string& args);
    void handleBenchmark(const std::string& args);
    bool isCommandAllowed(const std::string& cmd, const std::string& args, std::string& reason) const;
    bool isQueryAllowed(std::string& reason) const;
    bool hasDangerousArgPattern(const std::string& args) const;

    /**
     * Plan N N6 — single conversational generation boundary for all three chat arms.
     * Calls generateAndSanitizeChat; Class A skips tools; otherwise processToolCall(sanitized).
     */
    struct ConversationalTurnResult {
        std::string final_response;
        Thoth::ChatGeneration::ChatGenerationResult gen;
    };
    ConversationalTurnResult runConversationalGenerate(
        const std::string& prompt,
        const std::string& user_query,
        bool use_greeting_fallback,
        DecisionTrace& trace,
        const std::string& generation_stage_message,
        const std::optional<Thoth::InferenceChatRequest>& chat_request = std::nullopt);
    static void applyGenerationDiagnostics(Thoth::ChatRagResponseRecord& record,
                                           const Thoth::ChatGeneration::ChatGenerationResult& gen);
    static void applyChatTurnTelemetry(Thoth::ChatRagResponseRecord& record,
                                       const std::string& user_query,
                                       const ConversationalTurnResult& turn,
                                       const DecisionTrace& trace,
                                       std::int64_t queue_wait_ms,
                                       std::int64_t session_setup_ms,
                                       std::int64_t retrieval_latency_ms,
                                       std::int64_t prompt_build_latency_ms,
                                       std::int64_t post_processing_latency_ms);

    // helpers
    static std::string trim(const std::string& s);
    static std::string lstripSlash(const std::string& s);
    static std::string toLower(std::string s);
    static bool startsWith(const std::string& s, const std::string& prefix);
};
