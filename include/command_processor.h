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
    std::string processQuery(const std::string& input);
    void ensureInitialized();
    void setInitialized(bool value) { initialized = value; }
    void setController(std::shared_ptr<Thoth::ExecutiveController> ctrl) { controller = ctrl; }
    void set_session_id(const std::string& id) { session_id = id; }
    void syncPromptConfig();

    std::string processToolCall(const std::string& response, DecisionTrace& trace);

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

    void showConfig() const;
    void setConfig(const std::string& key, const std::string& value);

    std::pair<std::string, std::string> parseCommand(const std::string& input);
    void showHelp();
    std::string clearMemory();

    using CommandHandler = std::function<std::string(const std::string&)>;
    std::unordered_map<std::string, CommandHandler> commandHandlers;
    void initializeCommands();

    void handleRag(const std::string& args);
    void handleBackend(const std::string& args);
    void handleBenchmark(const std::string& args);
    bool isCommandAllowed(const std::string& cmd, const std::string& args, std::string& reason) const;
    bool isQueryAllowed(std::string& reason) const;
    bool hasDangerousArgPattern(const std::string& args) const;

    // helpers
    static std::string trim(const std::string& s);
    static std::string lstripSlash(const std::string& s);
    static std::string toLower(std::string s);
    static bool startsWith(const std::string& s, const std::string& prefix);
};
