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
#include "memory.h"
#include "rag.h"
#include "prompt_factory.h"
#include "llm_interface.h"
#include "webscraperTools.h"

class CommandProcessor {
public:
    CommandProcessor(Memory& mem, RAGPipeline& rag, LLMInterface& llm);

    // Starts a REPL loop
    void runLoop();

    // Handle a single line (used by runLoop, but also handy for tests)
    void handleCommand(const std::string& input);

    // NEW: Send query through Memory + RAG + LLM
    std::string processQuery(const std::string& input);

private:
    // Add to header
private:
    static constexpr size_t DEFAULT_MAX_QUERY_LENGTH = 10000;
    static constexpr int DEFAULT_RAG_TOP_K = 5;
    static constexpr int DEFAULT_SCRAPE_RESULTS = 3;
    
    size_t maxQueryLength = DEFAULT_MAX_QUERY_LENGTH;
    bool initialized = false;

    Memory& memory;
    RAGPipeline& rag;
    LLMInterface& llm;
    PromptFactory promptFactory;   
    WebScraperTools scraper;

    void ensureInitialized();
    std::pair<std::string, std::string> parseCommand(const std::string& input);
    void showHelp();
    void clearMemory();

    using CommandHandler = std::function<void(const std::string&)>;
    std::unordered_map<std::string, CommandHandler> commandHandlers;
    void initializeCommands();

    void handleScrape(const std::string& args);
    void handleRag(const std::string& args);
    void handleBackend(const std::string& args);

    // helpers
    static std::string trim(const std::string& s);
    static std::string lstripSlash(const std::string& s);
    static std::string toLower(std::string s);
    static bool startsWith(const std::string& s, const std::string& prefix);
};

