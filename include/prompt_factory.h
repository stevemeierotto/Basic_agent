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

class PromptFactory {
public:
    struct PromptConfig {
        size_t maxRecentMessages = 5;
        size_t maxContextLength = 4000;
        bool includeTimestamps = false;
        bool includeRoleLabels = true;
        std::string systemPrompt = "";
        std::string conversationSeparator = "\n";
    };

private:
    Memory& memory;
    RAGPipeline& rag;
    PromptConfig config;

public:
    // Overloaded constructors
    PromptFactory(Memory& mem, RAGPipeline& r);  
    PromptFactory(Memory& mem, RAGPipeline& r, const PromptConfig& cfg);

    void setConfig(const PromptConfig& cfg) { config = cfg; }
    PromptConfig getConfig() const { return config; }

    std::string buildConversationPrompt(const std::string& user_input,
                                        bool useExtendedSummary = false);

    std::string buildRagQueryPrompt(const std::string& query);

private:
    std::string truncateToLimit(const std::string& input, size_t maxLen) const;
};

