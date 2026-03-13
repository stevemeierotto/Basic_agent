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
        std::string systemPrompt = "Respond in natural language unless you need to perform an action. Only use the JSON tool format when a tool is strictly necessary.";
        std::string conversationSeparator = "\n";
        bool enableTools = true;
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

    std::string buildPlanPrompt(const std::string& goal,
                                const std::string& strategy_context,
                                const std::string& past_experience);

    std::string buildRevisionPrompt(const std::string& goal,
                                    const std::string& existing_plan_json,
                                    const std::string& failed_step_result_json);

private:
    std::string truncateToLimit(const std::string& input, size_t maxLen) const;
    std::string loadTemplate(const std::string& filename, const std::string& defaultValue);
    std::string applySubstitutions(std::string templateStr, const std::unordered_map<std::string, std::string>& subs);
    
    std::string getToolList();
    std::string getMemoryContext(bool useExtendedSummary);
    std::string getConversationHistory();
};

