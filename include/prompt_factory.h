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
#include "planner_injection_config.h"
#include "chat_rag_observability.h"

class PromptFactory {
public:
    struct PromptConfig {
        size_t maxRecentMessages = 5;
        size_t maxContextLength = 4000;
        bool includeTimestamps = false;
        bool includeRoleLabels = true;
        std::string systemPrompt =
            "You are Thoth, a helpful assistant. Respond in clear natural language. "
            "Only emit JSON tool calls when the user explicitly asks you to run a tool.";
        std::string conversationSeparator = "\n";
        bool enableTools = true;
    };

    struct ConversationBuildOptions {
        bool includeTools = false;
        bool grounded = false;
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
                                        bool useExtendedSummary = false,
                                        Thoth::ConversationPromptMetrics* metrics = nullptr);

    /** Unified chat prompt: RAG context + section-protected conversation assembly. */
    std::string buildChatPrompt(const std::string& user_input,
                                const std::string& ragContext,
                                bool useExtendedSummary,
                                const ConversationBuildOptions& options,
                                Thoth::ConversationPromptMetrics* metrics = nullptr);

    std::string buildRagQueryPrompt(const std::string& query);

    std::string buildPlanPrompt(const std::string& goal,
                                const std::string& strategy_context,
                                const std::string& past_experience,
                                Thoth::PlannerPromptMetrics* metrics = nullptr);

    std::string buildRevisionPrompt(const std::string& goal,
                                    const std::string& existing_plan_json,
                                    const std::string& failed_step_result_json,
                                    Thoth::PlannerPromptMetrics* metrics = nullptr);

    /** Writes bundled plan templates into agent_workspace when missing. */
    static void ensureDefaultTemplatesExist();

private:
    std::string assembleConversationSections(const std::string& user_input,
                                             bool useExtendedSummary,
                                             const ConversationBuildOptions& options,
                                             std::size_t budget,
                                             Thoth::ConversationPromptMetrics* metrics);
    std::string fitRagContextToBudget(const std::string& ragContext, std::size_t maxChars);
    std::string truncateToLimit(const std::string& input, size_t maxLen) const;
    std::string loadTemplate(const std::string& filename, const std::string& defaultValue);
    std::string applySubstitutions(std::string templateStr, const std::unordered_map<std::string, std::string>& subs);
    
    std::string getToolList();
    std::string getMemoryContext(bool useExtendedSummary);
    std::string getConversationHistory();
};

