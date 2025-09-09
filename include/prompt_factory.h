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
private:
    Memory& memory;
    RAGPipeline& rag;
    size_t recentMessages;

public:
    PromptFactory(Memory& mem, RAGPipeline& r, size_t lastN = 5);

    // Build a prompt with memory + recent conversation
    std::string buildConversationPrompt(const std::string& user_input,
                                        bool useExtendedSummary = false);

    // Build a prompt to query RAG for relevant code context
    std::string buildRagQueryPrompt(const std::string& query);
};

