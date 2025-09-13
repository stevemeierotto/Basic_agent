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
#include <vector>

class Tools {
public:
    virtual ~Tools() = default;

    // Example virtual interface for all tools
    virtual std::string summarizeText(const std::string& text, int numSentences = 3) = 0;

    // You can add generic methods, or keep them pure virtual for specialization
};
