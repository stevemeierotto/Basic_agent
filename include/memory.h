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
#include <json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

class Memory {
private:
    std::string filepath;
    json data;

    std::string getDefaultPath() const;

public:
    // Construct with optional path; defaults to ~/.code_agent_plugin/memory.json (Linux/macOS)
    explicit Memory(const std::string& path = "");

    // Persistence
    void load();
    void save() const;

    // Conversation
    void addMessage(const std::string& role, const std::string& content);
    std::vector<json> getConversation() const;
    void clear();

    // Summaries
    void setSummary(const std::string& summary);  // sets short_summary only
    std::string getSummary(bool useExtended = false) const;
    void updateSummary(const std::string& goal, const std::string& response);

    // Debug helpers
    std::string getFilePath() const { return filepath; }
    void printSummaries() const;

    // NOTE: In the future, you could integrate an LLM call here to generate
    // smarter short/extended summaries automatically instead of truncating.
};

