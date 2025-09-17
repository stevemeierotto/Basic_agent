/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * Uses either Ollama local models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include <json.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>

using json = nlohmann::json;

class Memory {
private:
    std::string filepath;
    json data;

    mutable std::mutex mtx;     // protects data and dirty state
    mutable bool isDirty = false;
    mutable std::chrono::steady_clock::time_point lastSave;
    static constexpr auto AUTO_SAVE_INTERVAL = std::chrono::minutes(5);

    std::string getDefaultPath() const;
    void markDirty() const;
    void saveIfNeeded() const;
    void saveUnlocked() const;

public:
    // Constructor / Destructor
    explicit Memory(const std::string& path = "");
    ~Memory();

    // Persistence
    void load();            // loads from disk (overwrites memory)
    void save() const;      // flushes to disk if dirty
    void flush() const;     // unconditional save, clears dirty flag

    // Conversation
    void addMessage(const std::string& role, const std::string& content);
    void addMessages(const std::vector<std::pair<std::string, std::string>>& messages);
    std::vector<json> getConversation() const;
    void clear();

    // Summaries
    void setSummary(const std::string& summary);  // sets short_summary only
    std::string getSummary(bool useExtended = false) const;
    void updateSummary(const std::string& goal, const std::string& response);

    // Debug helpers
    std::string getFilePath() const { return filepath; }
    void printSummaries() const;
};

