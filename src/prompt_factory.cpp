#include "../include/prompt_factory.h"
#include <sstream>
#include <iostream>

// Default constructor uses default PromptConfig()
PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r)
    : PromptFactory(mem, r, PromptConfig()) {}

// Constructor with explicit config
PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r, const PromptConfig& cfg)
    : memory(mem), rag(r), config(cfg) {}

std::string PromptFactory::buildConversationPrompt(const std::string& user_input,
                                                   bool useExtendedSummary) {
    std::ostringstream oss;

    // ---- 1. Optional system prompt ----
    if (!config.systemPrompt.empty()) {
        oss << config.systemPrompt << config.conversationSeparator << "\n";
    }

    // ---- 2. Memory summary (limited) ----
    std::string memSummary = memory.getSummary(useExtendedSummary);

    // Truncate memory summary to half of maxContextLength to avoid huge prompt
    const size_t maxMemLen = config.maxContextLength / 2;
    if (memSummary.length() > maxMemLen) {
        memSummary = memSummary.substr(memSummary.length() - maxMemLen);
    }
    oss << "[Memory Context]\n" << memSummary << "\n\n";

    // ---- 3. Last N conversation turns (limited) ----
    auto convo = memory.getConversation();
    size_t start = convo.size() > config.maxRecentMessages
                     ? convo.size() - config.maxRecentMessages
                     : 0;

    // Estimate target length for conversation portion
    const size_t maxConvoLen = config.maxContextLength / 2;

    size_t convoLen = 0;
    for (size_t i = start; i < convo.size(); i++) {
        std::ostringstream turn;
        if (config.includeRoleLabels) {
            turn << "[" << convo[i]["role"] << "] ";
        }
        turn << convo[i]["content"];
        if (config.includeTimestamps && convo[i].contains("timestamp")) {
            turn << " (" << convo[i]["timestamp"] << ")";
        }
        turn << config.conversationSeparator;

        convoLen += turn.str().length();
        if (convoLen > maxConvoLen) break; // stop adding more conversation if too long

        oss << turn.str();
    }

    // ---- 4. New user input ----
    oss << "[User] " << user_input << "\n[Agent] ";

    // ---- 5. Final truncate ----
    return truncateToLimit(oss.str(), config.maxContextLength);
}


std::string PromptFactory::buildRagQueryPrompt(const std::string& query) {
    std::ostringstream oss;
    oss << "You are a context retriever. Find relevant context for the following query:\n";
    oss << query << "\n";
    return truncateToLimit(oss.str(), config.maxContextLength);
}

std::string PromptFactory::truncateToLimit(const std::string& input, size_t maxLen) const {
    if (input.size() <= maxLen) return input;
    return input.substr(input.size() - maxLen); // Keep last maxLen chars
}

