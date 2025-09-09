#include "../include/prompt_factory.h"
#include <sstream>
#include <json.hpp>
#include <iostream>

PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r, size_t lastN)
    : memory(mem), rag(r), recentMessages(lastN) {}

std::string PromptFactory::buildConversationPrompt(const std::string& user_input,
                                                   bool useExtendedSummary) {
    std::ostringstream oss;

    // Memory summary
    oss << "[Memory Context]\n"
        << memory.getSummary(useExtendedSummary) << "\n\n";

    // Last N conversation turns
    auto convo = memory.getConversation();
    int start = convo.size() > recentMessages ? convo.size() - recentMessages : 0;
    for (size_t i = start; i < convo.size(); i++) {
        oss << "[" << convo[i]["role"] << "] " << convo[i]["content"] << "\n";
    }

    // New user input
    oss << "\n[User] " << user_input << "\n[Agent] ";
    return oss.str();
}

std::string PromptFactory::buildRagQueryPrompt(const std::string& query) {
    std::ostringstream oss;
    oss << "You are a  context retriever. Find relevant context for the following query:\n";
    oss << query << "\n";
    return oss.str();
}

