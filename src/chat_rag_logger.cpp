/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 0: append-only chat RAG observability log
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_rag_observability.h"
#include "file_handler.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace fs = std::filesystem;

namespace Thoth {

ChatRagLogger& ChatRagLogger::instance() {
    static ChatRagLogger logger;
    return logger;
}

ChatRagLogger::ChatRagLogger() = default;

std::string ChatRagLogger::logFilePath() const {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "chat_rag.jsonl").string();
}

void ChatRagLogger::appendJsonLine(const nlohmann::json& event) const {
    static std::mutex writeMutex;
    std::lock_guard<std::mutex> lock(writeMutex);

    nlohmann::json line = event;
    line["emitted_at_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();

    std::ofstream out(logFilePath(), std::ios::app);
    if (out.is_open()) {
        out << line.dump() << '\n';
    }
}

nlohmann::json ChatRagLogger::contextToJson(const ChatRagContextRecord& record) {
    nlohmann::json docs = nlohmann::json::array();
    nlohmann::json ranked = nlohmann::json::array();
    for (const auto& doc : record.documents) {
        docs.push_back({
            {"file", doc.file},
            {"chunk_id", doc.chunk_id},
            {"start_line", doc.start_line},
            {"end_line", doc.end_line},
            {"score", doc.score},
            {"chars", doc.chars},
            {"rank", doc.rank},
        });
        ranked.push_back({
            {"rank", doc.rank},
            {"file", doc.file},
            {"score", doc.score},
        });
    }

    return {
        {"event", "CHAT_RAG_CONTEXT"},
        {"request_id", record.request_id},
        {"query", record.query},
        {"documents", docs},
        {"ranked_documents", ranked},
        {"top_k", record.top_k},
        {"retrieved_chars", record.retrieved_chars},
        {"conversation_history_chars", record.conversation_history_chars},
        {"tool_schema_chars", record.tool_schema_chars},
        {"memory_context_chars", record.memory_context_chars},
        {"system_prompt_chars", record.system_prompt_chars},
        {"rag_wrapper_chars", record.rag_wrapper_chars},
        {"assembled_conversation_chars", record.assembled_conversation_chars},
        {"final_prompt_chars", record.final_prompt_chars},
        {"prompt_before_truncation_chars", record.prompt_before_truncation_chars},
        {"prompt_after_truncation_chars", record.prompt_after_truncation_chars},
        {"truncated", record.truncated},
        {"truncated_section", record.truncated_section},
        {"grounding_ratio", record.grounding_ratio},
        {"tool_ratio", record.tool_ratio},
        {"history_ratio", record.history_ratio},
        {"memory_ratio", record.memory_ratio},
        {"llm_model", record.llm_model},
        {"grounding_mode", record.grounding_mode},
    };
}

nlohmann::json ChatRagLogger::responseToJson(const ChatRagResponseRecord& record) {
    return {
        {"event", "CHAT_RAG_RESPONSE"},
        {"request_id", record.request_id},
        {"answer_chars", record.answer_chars},
        {"retrieved_doc_count", record.retrieved_doc_count},
        {"grounding_mode", record.grounding_mode},
        {"fallback_used", record.fallback_used},
    };
}

void ChatRagLogger::logContext(const ChatRagContextRecord& record) const {
    appendJsonLine(contextToJson(record));
}

void ChatRagLogger::logResponse(const ChatRagResponseRecord& record) const {
    appendJsonLine(responseToJson(record));
}

} // namespace Thoth
