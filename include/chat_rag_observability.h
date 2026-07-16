/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 0: Chat RAG observability types
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_RAG_OBSERVABILITY_H
#define THOTH_CHAT_RAG_OBSERVABILITY_H

#include "json.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace Thoth {

struct ConversationPromptMetrics {
    std::size_t system_prompt_chars = 0;
    std::size_t grounding_rules_chars = 0;
    std::size_t tool_schema_chars = 0;
    std::size_t tool_schema_chars_in_final = 0;
    bool tools_included = false;
    std::size_t memory_context_chars = 0;
    std::size_t conversation_history_chars = 0;
    std::size_t user_input_chars = 0;
    std::size_t rag_context_chars = 0;
    std::size_t assembled_prompt_chars = 0;
    std::size_t final_prompt_chars = 0;
    bool truncated = false;
    std::string truncated_section;
};

struct ChatRagDocumentMetric {
    int rank = 0;
    std::string file;
    int chunk_id = 0;
    int start_line = 0;
    int end_line = 0;
    float score = 0.0f;
    std::size_t chars = 0;
};

struct ChatRagContextRecord {
    std::string request_id;
    std::string query;
    std::vector<ChatRagDocumentMetric> documents;
    int top_k = 0;
    std::size_t retrieved_chars = 0;
    std::size_t conversation_history_chars = 0;
    std::size_t tool_schema_chars = 0;
    std::size_t memory_context_chars = 0;
    std::size_t system_prompt_chars = 0;
    std::size_t rag_wrapper_chars = 0;
    std::size_t assembled_conversation_chars = 0;
    std::size_t final_prompt_chars = 0;
    std::size_t prompt_before_truncation_chars = 0;
    std::size_t prompt_after_truncation_chars = 0;
    bool truncated = false;
    std::string truncated_section;
    float grounding_ratio = 0.0f;
    float tool_ratio = 0.0f;
    float history_ratio = 0.0f;
    float memory_ratio = 0.0f;
    std::string llm_model;
    std::string grounding_mode;

    // Plan M G1 (R1) — grounding gate telemetry (attempt vs success).
    bool retrieval_ran = false;
    std::string retrieval_skip_reason = "none";  // none | greeting | no_index
    int candidates_found = 0;                     // pre-floor candidate count
    int candidates_passed_gate = 0;               // post-floor (injected) count
    std::string grounding_decision_reason;        // injected_meaningful_hits | below_threshold
                                                  // | greeting_skip | empty_index | no_candidates
    bool grounded = false;                        // aligned with grounding_mode
    bool has_candidate_scores = false;            // true when max_score is meaningful
    float max_score = 0.0f;                        // max finite candidate score
    bool has_injected_scores = false;             // true when min_injected_score is meaningful
    float min_injected_score = 0.0f;               // min injected score
};

struct ChatRagResponseRecord {
    std::string request_id;
    std::size_t answer_chars = 0;
    int retrieved_doc_count = 0;
    std::string grounding_mode;
    bool fallback_used = false;
};

class ChatRagLogger {
public:
    static ChatRagLogger& instance();

    void logContext(const ChatRagContextRecord& record) const;
    void logResponse(const ChatRagResponseRecord& record) const;

    static nlohmann::json contextToJson(const ChatRagContextRecord& record);
    static nlohmann::json responseToJson(const ChatRagResponseRecord& record);

private:
    ChatRagLogger();
    std::string logFilePath() const;
    void appendJsonLine(const nlohmann::json& event) const;
};

} // namespace Thoth

#endif // THOTH_CHAT_RAG_OBSERVABILITY_H
