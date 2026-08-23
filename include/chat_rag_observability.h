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
    /** Phase 0 — ms spent in retrieval + session-goal embed before prompt build. */
    std::int64_t retrieval_latency_ms = 0;
    std::int64_t prompt_build_latency_ms = 0;
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
    /** Investigation — labeled | metadata_off (THOTH_CHAT_RAG_PRESENTATION). */
    std::string presentation_mode = "labeled";
    /** Phase A — completions | chat (THOTH_CHAT_INFERENCE_MODE). */
    std::string inference_mode = "completions";
    /** Investigation — serialized chat messages when THOTH_LOG_CHAT_PROMPT=1 and mode=chat. */
    nlohmann::json chat_messages = nlohmann::json::array();

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
    nlohmann::json retrieval_trace = nlohmann::json::object();
    /** Investigation — chat generation request parameters (not prompt text). */
    int generation_max_tokens = 0;
    int chat_stop_sequence_count = 0;
    /** Full final prompt when THOTH_LOG_CHAT_PROMPT=1 (investigation only). */
    std::string final_prompt;
};

/** Phase 0 — one row per actual LLM queryDetailed() call. */
struct ChatGenerationAttemptRecord {
    int attempt = 0;
    std::int64_t latency_ms = 0;
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    std::string finish_reason;
    bool provider_ok = false;
    std::size_t raw_answer_chars = 0;
    std::string sanitize_reason;
    std::size_t sanitized_answer_chars = 0;
    bool empty_after_sanitize = false;
    bool response_valid = false;
    std::string invalid_reason;
    int transcript_user_marker_count = 0;
    int transcript_agent_marker_count = 0;
    int max_tokens_requested = 0;
    std::string raw_sample_first;
    std::string raw_sample_last;
    /** Investigation — full provider text when THOTH_LOG_FULL_RAW_CHAT_COMPLETION=1. */
    std::string raw_completion;
    /** Investigation — post-sanitize text for this attempt (same env gate). */
    std::string sanitized_completion;
};

struct ChatRagResponseRecord {
    std::string request_id;
    std::size_t answer_chars = 0;
    int retrieved_doc_count = 0;
    std::string grounding_mode;
    bool fallback_used = false;

    // Plan N N6 — generation diagnostics (flags / reasons / counts only; no raw text).
    std::size_t raw_answer_chars = 0;
    std::size_t sanitized_answer_chars = 0;
    std::string sanitize_reason;
    bool retried_without_stops = false;
    bool retry_due_to_regurgitation = false;
    std::string regurgitation_retry_reason;
    bool regurgitation_detected = false;
    float regurgitation_score = 0.0f;
    bool used_stops = false;
    bool provider_ok = false;
    std::string finish_reason;

    // Phase 0 — turn timing / reconciliation (telemetry only; no behavior change).
    std::int64_t turn_started_at_ms = 0;
    std::int64_t turn_finished_at_ms = 0;
    std::int64_t turn_total_ms = 0;
    std::int64_t queue_wait_ms = 0;
    std::int64_t session_setup_ms = 0;
    std::int64_t retrieval_latency_ms = 0;
    std::int64_t prompt_build_latency_ms = 0;
    std::int64_t generation_latency_ms = 0;
    std::int64_t post_processing_latency_ms = 0;
    std::int64_t telemetry_accounted_ms = 0;
    std::int64_t telemetry_unaccounted_ms = 0;
    /** queue_wait_ms + turn_total_ms — full worker-path span from enqueue to response. */
    std::int64_t worker_turn_total_ms = 0;
    int generation_attempt_count = 0;
    std::vector<ChatGenerationAttemptRecord> generation_attempts;
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    bool response_valid = true;
    std::string invalid_reason = "none";
    /** Populated only when THOTH_LOG_RAW_CHAT_COMPLETION=1 (short samples, not full text). */
    std::string raw_sample_first;
    std::string raw_sample_last;
    /** Investigation — chars returned to caller after validity gate / fallback. */
    std::size_t final_answer_chars = 0;
    /** Investigation — text returned to caller when THOTH_LOG_FULL_RAW_CHAT_COMPLETION=1. */
    std::string final_answer;
    int generation_max_tokens = 0;
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
