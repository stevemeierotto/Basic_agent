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
    return fh.getLogsPath("chat_rag.jsonl");
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

    nlohmann::json j = {
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
        {"presentation_mode", record.presentation_mode},
        {"inference_mode", record.inference_mode},
        {"retrieval_ran", record.retrieval_ran},
        {"retrieval_skip_reason", record.retrieval_skip_reason},
        {"retrieval_latency_ms", record.retrieval_latency_ms},
        {"prompt_build_latency_ms", record.prompt_build_latency_ms},
        {"candidates_found", record.candidates_found},
        {"candidates_passed_gate", record.candidates_passed_gate},
        {"grounding_decision_reason", record.grounding_decision_reason},
        {"grounded", record.grounded},
        {"max_score", record.has_candidate_scores ? nlohmann::json(record.max_score)
                                                  : nlohmann::json(nullptr)},
        {"min_injected_score", record.has_injected_scores
                                   ? nlohmann::json(record.min_injected_score)
                                   : nlohmann::json(nullptr)},
        {"generation_max_tokens", record.generation_max_tokens},
        {"chat_stop_sequence_count", record.chat_stop_sequence_count},
    };
    if (!record.final_prompt.empty()) {
        j["final_prompt"] = record.final_prompt;
    }
    if (record.chat_messages.is_array() && !record.chat_messages.empty()) {
        j["chat_messages"] = record.chat_messages;
    }
    if (!record.retrieval_trace.is_null() && !record.retrieval_trace.empty()) {
        j["retrieval_trace"] = record.retrieval_trace;
    }
    return j;
}

nlohmann::json ChatRagLogger::responseToJson(const ChatRagResponseRecord& record) {
    nlohmann::json attempts = nlohmann::json::array();
    for (const auto& attempt : record.generation_attempts) {
        attempts.push_back({
            {"attempt", attempt.attempt},
            {"latency_ms", attempt.latency_ms},
            {"prompt_tokens", attempt.prompt_tokens},
            {"completion_tokens", attempt.completion_tokens},
            {"finish_reason", attempt.finish_reason},
            {"provider_ok", attempt.provider_ok},
            {"raw_answer_chars", attempt.raw_answer_chars},
            {"sanitize_reason", attempt.sanitize_reason},
            {"sanitized_answer_chars", attempt.sanitized_answer_chars},
            {"empty_after_sanitize", attempt.empty_after_sanitize},
            {"response_valid", attempt.response_valid},
            {"invalid_reason", attempt.invalid_reason},
            {"transcript_user_marker_count", attempt.transcript_user_marker_count},
            {"transcript_agent_marker_count", attempt.transcript_agent_marker_count},
            {"max_tokens_requested", attempt.max_tokens_requested},
            {"raw_sample_first", attempt.raw_sample_first},
            {"raw_sample_last", attempt.raw_sample_last},
        });
        if (!attempt.raw_completion.empty()) {
            attempts.back()["raw_completion"] = attempt.raw_completion;
        }
        if (!attempt.sanitized_completion.empty()) {
            attempts.back()["sanitized_completion"] = attempt.sanitized_completion;
        }
    }

    nlohmann::json j = {
        {"event", "CHAT_RAG_RESPONSE"},
        {"request_id", record.request_id},
        {"answer_chars", record.answer_chars},
        {"retrieved_doc_count", record.retrieved_doc_count},
        {"grounding_mode", record.grounding_mode},
        {"fallback_used", record.fallback_used},
        {"raw_answer_chars", record.raw_answer_chars},
        {"sanitized_answer_chars", record.sanitized_answer_chars},
        {"sanitize_reason", record.sanitize_reason},
        {"retried_without_stops", record.retried_without_stops},
        {"retry_due_to_regurgitation", record.retry_due_to_regurgitation},
        {"regurgitation_retry_reason", record.regurgitation_retry_reason},
        {"regurgitation_detected", record.regurgitation_detected},
        {"regurgitation_score", record.regurgitation_score},
        {"used_stops", record.used_stops},
        {"provider_ok", record.provider_ok},
        {"finish_reason", record.finish_reason},
        {"turn_started_at_ms", record.turn_started_at_ms},
        {"turn_finished_at_ms", record.turn_finished_at_ms},
        {"turn_total_ms", record.turn_total_ms},
        {"queue_wait_ms", record.queue_wait_ms},
        {"session_setup_ms", record.session_setup_ms},
        {"retrieval_latency_ms", record.retrieval_latency_ms},
        {"prompt_build_latency_ms", record.prompt_build_latency_ms},
        {"generation_latency_ms", record.generation_latency_ms},
        {"post_processing_latency_ms", record.post_processing_latency_ms},
        {"telemetry_accounted_ms", record.telemetry_accounted_ms},
        {"telemetry_unaccounted_ms", record.telemetry_unaccounted_ms},
        {"worker_turn_total_ms", record.worker_turn_total_ms},
        {"generation_attempt_count", record.generation_attempt_count},
        {"generation_attempts", attempts},
        {"prompt_tokens", record.prompt_tokens},
        {"completion_tokens", record.completion_tokens},
        {"response_valid", record.response_valid},
        {"invalid_reason", record.invalid_reason},
        {"final_answer_chars", record.final_answer_chars},
        {"generation_max_tokens", record.generation_max_tokens},
    };
    if (!record.task_id.empty()) {
        j["task_id"] = record.task_id;
    }
    if (!record.raw_sample_first.empty()) {
        j["raw_sample_first"] = record.raw_sample_first;
    }
    if (!record.raw_sample_last.empty()) {
        j["raw_sample_last"] = record.raw_sample_last;
    }
    if (!record.final_answer.empty()) {
        j["final_answer"] = record.final_answer;
    }
    return j;
}

void ChatRagLogger::logContext(const ChatRagContextRecord& record) const {
    appendJsonLine(contextToJson(record));
}

void ChatRagLogger::logResponse(const ChatRagResponseRecord& record) const {
    appendJsonLine(responseToJson(record));
}

} // namespace Thoth
