/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan N chat generation safety (sanitize + ChatGenerationResult)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_GENERATION_SAFETY_H
#define THOTH_CHAT_GENERATION_SAFETY_H

#include "chat_prompt_config.h"
#include "inference_types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class LLMInterface;

namespace Thoth {
namespace ChatGeneration {

/** Plan N sanitize_reason values. */
inline constexpr const char* kSanitizeNone = "none";
inline constexpr const char* kSanitizeTruncatedTranscriptMarker = "truncated_transcript_marker";
inline constexpr const char* kSanitizeStrippedLeadingScaffold = "stripped_leading_scaffold";
inline constexpr const char* kSanitizeAllScaffold = "all_scaffold";
inline constexpr const char* kSanitizeStrippedChunkScaffold = "stripped_chunk_scaffold";
inline constexpr const char* kSanitizeAllChunkScaffold = "all_chunk_scaffold";

/** CSG-B regurgitation_retry_reason values (§10). */
inline constexpr const char* kRegurgitationRetryReasonNone = "none";
inline constexpr const char* kRegurgitationRetryReasonScaffoldRemaining = "scaffold_remaining";
inline constexpr const char* kRegurgitationRetryReasonPastedContextAfterStrip =
    "pasted_context_after_strip";
inline constexpr const char* kRegurgitationRetryReasonScaffoldAndPaste = "scaffold_and_paste";

/** Plan N L8 fallback copy. */
inline constexpr const char* kFallbackGreeting = "I'm here — how can I help?";
inline constexpr const char* kFallbackGeneric = "I couldn't generate a reply.";

struct SanitizeOutcome {
    std::string sanitized_text;
    std::string sanitize_reason = kSanitizeNone;
    bool empty_after_sanitize = false;
};

/**
 * Pure transcript-scaffold cleanup (Plan N N1).
 * Deterministic, side-effect free — no inference, memory, HTTP, or JSON parsing.
 */
SanitizeOutcome sanitizeChatAssistantText(std::string_view raw);

/** CSG-B §5 — Layer 1 scaffold assessment. */
struct RegurgitationAssessment {
    bool detected = false;
    float score = 0.0f;
    int document_header_count = 0;
    int source_span_count = 0;
    int scaffold_separator_count = 0;
};

RegurgitationAssessment assessChunkFormatRegurgitation(std::string_view text);

/** CSG-B §6 — Layer 2 answer-quality assessment (post-scaffold). */
struct AnswerQualityAssessment {
    bool complete_assistant_answer = false;
    bool pasted_retrieval_context = false;
};

AnswerQualityAssessment assessAnswerQuality(std::string_view text);

/** Phase 1 — response validity / acceptance gate. */
struct ResponseValidityAssessment {
    bool valid = true;
    std::string invalid_reason = "none";
};

inline constexpr const char* kInvalidReasonNone = "none";
inline constexpr const char* kInvalidReasonEmpty = "empty";
inline constexpr const char* kInvalidReasonQueryEcho = "query_echo";
inline constexpr const char* kInvalidReasonTruncateShortPrefix = "truncate_short_prefix";
inline constexpr const char* kInvalidReasonProviderError = "provider_error";
inline constexpr const char* kInvalidReasonNoUsableGeneration = "no_usable_generation";

/** Phase 1 — max queryDetailed() calls per chat turn (hard cap). */
inline constexpr int kMaxChatGenerationAttempts = 2;

ResponseValidityAssessment assessResponseValidity(std::string_view sanitized_text,
                                                  std::string_view user_query,
                                                  std::string_view sanitize_reason);

/** Back-compat alias — same semantics as assessResponseValidity. */
inline ResponseValidityAssessment assessResponseValidityForTelemetry(
    std::string_view sanitized_text,
    std::string_view user_query,
    std::string_view sanitize_reason) {
    return assessResponseValidity(sanitized_text, user_query, sanitize_reason);
}

/** CSG-B §4 — strip chunk injection scaffold lines; preserve body prose. */
SanitizeOutcome sanitizeChunkFormatScaffold(std::string_view raw);

/** Phase 0 — one row per actual queryDetailed() invocation. */
struct GenerationAttemptTelemetry {
    int attempt = 0;
    std::int64_t latency_ms = 0;
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    std::string finish_reason;
    bool provider_ok = false;
    std::size_t raw_answer_chars = 0;
    std::string sanitize_reason;
    /** Investigation — post-sanitize size before validity gate (no content change). */
    std::size_t sanitized_answer_chars = 0;
    bool empty_after_sanitize = false;
    bool response_valid = false;
    std::string invalid_reason = kInvalidReasonNone;
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

/** Phase 0 — short head/tail sample when THOTH_LOG_RAW_CHAT_COMPLETION=1. */
struct RawCompletionSample {
    std::string first;
    std::string last;
};

RawCompletionSample buildRawCompletionSample(std::string_view raw_text);

/** Investigation — log full final prompt in CHAT_RAG_CONTEXT when THOTH_LOG_CHAT_PROMPT=1. */
bool chatPromptLoggingEnabled();

/** Investigation — log full raw + sanitized per attempt when THOTH_LOG_FULL_RAW_CHAT_COMPLETION=1. */
bool chatFullRawLoggingEnabled();

struct ChatGenerationResult {
    std::string raw_text;
    std::string sanitized_text;
    std::string finish_reason;
    bool stop_triggered = false;
    std::string sanitize_reason = kSanitizeNone;
    bool empty_after_sanitize = false;
    bool used_stops = false;
    bool retried_without_stops = false;
    bool retry_due_to_regurgitation = false;
    std::string regurgitation_retry_reason = kRegurgitationRetryReasonNone;
    bool regurgitation_detected = false;
    float regurgitation_score = 0.0f;
    bool fallback_used = false;
    /** Phase 1 — whether the returned assistant text passed the acceptance gate. */
    bool response_valid = false;
    std::string invalid_reason = kInvalidReasonNone;
    bool provider_ok = false;
    std::string provider_error;
    /** Actual queryDetailed() call count for this turn. */
    int generation_attempt_count = 0;
    std::int64_t generation_latency_ms = 0;
    std::vector<GenerationAttemptTelemetry> generation_attempts;
    RawCompletionSample raw_sample;
};

/** Plan N N2 — options for generateAndSanitizeChat. */
struct ChatGenerateOptions {
    int max_tokens = ChatPrompt::kChatMaxTokens;
    std::vector<std::string> stop_sequences;
    bool use_greeting_fallback = false;
    /** Phase 1 — user query for echo / validity checks. */
    std::string user_query;
    /** Phase A — when set, use /v1/chat/completions instead of flat prompt completion. */
    std::optional<Thoth::InferenceChatRequest> chat_request;
};

/**
 * Plan N N2 + CSG-B — structured chat generate + sanitize + retries + L8 fallback.
 * Does not write memory or format UI errors.
 */
ChatGenerationResult generateAndSanitizeChat(LLMInterface& llm,
                                             const std::string& prompt,
                                             const ChatGenerateOptions& opts);

} // namespace ChatGeneration
} // namespace Thoth

#endif // THOTH_CHAT_GENERATION_SAFETY_H
