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

/**
 * Full chat generation envelope (filled by N2+). N1 defines the type with safe defaults.
 */
struct ChatGenerationResult {
    std::string raw_text;
    std::string sanitized_text;
    std::string finish_reason;
    bool stop_triggered = false;
    std::string sanitize_reason = kSanitizeNone;
    bool empty_after_sanitize = false;
    bool used_stops = false;
    bool retried_without_stops = false;
    bool fallback_used = false;
    bool provider_ok = false;
    std::string provider_error;
};

/** Plan N N2 — options for generateAndSanitizeChat. */
struct ChatGenerateOptions {
    int max_tokens = ChatPrompt::kChatMaxTokens;
    std::vector<std::string> stop_sequences;
    bool use_greeting_fallback = false;
};

/**
 * Plan N N2 — structured chat generate + sanitize + one stop-free retry + L8 fallback.
 * Does not write memory or format UI errors.
 */
ChatGenerationResult generateAndSanitizeChat(LLMInterface& llm,
                                             const std::string& prompt,
                                             const ChatGenerateOptions& opts);

} // namespace ChatGeneration
} // namespace Thoth

#endif // THOTH_CHAT_GENERATION_SAFETY_H
