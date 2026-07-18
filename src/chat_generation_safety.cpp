/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan N chat generation safety (sanitize)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_generation_safety.h"
#include "../include/llm_interface.h"

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace Thoth {
namespace ChatGeneration {
namespace {

// UTF-8 for U+1F4DD (📝)
constexpr std::string_view kNoteEmoji = "\xF0\x9F\x93\x9D";

bool isHorizontalWs(char c) {
    return c == ' ' || c == '\t';
}

bool isAllWhitespace(std::string_view s) {
    for (unsigned char c : s) {
        if (!std::isspace(c)) {
            return false;
        }
    }
    return true;
}

bool isScaffoldContent(std::string_view line) {
    std::size_t i = 0;
    while (i < line.size() && isHorizontalWs(line[i])) {
        ++i;
    }
    const std::string_view body = line.substr(i);
    if (body.empty()) {
        return false;
    }
    return body.starts_with("[User]") || body.starts_with("[Agent]")
           || body.starts_with("[RAG Context]") || body.starts_with(kNoteEmoji);
}

struct LineSpan {
    std::size_t begin = 0;
    std::size_t end = 0; // exclusive end of line content (before \n)
    bool blank = false;
    bool scaffold = false;
};

std::vector<LineSpan> splitLines(std::string_view raw) {
    std::vector<LineSpan> lines;
    std::size_t i = 0;
    while (i < raw.size()) {
        LineSpan span;
        span.begin = i;
        while (i < raw.size() && raw[i] != '\n') {
            ++i;
        }
        span.end = i;
        std::size_t content_end = span.end;
        if (content_end > span.begin && raw[content_end - 1] == '\r') {
            --content_end;
        }
        const std::string_view content = raw.substr(span.begin, content_end - span.begin);
        span.blank = isAllWhitespace(content);
        span.scaffold = !span.blank && isScaffoldContent(content);
        lines.push_back(span);
        if (i < raw.size() && raw[i] == '\n') {
            ++i;
        }
    }
    return lines;
}

} // namespace

SanitizeOutcome sanitizeChatAssistantText(std::string_view raw) {
    SanitizeOutcome out;

    if (raw.empty() || isAllWhitespace(raw)) {
        out.sanitize_reason = kSanitizeNone;
        out.empty_after_sanitize = true;
        return out;
    }

    const std::vector<LineSpan> lines = splitLines(raw);
    if (lines.empty()) {
        out.sanitize_reason = kSanitizeNone;
        out.empty_after_sanitize = true;
        return out;
    }

    bool any_non_scaffold_content = false;
    for (const LineSpan& line : lines) {
        if (!line.blank && !line.scaffold) {
            any_non_scaffold_content = true;
            break;
        }
    }
    if (!any_non_scaffold_content) {
        out.sanitize_reason = kSanitizeAllScaffold;
        out.empty_after_sanitize = true;
        return out;
    }

    std::size_t start_idx = 0;
    bool stripped_scaffold = false;
    while (start_idx < lines.size()) {
        if (lines[start_idx].blank) {
            ++start_idx;
            continue;
        }
        if (lines[start_idx].scaffold) {
            stripped_scaffold = true;
            ++start_idx;
            continue;
        }
        break;
    }

    if (start_idx >= lines.size()) {
        out.sanitize_reason = kSanitizeAllScaffold;
        out.empty_after_sanitize = true;
        return out;
    }

    std::size_t end_idx = lines.size();
    bool truncated = false;
    for (std::size_t i = start_idx; i < lines.size(); ++i) {
        if (lines[i].scaffold) {
            end_idx = i;
            truncated = true;
            break;
        }
    }

    if (end_idx <= start_idx) {
        out.sanitize_reason = kSanitizeAllScaffold;
        out.empty_after_sanitize = true;
        return out;
    }

    std::size_t last_kept = end_idx - 1;
    while (last_kept > start_idx && lines[last_kept].blank) {
        --last_kept;
    }

    out.sanitized_text.assign(
        raw.substr(lines[start_idx].begin, lines[last_kept].end - lines[start_idx].begin));

    if (isAllWhitespace(out.sanitized_text)) {
        out.sanitized_text.clear();
        out.sanitize_reason = kSanitizeAllScaffold;
        out.empty_after_sanitize = true;
        return out;
    }

    out.empty_after_sanitize = false;
    if (truncated) {
        out.sanitize_reason = kSanitizeTruncatedTranscriptMarker;
    } else if (stripped_scaffold) {
        out.sanitize_reason = kSanitizeStrippedLeadingScaffold;
    } else {
        out.sanitize_reason = kSanitizeNone;
    }
    return out;
}

namespace {

void applyProviderAttempt(ChatGenerationResult& out,
                          const InferenceGenerateResult& generated,
                          bool used_stops) {
    out.provider_ok = generated.ok;
    out.provider_error = generated.ok ? std::string() : generated.error;
    out.raw_text = generated.ok ? generated.text : std::string();
    out.finish_reason = generated.finish_reason;
    out.stop_triggered = (generated.finish_reason == "stop");
    out.used_stops = used_stops;
}

void applySanitize(ChatGenerationResult& out) {
    const SanitizeOutcome sanitized = sanitizeChatAssistantText(out.raw_text);
    out.sanitized_text = sanitized.sanitized_text;
    out.sanitize_reason = sanitized.sanitize_reason;
    out.empty_after_sanitize = sanitized.empty_after_sanitize;
}

} // namespace

ChatGenerationResult generateAndSanitizeChat(LLMInterface& llm,
                                             const std::string& prompt,
                                             const ChatGenerateOptions& opts) {
    ChatGenerationResult out;

    const InferenceGenerateResult first =
        llm.queryDetailed(prompt, opts.max_tokens, opts.stop_sequences);
    applyProviderAttempt(out, first, !opts.stop_sequences.empty());
    if (!first.ok) {
        out.sanitized_text.clear();
        out.empty_after_sanitize = true;
        return out;
    }

    applySanitize(out);
    if (!out.empty_after_sanitize && !out.sanitized_text.empty()) {
        return out;
    }

    // Class B or C — one retry without stops.
    out.retried_without_stops = true;
    const InferenceGenerateResult second = llm.queryDetailed(prompt, opts.max_tokens, {});
    applyProviderAttempt(out, second, false);
    if (!second.ok) {
        out.sanitized_text.clear();
        out.empty_after_sanitize = true;
        return out;
    }

    applySanitize(out);
    if (!out.empty_after_sanitize && !out.sanitized_text.empty()) {
        return out;
    }

    out.fallback_used = true;
    out.sanitized_text =
        opts.use_greeting_fallback ? kFallbackGreeting : kFallbackGeneric;
    out.empty_after_sanitize = false;
    return out;
}

} // namespace ChatGeneration
} // namespace Thoth
