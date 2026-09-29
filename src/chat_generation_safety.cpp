/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Plan N + CSG-B chat generation safety
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_generation_safety.h"
#include "../include/llm_interface.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace Thoth {
namespace ChatGeneration {

namespace {

constexpr std::string_view kNoteEmoji = "\xF0\x9F\x93\x9D";
constexpr std::size_t kFragmentLineMaxChars = 60;

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

struct LlmAttemptContext {
    LLMInterface& llm;
    int max_tokens = 0;
    std::vector<std::string> stop_sequences;
    std::vector<GenerationAttemptTelemetry>& attempts;
    std::int64_t& generation_latency_ms;
};

InferenceGenerateResult runTrackedGeneration(LlmAttemptContext& ctx,
                                             const std::string& prompt,
                                             const ChatGenerateOptions& opts) {
    const auto startMs = nowMs();
    InferenceGenerateResult generated;
    if (opts.chat_request.has_value()) {
        Thoth::InferenceChatRequest chatReq = *opts.chat_request;
        chatReq.max_tokens = ctx.max_tokens;
        chatReq.stop_sequences = ctx.stop_sequences;
        generated = ctx.llm.queryDetailedChat(chatReq, ctx.max_tokens, ctx.stop_sequences);
    } else {
        generated = ctx.llm.queryDetailed(prompt, ctx.max_tokens, ctx.stop_sequences);
    }
    const auto latencyMs = nowMs() - startMs;
    ctx.generation_latency_ms += latencyMs;

    GenerationAttemptTelemetry row;
    row.attempt = static_cast<int>(ctx.attempts.size()) + 1;
    row.latency_ms = latencyMs;
    row.usage_unavailable = !generated.provider_usage_reported;
    row.prompt_tokens = generated.provider_usage_reported ? generated.token_usage.prompt_tokens : 0;
    row.completion_tokens = generated.provider_usage_reported ? generated.token_usage.completion_tokens : 0;
    row.finish_reason = generated.finish_reason;
    row.provider_ok = generated.ok;
    row.raw_answer_chars = generated.ok ? generated.text.size() : 0;
    ctx.attempts.push_back(std::move(row));

    return generated;
}

void attachAttemptSanitizeReason(std::vector<GenerationAttemptTelemetry>& attempts,
                                 const std::string& sanitize_reason) {
    if (!attempts.empty()) {
        attempts.back().sanitize_reason = sanitize_reason;
    }
}

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

std::string_view trimHorizontal(std::string_view line) {
    std::size_t begin = 0;
    while (begin < line.size() && isHorizontalWs(line[begin])) {
        ++begin;
    }
    std::size_t end = line.size();
    while (end > begin && isHorizontalWs(line[end - 1])) {
        --end;
    }
    return line.substr(begin, end - begin);
}

std::string_view lineContent(std::string_view raw, std::size_t begin, std::size_t end) {
    std::size_t content_end = end;
    if (content_end > begin && raw[content_end - 1] == '\r') {
        --content_end;
    }
    return raw.substr(begin, content_end - begin);
}

bool isTranscriptScaffoldContent(std::string_view line) {
    const std::string_view body = trimHorizontal(line);
    if (body.empty()) {
        return false;
    }
    return body.starts_with("[User]") || body.starts_with("[Agent]")
           || body.starts_with("[RAG Context]") || body.starts_with(kNoteEmoji);
}

bool isChunkScaffoldContent(std::string_view line) {
    const std::string_view body = trimHorizontal(line);
    if (body.empty()) {
        return false;
    }
    if (body.starts_with("Document:")) {
        return true;
    }
    if (body.starts_with("source_span=")) {
        return true;
    }
    return body == "---";
}

struct LineSpan {
    std::size_t begin = 0;
    std::size_t end = 0;
    bool blank = false;
    bool transcript_scaffold = false;
    bool chunk_scaffold = false;
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
        const std::string_view content = lineContent(raw, span.begin, span.end);
        span.blank = isAllWhitespace(content);
        span.transcript_scaffold = !span.blank && isTranscriptScaffoldContent(content);
        span.chunk_scaffold = !span.blank && isChunkScaffoldContent(content);
        lines.push_back(span);
        if (i < raw.size() && raw[i] == '\n') {
            ++i;
        }
    }
    return lines;
}

RegurgitationAssessment assessLines(const std::vector<LineSpan>& lines, std::string_view raw) {
    RegurgitationAssessment out;
    int non_blank = 0;
    int prose_lines = 0;
    std::size_t prose_chars = 0;
    std::size_t scaffold_chars = 0;
    bool seen_prose = false;
    bool scaffold_before_prose = false;

    for (const LineSpan& line : lines) {
        if (line.blank) {
            continue;
        }
        ++non_blank;
        const std::string_view content = lineContent(raw, line.begin, line.end);
        const std::string_view trimmed = trimHorizontal(content);
        const bool is_doc = trimmed.starts_with("Document:");
        const bool is_span = trimmed.starts_with("source_span=");
        const bool is_sep = trimmed == "---";
        const bool is_scaffold = is_doc || is_span || is_sep;

        if (is_doc) {
            ++out.document_header_count;
        }
        if (is_span) {
            ++out.source_span_count;
        }
        if (is_sep) {
            ++out.scaffold_separator_count;
        }

        if (is_scaffold) {
            scaffold_chars += trimmed.size();
            if (!seen_prose) {
                scaffold_before_prose = true;
            }
        } else {
            ++prose_lines;
            prose_chars += trimmed.size();
            seen_prose = true;
        }
    }

    if (out.document_header_count >= 2) {
        out.detected = true;
        out.score = std::max(out.score, 0.9f);
    }
    if (out.source_span_count >= 2) {
        out.detected = true;
        out.score = std::max(out.score, 0.85f);
    }
    if (out.document_header_count >= 1 && out.source_span_count >= 1) {
        if (prose_lines == 0 || prose_chars <= scaffold_chars) {
            out.detected = true;
            out.score = std::max(out.score, 0.8f);
        } else if (scaffold_before_prose) {
            out.detected = true;
            out.score = std::max(out.score, 0.75f);
        }
    }
    if (out.document_header_count >= 1 && out.scaffold_separator_count >= 1 && prose_lines <= 1) {
        out.detected = true;
        out.score = std::max(out.score, 0.7f);
    }
    if (non_blank > 0) {
        const float scaffold_ratio =
            static_cast<float>(out.document_header_count + out.source_span_count
                               + out.scaffold_separator_count)
            / static_cast<float>(non_blank);
        if (scaffold_ratio >= 0.5f && out.document_header_count >= 1 && out.source_span_count >= 1) {
            out.detected = true;
            out.score = std::max(out.score, scaffold_ratio);
        }
    }
    if (out.detected && out.score <= 0.0f) {
        out.score = 0.5f;
    }
    return out;
}

bool hasConversationalFraming(std::string_view text) {
    const std::vector<LineSpan> lines = splitLines(text);
    for (const LineSpan& line : lines) {
        if (line.blank) {
            continue;
        }
        const std::string_view trimmed = trimHorizontal(lineContent(text, line.begin, line.end));
        if (trimmed.starts_with("The ") || trimmed.starts_with("In ") || trimmed.starts_with("For ")
            || trimmed.starts_with("When ") || trimmed.starts_with("Sidebar")
            || trimmed.starts_with("Thoth")) {
            return true;
        }
        if (trimmed.find(" must ") != std::string_view::npos
            || trimmed.find(" requires ") != std::string_view::npos
            || trimmed.find(" should ") != std::string_view::npos
            || trimmed.find(" uses ") != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

int countSentenceBreaks(std::string_view text) {
    int breaks = 0;
    for (std::size_t i = 0; i + 1 < text.size(); ++i) {
        if (text[i] == '.' && text[i + 1] == ' ') {
            ++breaks;
        }
    }
    if (!text.empty() && text.back() == '.') {
        ++breaks;
    }
    return breaks;
}

bool hasFragmentCollage(const std::vector<LineSpan>& lines, std::string_view raw) {
    int consecutive_short = 0;
    for (const LineSpan& line : lines) {
        if (line.blank || line.chunk_scaffold) {
            consecutive_short = 0;
            continue;
        }
        const std::string_view trimmed = trimHorizontal(lineContent(raw, line.begin, line.end));
        if (trimmed.size() <= kFragmentLineMaxChars) {
            ++consecutive_short;
            if (consecutive_short >= 2) {
                return true;
            }
        } else {
            consecutive_short = 0;
        }
    }
    return false;
}

struct ProcessedAttempt {
    std::string sanitized_text;
    std::string sanitize_reason = kSanitizeNone;
    bool empty = false;
    bool acceptable = false;
    bool needs_regurgitation_retry = false;
    std::string regurgitation_retry_reason = kRegurgitationRetryReasonNone;
    RegurgitationAssessment scaffold;
};

ProcessedAttempt processGenerationPipeline(std::string_view raw) {
    ProcessedAttempt out;

    const SanitizeOutcome transcript = sanitizeChatAssistantText(raw);
    if (transcript.empty_after_sanitize) {
        out.empty = true;
        out.scaffold = assessChunkFormatRegurgitation(raw);
        out.regurgitation_retry_reason = kRegurgitationRetryReasonNone;
        return out;
    }

    std::string_view working = transcript.sanitized_text;
    out.sanitize_reason = transcript.sanitize_reason;
    out.scaffold = assessChunkFormatRegurgitation(working);

    if (!out.scaffold.detected) {
        out.sanitized_text.assign(working.begin(), working.end());
        out.acceptable = true;
        return out;
    }

    const SanitizeOutcome chunk = sanitizeChunkFormatScaffold(working);
    if (chunk.empty_after_sanitize) {
        out.empty = true;
        out.sanitize_reason = chunk.sanitize_reason;
        out.needs_regurgitation_retry = true;
        out.regurgitation_retry_reason = kRegurgitationRetryReasonPastedContextAfterStrip;
        return out;
    }

    out.sanitized_text = chunk.sanitized_text;
    if (chunk.sanitize_reason != kSanitizeNone) {
        out.sanitize_reason = chunk.sanitize_reason;
    }

    const RegurgitationAssessment after_strip =
        assessChunkFormatRegurgitation(out.sanitized_text);
    const AnswerQualityAssessment quality = assessAnswerQuality(out.sanitized_text);
    const bool scaffold_remaining = after_strip.detected;

    if (scaffold_remaining && quality.pasted_retrieval_context) {
        out.needs_regurgitation_retry = true;
        out.regurgitation_retry_reason = kRegurgitationRetryReasonScaffoldAndPaste;
    } else if (scaffold_remaining) {
        out.needs_regurgitation_retry = true;
        out.regurgitation_retry_reason = kRegurgitationRetryReasonScaffoldRemaining;
    } else if (quality.pasted_retrieval_context || !quality.complete_assistant_answer) {
        out.needs_regurgitation_retry = true;
        out.regurgitation_retry_reason = kRegurgitationRetryReasonPastedContextAfterStrip;
    } else {
        out.acceptable = true;
    }

    out.scaffold = out.scaffold.detected ? out.scaffold : after_strip;
    if (out.scaffold.score <= 0.0f && after_strip.detected) {
        out.scaffold = after_strip;
    }
    return out;
}

struct TranscriptMarkerCounts {
    int user_markers = 0;
    int agent_markers = 0;
};

TranscriptMarkerCounts countTranscriptMarkers(std::string_view raw) {
    TranscriptMarkerCounts counts;
    if (raw.empty()) {
        return counts;
    }
    const std::vector<LineSpan> lines = splitLines(raw);
    for (const LineSpan& line : lines) {
        if (line.blank || !line.transcript_scaffold) {
            continue;
        }
        const std::string_view body = trimHorizontal(lineContent(raw, line.begin, line.end));
        if (body.starts_with("[User]")) {
            ++counts.user_markers;
        } else if (body.starts_with("[Agent]")) {
            ++counts.agent_markers;
        }
    }
    return counts;
}

void attachAttemptPipelineDiagnostics(std::vector<GenerationAttemptTelemetry>& attempts,
                                      const ProcessedAttempt& pipeline,
                                      std::string_view raw,
                                      const ResponseValidityAssessment& validity,
                                      int max_tokens_requested) {
    if (attempts.empty()) {
        return;
    }
    GenerationAttemptTelemetry& row = attempts.back();
    row.sanitized_answer_chars = pipeline.sanitized_text.size();
    row.empty_after_sanitize = pipeline.empty;
    row.response_valid = validity.valid;
    row.invalid_reason = validity.invalid_reason;
    const TranscriptMarkerCounts counts = countTranscriptMarkers(raw);
    row.transcript_user_marker_count = counts.user_markers;
    row.transcript_agent_marker_count = counts.agent_markers;
    row.max_tokens_requested = max_tokens_requested;
    const RawCompletionSample sample = buildRawCompletionSample(raw);
    row.raw_sample_first = sample.first;
    row.raw_sample_last = sample.last;
    if (chatFullRawLoggingEnabled()) {
        row.raw_completion.assign(raw.begin(), raw.end());
        row.sanitized_completion = pipeline.sanitized_text;
    }
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
        if (!line.blank && !line.transcript_scaffold) {
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
        if (lines[start_idx].transcript_scaffold) {
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
        if (lines[i].transcript_scaffold) {
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

RegurgitationAssessment assessChunkFormatRegurgitation(std::string_view text) {
    if (text.empty() || isAllWhitespace(text)) {
        return {};
    }
    return assessLines(splitLines(text), text);
}

AnswerQualityAssessment assessAnswerQuality(std::string_view text) {
    AnswerQualityAssessment out;
    if (text.empty() || isAllWhitespace(text)) {
        return out;
    }

    const std::vector<LineSpan> lines = splitLines(text);
    int atx_headers = 0;
    int numbered_lines = 0;

    for (const LineSpan& line : lines) {
        if (line.blank) {
            continue;
        }
        const std::string_view trimmed = trimHorizontal(lineContent(text, line.begin, line.end));
        if (trimmed.starts_with("#")) {
            ++atx_headers;
        }
        if (trimmed.size() >= 3 && std::isdigit(static_cast<unsigned char>(trimmed[0]))) {
            std::size_t i = 0;
            while (i < trimmed.size() && std::isdigit(static_cast<unsigned char>(trimmed[i]))) {
                ++i;
            }
            if (i < trimmed.size() && trimmed[i] == '.' && i + 1 < trimmed.size()
                && trimmed[i + 1] == ' ') {
                ++numbered_lines;
            }
        }
    }

    const bool fragment_collage = hasFragmentCollage(lines, text);
    const bool framing = hasConversationalFraming(text);

    if (atx_headers >= 2) {
        out.pasted_retrieval_context = true;
    }
    if (numbered_lines >= 3) {
        out.pasted_retrieval_context = true;
    }
    if (fragment_collage && !framing) {
        out.pasted_retrieval_context = true;
    }

    if (out.pasted_retrieval_context) {
        return out;
    }

    if (framing) {
        out.complete_assistant_answer = true;
        return out;
    }
    if (countSentenceBreaks(text) >= 2 && text.size() >= 40) {
        out.complete_assistant_answer = true;
        return out;
    }

    return out;
}

SanitizeOutcome sanitizeChunkFormatScaffold(std::string_view raw) {
    SanitizeOutcome out;

    if (raw.empty() || isAllWhitespace(raw)) {
        out.sanitize_reason = kSanitizeNone;
        out.empty_after_sanitize = true;
        return out;
    }

    const std::vector<LineSpan> lines = splitLines(raw);
    bool removed_scaffold = false;
    bool kept_any = false;
    std::ostringstream assembled;
    bool pending_blank = false;

    for (const LineSpan& line : lines) {
        if (line.blank) {
            if (kept_any) {
                pending_blank = true;
            }
            continue;
        }
        if (line.chunk_scaffold) {
            removed_scaffold = true;
            continue;
        }
        if (pending_blank) {
            assembled << '\n';
            pending_blank = false;
        }
        assembled.write(raw.data() + static_cast<std::streamsize>(line.begin),
                        static_cast<std::streamsize>(line.end - line.begin));
        assembled << '\n';
        kept_any = true;
    }

    if (!kept_any) {
        out.sanitize_reason = kSanitizeAllChunkScaffold;
        out.empty_after_sanitize = true;
        return out;
    }

    out.sanitized_text = assembled.str();
    while (!out.sanitized_text.empty() && out.sanitized_text.back() == '\n') {
        out.sanitized_text.pop_back();
    }

    if (isAllWhitespace(out.sanitized_text)) {
        out.sanitized_text.clear();
        out.sanitize_reason = kSanitizeAllChunkScaffold;
        out.empty_after_sanitize = true;
        return out;
    }

    out.empty_after_sanitize = false;
    out.sanitize_reason = removed_scaffold ? kSanitizeStrippedChunkScaffold : kSanitizeNone;
    return out;
}

ResponseValidityAssessment assessResponseValidity(std::string_view sanitized_text,
                                                  std::string_view user_query,
                                                  std::string_view sanitize_reason) {
    ResponseValidityAssessment out;

    auto collapseWs = [](std::string_view text) {
        std::string normalized;
        normalized.reserve(text.size());
        bool in_space = false;
        for (unsigned char ch : text) {
            if (std::isspace(ch) != 0) {
                if (!in_space) {
                    normalized.push_back(' ');
                    in_space = true;
                }
            } else {
                normalized.push_back(static_cast<char>(ch));
                in_space = false;
            }
        }
        while (!normalized.empty() && normalized.front() == ' ') {
            normalized.erase(normalized.begin());
        }
        while (!normalized.empty() && normalized.back() == ' ') {
            normalized.pop_back();
        }
        return normalized;
    };

    bool sanitized_blank = sanitized_text.empty();
    if (!sanitized_blank) {
        sanitized_blank = true;
        for (unsigned char ch : sanitized_text) {
            if (!std::isspace(ch)) {
                sanitized_blank = false;
                break;
            }
        }
    }
    if (sanitized_blank) {
        out.valid = false;
        out.invalid_reason = kInvalidReasonEmpty;
        return out;
    }

    const std::string normalized_sanitized = collapseWs(sanitized_text);
    const std::string normalized_query = collapseWs(user_query);
    if (!normalized_query.empty() && normalized_sanitized == normalized_query) {
        out.valid = false;
        out.invalid_reason = kInvalidReasonQueryEcho;
        return out;
    }

    if (sanitize_reason == kSanitizeTruncatedTranscriptMarker && sanitized_text.size() <= 80) {
        const AnswerQualityAssessment quality = assessAnswerQuality(sanitized_text);
        if (!quality.complete_assistant_answer) {
            out.valid = false;
            out.invalid_reason = kInvalidReasonTruncateShortPrefix;
            return out;
        }
    }

    out.valid = true;
    out.invalid_reason = kInvalidReasonNone;
    return out;
}

bool chatPromptLoggingEnabled() {
    const char* env = std::getenv("THOTH_LOG_CHAT_PROMPT");
    return env && *env
           && (std::string_view(env) == "1" || std::string_view(env) == "true"
               || std::string_view(env) == "TRUE");
}

bool chatFullRawLoggingEnabled() {
    const char* env = std::getenv("THOTH_LOG_FULL_RAW_CHAT_COMPLETION");
    return env && *env
           && (std::string_view(env) == "1" || std::string_view(env) == "true"
               || std::string_view(env) == "TRUE");
}

RawCompletionSample buildRawCompletionSample(std::string_view raw_text) {
    RawCompletionSample sample;
    constexpr std::size_t kSampleChars = 80;
    const char* env = std::getenv("THOTH_LOG_RAW_CHAT_COMPLETION");
    const bool enabled =
        env && *env
        && (std::string_view(env) == "1" || std::string_view(env) == "true"
            || std::string_view(env) == "TRUE");
    if (!enabled || raw_text.empty()) {
        return sample;
    }
    if (raw_text.size() <= kSampleChars) {
        sample.first.assign(raw_text.begin(), raw_text.end());
        return sample;
    }
    sample.first.assign(raw_text.begin(), raw_text.begin() + static_cast<std::ptrdiff_t>(kSampleChars));
    sample.last.assign(raw_text.end() - static_cast<std::ptrdiff_t>(kSampleChars), raw_text.end());
    return sample;
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

void finalizeSuccessfulAttempt(ChatGenerationResult& out, const ProcessedAttempt& pipeline) {
    out.sanitized_text = pipeline.sanitized_text;
    out.sanitize_reason = pipeline.sanitize_reason;
    out.empty_after_sanitize = false;
    out.regurgitation_score = pipeline.scaffold.score;
    out.regurgitation_detected = false;
    out.regurgitation_retry_reason = kRegurgitationRetryReasonNone;
}

void finalizeAttemptTelemetry(ChatGenerationResult& out) {
    out.generation_attempt_count = static_cast<int>(out.generation_attempts.size());
    out.raw_sample = buildRawCompletionSample(out.raw_text);
}

} // namespace

ChatGenerationResult generateAndSanitizeChat(LLMInterface& llm,
                                             const std::string& prompt,
                                             const ChatGenerateOptions& opts) {
    ChatGenerationResult out;

    LlmAttemptContext attemptCtx{
        llm, opts.max_tokens, opts.stop_sequences, out.generation_attempts, out.generation_latency_ms};

    struct AttemptBundle {
        InferenceGenerateResult gen;
        ProcessedAttempt pipeline;
        bool used_stops = false;
    };

    auto assessValid = [&](const ProcessedAttempt& pipeline) {
        return assessResponseValidity(pipeline.sanitized_text, opts.user_query, pipeline.sanitize_reason);
    };

    auto applyAttemptToOut = [&](const AttemptBundle& attempt) {
        applyProviderAttempt(out, attempt.gen, attempt.used_stops);
        out.regurgitation_score = attempt.pipeline.scaffold.score;
        out.regurgitation_detected = attempt.pipeline.scaffold.detected;
    };

    auto finalizeSuccess = [&](const AttemptBundle& attempt) {
        applyAttemptToOut(attempt);
        finalizeSuccessfulAttempt(out, attempt.pipeline);
        const auto validity = assessValid(attempt.pipeline);
        out.response_valid = validity.valid;
        out.invalid_reason = validity.invalid_reason;
        out.fallback_used = false;
        finalizeAttemptTelemetry(out);
    };

    auto finalizeFallback = [&](const AttemptBundle& lastAttempt) {
        applyAttemptToOut(lastAttempt);
        out.fallback_used = true;
        out.sanitized_text =
            opts.use_greeting_fallback ? kFallbackGreeting : kFallbackGeneric;
        out.empty_after_sanitize = false;
        out.response_valid = false;
        out.invalid_reason = kInvalidReasonNoUsableGeneration;
        if (lastAttempt.pipeline.scaffold.detected) {
            out.regurgitation_detected = true;
            out.regurgitation_score = lastAttempt.pipeline.scaffold.score;
        }
        finalizeAttemptTelemetry(out);
    };

    // --- Attempt 1 ---
    const bool firstUsesStops = !opts.stop_sequences.empty();
    AttemptBundle first;
    first.used_stops = firstUsesStops;
    first.gen = runTrackedGeneration(attemptCtx, prompt, opts);
    if (!first.gen.ok) {
        applyProviderAttempt(out, first.gen, firstUsesStops);
        out.sanitized_text.clear();
        out.empty_after_sanitize = true;
        out.response_valid = false;
        out.invalid_reason = kInvalidReasonProviderError;
        finalizeAttemptTelemetry(out);
        return out;
    }

    applyProviderAttempt(out, first.gen, firstUsesStops);
    first.pipeline = processGenerationPipeline(out.raw_text);
    attachAttemptSanitizeReason(out.generation_attempts, first.pipeline.sanitize_reason);
    attachAttemptPipelineDiagnostics(out.generation_attempts,
                                     first.pipeline,
                                     first.gen.text,
                                     assessValid(first.pipeline),
                                     attemptCtx.max_tokens);

    if (!first.pipeline.empty && assessValid(first.pipeline).valid) {
        finalizeSuccess(first);
        return out;
    }

    if (static_cast<int>(out.generation_attempts.size()) >= kMaxChatGenerationAttempts) {
        finalizeFallback(first);
        return out;
    }

    // --- Attempt 2 (stop-free; Phase 1 hard cap — no third regurgitation generation) ---
    out.retried_without_stops = true;
    if (first.pipeline.needs_regurgitation_retry || first.pipeline.scaffold.detected) {
        out.retry_due_to_regurgitation = true;
        out.regurgitation_retry_reason = first.pipeline.regurgitation_retry_reason;
        if (out.regurgitation_retry_reason == kRegurgitationRetryReasonNone
            && first.pipeline.scaffold.detected) {
            out.regurgitation_retry_reason = kRegurgitationRetryReasonScaffoldRemaining;
        }
    }

    attemptCtx.stop_sequences = {};
    AttemptBundle second;
    second.used_stops = false;
    second.gen = runTrackedGeneration(attemptCtx, prompt, opts);
    if (!second.gen.ok) {
        if (!first.pipeline.empty && assessValid(first.pipeline).valid) {
            finalizeSuccess(first);
            return out;
        }
        finalizeFallback(first);
        return out;
    }

    applyProviderAttempt(out, second.gen, false);
    second.pipeline = processGenerationPipeline(out.raw_text);
    attachAttemptSanitizeReason(out.generation_attempts, second.pipeline.sanitize_reason);
    attachAttemptPipelineDiagnostics(out.generation_attempts,
                                     second.pipeline,
                                     second.gen.text,
                                     assessValid(second.pipeline),
                                     attemptCtx.max_tokens);

    if (!second.pipeline.empty && assessValid(second.pipeline).valid) {
        finalizeSuccess(second);
        return out;
    }

    if (!first.pipeline.empty && assessValid(first.pipeline).valid) {
        finalizeSuccess(first);
        return out;
    }

    finalizeFallback(second);
    return out;
}

} // namespace ChatGeneration
} // namespace Thoth
