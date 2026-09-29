/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — call-scoped generation metadata for MTCP telemetry
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Thoth {

/** Copied by value into a generation call. Not a process-wide last result. */
struct GenerationCallContext {
    std::string task_id;
    std::string plan_id;
    std::string session_id;
    std::string call_type;
    int attempt = 0;
    bool reflection = false;
};

/**
 * Thread-local stack so existing planner entry points can carry a task id
 * onto the thread that performs the synchronous provider call. Each thread
 * has its own stack. Push and pop are paired by GenerationCallScope.
 */
class GenerationCallScope {
public:
    explicit GenerationCallScope(GenerationCallContext context);
    ~GenerationCallScope();

    GenerationCallScope(const GenerationCallScope&) = delete;
    GenerationCallScope& operator=(const GenerationCallScope&) = delete;

    static GenerationCallContext current();

private:
    static std::vector<GenerationCallContext>& stack();
};

struct GenerationOutcome {
    std::string generation_id;
    std::string text;
    bool ok = false;
    std::string error;
    int requested_max_tokens = 0;
    std::int64_t prompt_tokens = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t total_tokens = 0;
    bool has_total_tokens = false;
    /** False means provider usage is unavailable. Token integers are then not measurements. */
    bool provider_usage_reported = false;
    std::string finish_reason;
    std::int64_t elapsed_ms = 0;
    GenerationCallContext context;
    std::string wrapper_sha256;
};

class RawProviderCapture {
public:
    static void store(const std::string& capture_id, const std::string& raw_text);
    static std::optional<std::string> take(const std::string& capture_id);
};

struct GenerationRecordFields {
    bool parse_ok = false;
    bool has_parse_ok = false;
    bool validation_ok = false;
    bool has_validation_ok = false;
    bool fallback_used = false;
    bool kept_existing_plan = false;
    /** Provider generation this decision refers to. Same id as outcome.generation_id. */
    std::string associated_generation_id;
    bool context_overflow = false;
    /** Completed provider text was empty. Not inferred from token counts. */
    bool synthesis_observed_empty = false;
};

inline bool reportedContextOverflow(const GenerationOutcome& outcome) {
    return outcome.provider_usage_reported
        && outcome.prompt_tokens + static_cast<std::int64_t>(outcome.requested_max_tokens) > 8192;
}

} // namespace Thoth
