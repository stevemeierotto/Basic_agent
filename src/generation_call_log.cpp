/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — append-only non-chat generation telemetry
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/generation_call_log.h"

#include "../include/json.hpp"
#include "file_handler.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>

namespace Thoth {

namespace {

bool containsForbiddenText(const nlohmann::json& line) {
    return line.contains("prompt") || line.contains("completion") || line.contains("text")
           || line.contains("raw_completion") || line.contains("chain_of_thought");
}

} // namespace

std::string GenerationCallLog::logFilePath() {
    if (const char* overridePath = std::getenv("THOTH_GENERATION_CALLS_LOG")) {
        if (*overridePath) {
            return overridePath;
        }
    }
    FileHandler fh;
    return fh.getLogsPath("generation_calls.jsonl");
}

bool GenerationCallLog::append(const GenerationOutcome& outcome, const GenerationRecordFields& fields) {
    if (outcome.context.call_type.empty() || outcome.context.call_type == "chat") {
        return false;
    }

    nlohmann::json line = {
        {"event", "GENERATION_CALL"},
        {"timestamp_ms",
         std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
             .count()},
        {"generation_id", outcome.generation_id},
        {"associated_generation_id",
         fields.associated_generation_id.empty() ? outcome.generation_id
                                                 : fields.associated_generation_id},
        {"call_type", outcome.context.call_type},
        {"attempt", outcome.context.attempt},
        {"reflection", outcome.context.reflection},
        {"requested_max_tokens", outcome.requested_max_tokens},
        {"prompt_tokens", outcome.prompt_tokens},
        {"completion_tokens", outcome.completion_tokens},
        {"finish_reason", outcome.finish_reason},
        {"elapsed_ms", outcome.elapsed_ms},
        {"provider_ok", outcome.ok},
        {"fallback_used", fields.fallback_used},
        {"kept_existing_plan", fields.kept_existing_plan},
        {"context_overflow", fields.context_overflow},
    };
    if (outcome.has_total_tokens) {
        line["total_tokens"] = outcome.total_tokens;
    }
    if (!outcome.context.task_id.empty()) {
        line["task_id"] = outcome.context.task_id;
    }
    if (!outcome.context.plan_id.empty()) {
        line["plan_id"] = outcome.context.plan_id;
    }
    if (!outcome.context.session_id.empty()) {
        line["session_id"] = outcome.context.session_id;
    }
    if (fields.has_parse_ok) {
        line["parse_ok"] = fields.parse_ok;
    }
    if (fields.has_validation_ok) {
        line["validation_ok"] = fields.validation_ok;
    }
    if (!outcome.wrapper_sha256.empty()) {
        line["wrapper_sha256"] = outcome.wrapper_sha256;
    }
    if (!outcome.error.empty()) {
        line["provider_error"] = outcome.error;
    }
    if (containsForbiddenText(line)) {
        return false;
    }

    static std::mutex writeMutex;
    std::lock_guard<std::mutex> lock(writeMutex);
    std::ofstream out(logFilePath(), std::ios::app);
    if (!out.is_open()) {
        return false;
    }
    out << line.dump() << '\n';
    return true;
}

} // namespace Thoth
