/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — PlanParser for dynamic, LLM-generated plans
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/plan_parser.h"
#include "../include/decision_trace.h"
#include "../include/json.hpp"
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace Thoth {

using json = nlohmann::json;

namespace {

std::optional<std::string> balancedObject(const std::string& text, std::size_t open, std::size_t limit) {
    if (open >= limit || text[open] != '{') {
        return std::nullopt;
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = open; i < limit; ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                return text.substr(open, i - open + 1);
            }
        }
    }
    return std::nullopt;
}

/** Leftmost balanced object whose root has a plan array. Schema is checked by the caller. */
std::optional<std::string> leftmostPlanObject(const std::string& raw) {
    std::size_t begin = 0;
    std::size_t end = raw.size();
    const std::size_t fence = raw.find("```json");
    if (fence != std::string::npos) {
        begin = fence + std::string("```json").size();
        const std::size_t close = raw.find("```", begin);
        if (close != std::string::npos) {
            end = close;
        }
    }

    for (std::size_t i = begin; i < end; ++i) {
        if (raw[i] != '{') {
            continue;
        }
        const std::optional<std::string> slice = balancedObject(raw, i, end);
        if (!slice.has_value()) {
            continue;
        }
        try {
            const json parsed = json::parse(*slice);
            if (parsed.is_object() && parsed.contains("plan") && parsed["plan"].is_array()) {
                return slice;
            }
        } catch (const json::exception&) {
        }
    }
    return std::nullopt;
}

} // namespace

// Helper to safely get a value from json or a default if not present/wrong type
template<typename T>
T get_optional_value(const json& j, const std::string& key, T default_value) {
    if (j.contains(key) && !j.at(key).is_null()) {
        try {
            return j.at(key).get<T>();
        } catch (const json::type_error&) {
            return default_value;
        }
    }
    return default_value;
}


std::string PlanParser::extractJsonString(const std::string& raw_output) {
    // Find the start of the JSON markdown block
    size_t start_pos = raw_output.find("```json");
    if (start_pos == std::string::npos) {
        start_pos = raw_output.find("{"); // Fallback to first brace
        if (start_pos == std::string::npos) return raw_output; // No JSON found
    } else {
        start_pos = raw_output.find("{", start_pos); // Find the actual start of the object
        if (start_pos == std::string::npos) return raw_output;
    }

    // Find the end of the JSON block
    size_t end_pos = raw_output.rfind("}");
    if (end_pos == std::string::npos || end_pos < start_pos) {
        return raw_output; // No matching end brace found
    }

    return raw_output.substr(start_pos, end_pos - start_pos + 1);
}

std::optional<Plan> PlanParser::parse(const std::string& raw_llm_output,
                                         const std::string& plan_id,
                                         std::string* failure_reason) {
    DecisionTraceLogger logger;
    auto log_failure = [&](const std::string& reason, const std::string& bad_json) {
        if (failure_reason != nullptr) {
            *failure_reason = reason;
        }
        DecisionTrace trace = logger.startTrace("plan_parsing", raw_llm_output.length());
        logger.addStage(trace, "parse_failed", false, reason, {
            {"input", raw_llm_output},
            {"extracted_json", bad_json}
        });
        logger.finishTrace(trace, false, "Plan parsing failed");
        logger.writeTrace(trace);
    };
    if (failure_reason != nullptr) {
        failure_reason->clear();
    }

    const std::optional<std::string> plan_object = leftmostPlanObject(raw_llm_output);
    std::string json_str = plan_object.has_value() ? *plan_object : extractJsonString(raw_llm_output);
    json parsed_json;

    try {
        parsed_json = json::parse(json_str);
    } catch (const json::parse_error& e) {
        log_failure("Malformed JSON: " + std::string(e.what()), json_str);
        return std::nullopt;
    }

    if (!parsed_json.is_object() || !parsed_json.contains("plan") || !parsed_json["plan"].is_array()) {
        log_failure("Root 'plan' array not found or is not an array.", parsed_json.dump());
        return std::nullopt;
    }

    Plan plan;
    plan.plan_id = plan_id;
    int step_index = 0;

    for (const auto& step_json : parsed_json["plan"]) {
        if (!step_json.is_object()) {
            log_failure("Plan step is not a JSON object.", step_json.dump());
            return std::nullopt;
        }

        PlanStep step;

        // Required fields
        if (!step_json.contains("step_type") || !step_json["step_type"].is_string()) {
            log_failure("Step missing or invalid 'step_type'.", step_json.dump());
            return std::nullopt;
        }
        if (!step_json.contains("description") || !step_json["description"].is_string()
            || step_json["description"].get<std::string>().find_first_not_of(" \t\n\r")
                   == std::string::npos) {
            log_failure("Step missing or invalid 'description'.", step_json.dump());
            return std::nullopt;
        }

        // step_id (generate if missing)
        step.step_id = get_optional_value<std::string>(step_json, "step_id", plan_id + "-step-" + std::to_string(step_index));
        step.description = step_json["description"];

        // step_type (parse from string)
        std::string type_str = step_json["step_type"];
        if (type_str == "RETRIEVAL") step.type = StepType::RETRIEVAL;
        else if (type_str == "TOOL") step.type = StepType::TOOL;
        else if (type_str == "LLM") step.type = StepType::LLM;
        else if (type_str == "NODE") step.type = StepType::NODE;
        else {
            log_failure("Invalid 'step_type': " + type_str, step_json.dump());
            return std::nullopt;
        }

        // payload (optional)
        step.payload = get_optional_value<json>(step_json, "payload", json::object());

        // failure_policy (optional with defaults)
        if (step_json.contains("failure_policy") && step_json["failure_policy"].is_object()) {
            const auto& fp_json = step_json["failure_policy"];
            step.failure_policy.max_retries = get_optional_value<int>(fp_json, "max_retries", 1);
            step.failure_policy.abort_on_failure = get_optional_value<bool>(fp_json, "abort_on_failure", false);
            step.failure_policy.revise_plan_on_failure = get_optional_value<bool>(fp_json, "revise_plan_on_failure", false);
            step.failure_policy.timeout_ms = get_optional_value<int>(fp_json, "timeout_ms", 30000);
        }

        // depends_on (optional)
        step.depends_on = get_optional_value<std::vector<std::string>>(step_json, "depends_on", {});

        plan.steps.push_back(step);
        step_index++;
    }
    
    if (plan.steps.empty()) {
        log_failure("Plan contains no steps.", parsed_json.dump());
        return std::nullopt;
    }

    return plan;
}

} // namespace Thoth
