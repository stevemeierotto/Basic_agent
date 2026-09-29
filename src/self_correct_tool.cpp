/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — self_correct tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/self_correct_tool.h"
#include "../include/generation_call.h"
#include "../include/generation_call_log.h"
#include <iostream>
#include <sstream>

SelfCorrectTool::SelfCorrectTool(LLMInterface& llm) : llm_(llm) {}

nlohmann::json SelfCorrectTool::input_schema() const {
    return {
        {"type", "object"},
        {"properties", {
            {"result", {{"type", "string"}, {"description", "The raw result or output to verify."}}},
            {"expectations", {{"type", "string"}, {"description", "A description of what the result should contain or look like."}}}
        }},
        {"required", {"result", "expectations"}},
        {"additionalProperties", false}
    };
}

bool SelfCorrectTool::requires_confirmation() const {
    return false;
}

nlohmann::json SelfCorrectTool::execute(const nlohmann::json& input) const {
    std::string result_str = input.at("result");
    std::string expectations_str = input.at("expectations");

    std::ostringstream prompt;
    prompt << "You are the Thoth Verification Engine. Your task is to determine if a given RESULT meets the provided EXPECTATIONS.\n\n";
    prompt << "[RESULT]\n" << result_str << "\n\n";
    prompt << "[EXPECTATIONS]\n" << expectations_str << "\n\n";
    prompt << "Instructions:\n";
    prompt << "1. Analyze the RESULT against the EXPECTATIONS.\n";
    prompt << "2. Determine if the expectations are met (valid: true) or not (valid: false).\n";
    prompt << "3. Provide a concise reason for your decision.\n";
    prompt << "4. Respond with the following JSON schema ONLY. No preamble or markdown fences.\n\n";
    prompt << "{\"valid\": boolean, \"reason\": \"string\"}";

    Thoth::GenerationCallContext call = Thoth::GenerationCallScope::current();
    call.call_type = "self_correct";
    const Thoth::GenerationOutcome generated = llm_.generateCall(prompt.str(), -1, {}, call);
    std::string response = generated.ok ? generated.text : std::string("Assistant: [Error] ") + generated.error;
    Thoth::GenerationRecordFields fields;
    fields.associated_generation_id = generated.generation_id;
    fields.context_overflow = Thoth::reportedContextOverflow(generated);

    try {
        // Simple attempt to find JSON in response if it has preamble
        size_t start = response.find("{");
        size_t end = response.rfind("}");
        if (start != std::string::npos && end != std::string::npos && end > start) {
            response = response.substr(start, end - start + 1);
        }

        nlohmann::json parsed = nlohmann::json::parse(response);
        
        fields.has_parse_ok = true;
        fields.parse_ok = parsed.contains("valid") && parsed["valid"].is_boolean();
        Thoth::GenerationCallLog::append(generated, fields);
        if (!parsed.contains("valid") || !parsed["valid"].is_boolean()) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", "LLM response did not contain a valid 'valid' boolean."}
            };
        }

        return {
            {"status", "success"},
            {"data", parsed},
            {"error_message", nullptr}
        };
    } catch (const std::exception& e) {
        fields.has_parse_ok = true;
        fields.parse_ok = false;
        Thoth::GenerationCallLog::append(generated, fields);
        return {
            {"status", "error"},
            {"data", {{"raw_response", response}}},
            {"error_message", std::string("Failed to parse LLM verification response: ") + e.what()}
        };
    }
}
