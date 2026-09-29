/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — SummaryGenerator implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/summary_generator.h"
#include "../include/generation_budget.h"
#include "../include/generation_call.h"
#include "../include/generation_call_log.h"
#include "../include/llm_interface.h"
#include <cstdlib>
#include <sstream>

namespace Thoth {

namespace {

std::string formatBatchForPrompt(const std::vector<MessageRecord>& batch) {
    std::ostringstream out;
    for (const auto& msg : batch) {
        out << "[" << msg.role << "] " << msg.content << "\n";
    }
    return out.str();
}

SummaryGenerationResult mockExtract(const std::vector<MessageRecord>& batch) {
    SummaryGenerationResult result;
    result.llm_success = true;
    result.parse_success = true;
    result.prompt_version = "episodic_mock_v1";
    result.llm_model = "mock";

    for (const auto& msg : batch) {
        if (msg.role == "user") {
            result.memory.goals.push_back(msg.content);
            if (msg.content.find("dog") != std::string::npos ||
                msg.content.find("name is") != std::string::npos) {
                result.memory.user_preferences.push_back(msg.content);
            }
        } else if (msg.role == "assistant") {
            result.memory.facts_learned.push_back(msg.content);
        }
    }
    if (!result.memory.goals.empty()) {
        result.memory.decisions.push_back("Discussed: " + result.memory.goals.front());
    }
    return result;
}

float computeConfidence(const SummaryGenerationResult& result) {
    if (!result.llm_success || !result.parse_success) {
        return 0.5f;
    }
    if (result.truncated || result.fields_inferred) {
        return 0.75f;
    }
    return 1.0f;
}

} // namespace

SummaryGenerator::SummaryGenerator(LLMInterface* llm) : llm_(llm) {}

SummaryGenerationResult SummaryGenerator::extract(const std::vector<MessageRecord>& batch) const {
    if (batch.empty()) {
        return {};
    }

    const char* mockEnv = std::getenv("THOTH_MOCK_EPISODIC");
    if (mockEnv && (std::string(mockEnv) == "1" || std::string(mockEnv) == "true")) {
        auto result = mockExtract(batch);
        result.memory.importance = scoreEpisodicImportance(result.memory);
        result.memory.novelty = 0.5f;
        result.memory.confidence = computeConfidence(result);
        return result;
    }

    if (!llm_) {
        SummaryGenerationResult failed;
        failed.llm_success = false;
        failed.parse_success = false;
        failed.memory.confidence = 0.5f;
        failed.memory.novelty = 0.5f;
        failed.memory.importance = 0.2f;
        return failed;
    }

    const std::string transcript = formatBatchForPrompt(batch);
    const std::string prompt =
        "Extract episodic memory from this conversation batch. Respond with JSON only, no markdown.\n"
        "Schema:\n"
        "{\n"
        "  \"goals\": [\"string\"],\n"
        "  \"plans_attempted\": [\"string\"],\n"
        "  \"decisions\": [\"string\"],\n"
        "  \"failures\": [\"string\"],\n"
        "  \"tool_results\": [\"string\"],\n"
        "  \"open_tasks\": [\"string\"],\n"
        "  \"facts_learned\": [\"string\"],\n"
        "  \"user_preferences\": [\"string\"],\n"
        "  \"outstanding_questions\": [\"string\"]\n"
        "}\n\n"
        "Transcript:\n" +
        transcript;

    SummaryGenerationResult result;
    result.prompt_version = "episodic_v1";
    result.llm_model = llm_->getSelectedModel();

    const int summaryCeiling = Thoth::GenerationBudget::resolvedOr(512);
    Thoth::GenerationCallContext call = Thoth::GenerationCallScope::current();
    call.call_type = "memory_summary";
    const Thoth::GenerationOutcome generated = llm_->generateCall(prompt, summaryCeiling, {}, call);
    const std::string response = generated.ok ? generated.text : std::string();
    Thoth::GenerationRecordFields fields;
    fields.associated_generation_id = generated.generation_id;
    fields.has_parse_ok = true;
    fields.context_overflow = generated.prompt_tokens + generated.requested_max_tokens > 8192;
    result.llm_success = generated.ok && !response.empty();

    try {

        std::string jsonText = response;
        const auto start = jsonText.find('{');
        const auto end = jsonText.rfind('}');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            jsonText = jsonText.substr(start, end - start + 1);
        }

        const auto parsed = nlohmann::json::parse(jsonText);
        result.memory = EpisodicMemory::fromJson(parsed);
        result.parse_success = true;
        fields.parse_ok = true;
        Thoth::GenerationCallLog::append(generated, fields);
    } catch (...) {
        result.parse_success = false;
        fields.parse_ok = false;
        Thoth::GenerationCallLog::append(generated, fields);
    }

    result.memory.importance = scoreEpisodicImportance(result.memory);
    result.memory.novelty = 0.5f;
    result.memory.confidence = computeConfidence(result);
    return result;
}

} // namespace Thoth
