/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — PlanParser for dynamic, LLM-generated plans
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_PLAN_PARSER_H
#define THOTH_PLAN_PARSER_H

#include <string>
#include <optional>

#include "plan.h"

namespace Thoth {

/**
 * @class PlanParser
 * @brief A utility class to parse raw LLM string output into a validated Plan object.
 *
 * This class is designed to handle common LLM output issues such as markdown fences,
 * preamble text, and schema deviations. It provides a single static method to perform
 * the parsing and validation.
 */
class PlanParser {
public:
    /**
     * @brief Parses a raw string from an LLM into a structured Plan.
     *
     * This function will first attempt to extract a valid JSON object from the input string,
     * stripping common markdown fences (```json ... ```) and other extraneous text.
     * It then validates the JSON against the expected schema for a Plan.
     *
     * @param raw_llm_output The raw string response from the language model.
     * @param plan_id The parent plan ID to use when generating deterministic step IDs.
     * @return A std::optional<Plan> containing the valid Plan if parsing and validation
     *         succeed, otherwise std::nullopt.
     */
    static std::optional<Plan> parse(const std::string& raw_llm_output,
                                     const std::string& plan_id,
                                     std::string* failure_reason = nullptr);

private:
    /**
     * @brief Extracts a JSON string from a larger text block, removing markdown fences.
     * @param raw_output The string which may contain a JSON block.
     * @return The extracted JSON string, or the original string if no fences are found.
     */
    static std::string extractJsonString(const std::string& raw_output);
};

} // namespace Thoth

#endif // THOTH_PLAN_PARSER_H
