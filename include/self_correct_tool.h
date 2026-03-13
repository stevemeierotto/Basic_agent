/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — self_correct tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_SELF_CORRECT_TOOL_H
#define THOTH_SELF_CORRECT_TOOL_H

#include "itool.h"
#include "llm_interface.h"

/**
 * @class SelfCorrectTool
 * @brief Uses an LLM to verify if a result meets specific expectations.
 *
 * Implemented according to TOOLS.md v1.0.
 */
class SelfCorrectTool : public ITool {
public:
    explicit SelfCorrectTool(LLMInterface& llm);

    std::string name() const override { return "self_correct"; }
    
    std::string description() const override {
        return "Verifies if a specific result meets the provided expectations using LLM reasoning.";
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;


private:
    LLMInterface& llm_;
};

#endif // THOTH_SELF_CORRECT_TOOL_H
