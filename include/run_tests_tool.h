/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — run_tests tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_RUN_TESTS_TOOL_H
#define THOTH_RUN_TESTS_TOOL_H

#include "itool.h"

class Config;

/**
 * @class RunTestsTool
 * @brief Executes the Thoth unit test suite and returns structured results.
 *
 * Implemented according to TOOLS.md v1.0.
 */
class RunTestsTool : public ITool {
public:
    explicit RunTestsTool(Config* config = nullptr);

    std::string name() const override { return "run_tests"; }
    
    std::string description() const override {
        return "Executes the Thoth unit test suite and returns a summary of passed and failed tests.";
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;

private:
    Config* config_ = nullptr;
};

#endif // THOTH_RUN_TESTS_TOOL_H
