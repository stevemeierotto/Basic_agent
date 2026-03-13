/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — code_modify tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CODE_MODIFY_TOOL_H
#define THOTH_CODE_MODIFY_TOOL_H

#include "itool.h"

/**
 * @class CodeModifyTool
 * @brief Enables the agent to read and modify its own codebase via diffs.
 *
 * Implemented according to TOOLS.md v1.0.
 */
class CodeModifyTool : public ITool {
public:
    std::string name() const override { return "code_modify"; }
    
    std::string description() const override {
        return "Reads file contents or applies unified diffs to files within the project root.";
    }

    nlohmann::json input_schema() const override;

    bool requires_confirmation() const override;

    nlohmann::json execute(const nlohmann::json& input) const override;

private:
    /**
     * @brief Validates that a path is safe and within the project root.
     */
    bool isPathSafe(const std::string& relative_path, const std::string& project_root) const;

    /**
     * @brief Applies a unified diff to a string of text.
     * 
     * Note: This is a simplified diff applicator for v1.0.
     */
    std::string applyDiff(const std::string& original, const std::string& diff, std::string& error) const;
};

#endif // THOTH_CODE_MODIFY_TOOL_H
