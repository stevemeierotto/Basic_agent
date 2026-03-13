/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — project_analyze tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_PROJECT_ANALYZE_TOOL_H
#define THOTH_PROJECT_ANALYZE_TOOL_H

#include "itool.h"

/**
 * @class ProjectAnalyzeTool
 * @brief Scans the Thoth source directory and returns a structured project graph.
 *
 * Implemented according to TOOLS.md v1.0.
 */
class ProjectAnalyzeTool : public ITool {
public:
    std::string name() const override { return "project_analyze"; }
    
    std::string description() const override {
        return "Scans the source directory and returns a structured project graph including files, classes, and dependencies.";
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;


private:
    /**
     * @brief Extracts class names from a header file.
     */
    std::vector<std::string> extractClasses(const std::string& content) const;

    /**
     * @brief Extracts included headers from a source file.
     */
    std::vector<std::string> extractIncludes(const std::string& content) const;
};

#endif // THOTH_PROJECT_ANALYZE_TOOL_H
