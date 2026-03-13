#pragma once

#include <string>
#include <json.hpp>

/**
 * @brief Interface for all tools in the Thoth agent system.
 * Implemented according to TOOLS.md v1.0
 */
class ITool {
public:
    virtual ~ITool() = default;

    /**
     * @brief Returns the unique name of the tool (snake_case).
     */
    virtual std::string name() const = 0;

    /**
     * @brief Returns a concise description of the tool's purpose.
     */
    virtual std::string description() const = 0;

    /**
     * @brief Returns the JSON schema for the tool's input.
     */
    virtual nlohmann::json input_schema() const = 0;

    /**
     * @brief Whether this tool requires user confirmation before execution.
     */
    virtual bool requires_confirmation() const = 0;

    /**
     * @brief Executes the tool with the provided input.
     * 
     * @param input Validated JSON input.
     * @return nlohmann::json Structured output: { "status": "...", "data": {...}, "error_message": "..." }
     */
    virtual nlohmann::json execute(const nlohmann::json& input) const = 0;
};
