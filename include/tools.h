/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * 
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once

#include "itool.h"

class Config;
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Thoth { class FactStore; }

/**
 * @brief Global registry for all tools in the Thoth agent system.
 * Implemented according to TOOLS.md v1.0
 */
class ToolRegistry {
public:
    ToolRegistry();

    /**
     * @brief Registers a tool in the global registry.
     * Throws an error if a tool with the same name already exists.
     */
    void registerTool(std::unique_ptr<ITool> tool);

    /**
     * @brief Initializes tools that require external dependencies (like FactStore and LLM).
     */
    void initialize(std::shared_ptr<Thoth::FactStore> factStore, class LLMInterface* llm);

    /**
     * @brief Binds runtime config to tools that enforce security flags (e.g. allow_shell_exec).
     */
    void setConfig(Config* cfg);

    /**
     * @brief Executes a tool by name with the given JSON input.
     */
    nlohmann::json executeTool(const std::string& name, const nlohmann::json& input) const;

    /**
     * @brief Returns a list of all registered tools.
     */
    std::vector<const ITool*> getAvailableTools() const;

    /**
     * @brief Singleton instance of the ToolRegistry.
     */
    static ToolRegistry& instance();

private:
    std::unordered_map<std::string, std::unique_ptr<ITool>> tools_;
    mutable std::mutex mutex_;
};
