#include "tools.h"
#include "gmail_read_labels_tool.h"
#include "summarize_text_tool.h"
#include "project_analyze_tool.h"
#include "run_tests_tool.h"
#include "code_modify_tool.h"
#include "web_scrape_tool.h"
#include "self_correct_tool.h"
#include "gmail_read_messages_tool.h"
#include "store_fact_tool.h"
#include "fact_store.h"
#include "llm_interface.h"
#include <iostream>

ToolRegistry::ToolRegistry() {
    registerTool(std::make_unique<SummarizeTextTool>());
    registerTool(std::make_unique<GmailReadLabelsTool>());
    registerTool(std::make_unique<ProjectAnalyzeTool>());
    registerTool(std::make_unique<RunTestsTool>());
    registerTool(std::make_unique<CodeModifyTool>());
    registerTool(std::make_unique<WebScrapeTool>());
    registerTool(std::make_unique<GmailReadMessagesTool>());
}

void ToolRegistry::initialize(std::shared_ptr<Thoth::FactStore> factStore, class LLMInterface* llm) {
    if (factStore) {
        registerTool(std::make_unique<StoreFactTool>(*factStore));
    }
    if (llm) {
        registerTool(std::make_unique<SelfCorrectTool>(*llm));
    }
}

void ToolRegistry::registerTool(std::unique_ptr<ITool> tool) {
    if (!tool) return;
    
    std::string toolName = tool->name();
    std::lock_guard<std::mutex> lock(mutex_);
    
    // In v1.0, we allow overwriting to support test re-initialization
    tools_[toolName] = std::move(tool);
}

nlohmann::json ToolRegistry::executeTool(const std::string& name, const nlohmann::json& input) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        return {
            {"status", "error"},
            {"error_message", "Tool not found: " + name}
        };
    }

    const ITool* tool = it->second.get();
    if (tool->requires_confirmation()) {
        if (!input.contains("confirmed") || !input["confirmed"].get<bool>()) {
            return {
                {"status", "error"},
                {"data", {{"requires_confirmation", true}}},
                {"error_message", "Tool requires confirmation. Please re-run with 'confirmed': true."}
            };
        }
    }
    
    return tool->execute(input);
}

std::vector<const ITool*> ToolRegistry::getAvailableTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ITool*> available;
    available.reserve(tools_.size());

    for (const auto& [_, tool] : tools_) {
        available.push_back(tool.get());
    }

    return available;
}

ToolRegistry& ToolRegistry::instance() {
    static ToolRegistry registry;
    return registry;
}
