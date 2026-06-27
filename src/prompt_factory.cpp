#include "../include/prompt_factory.h"
#include "../include/tools.h"
#include "file_handler.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <filesystem>

namespace {

const char* kDefaultPlanGenerationTemplate =
    "Generate a JSON plan for the goal below using ONLY retrieval and synthesis.\n"
    "Goal: {{goal}}\n"
    "{{strategy_context}}\n"
    "{{past_experience}}\n"
    "Rules:\n"
    "- Respond with JSON only (no markdown fences).\n"
    "- Minimum 2 steps.\n"
    "- Step 1 MUST be step_type RETRIEVAL with payload {\"query\": \"<short query derived from goal>\", \"top_k\": 5}.\n"
    "- Step 2 MUST be step_type LLM to synthesize the answer from retrieved context.\n"
    "- Step 2 MUST set depends_on to the RETRIEVAL step_id.\n"
    "- Do NOT emit TOOL steps.\n"
    "Schema:\n"
    "{\"plan\":[{\"step_id\":\"retrieve-context\",\"step_type\":\"RETRIEVAL\","
    "\"description\":\"Retrieve relevant corpus context\",\"payload\":{\"query\":\"...\",\"top_k\":5}},"
    "{\"step_id\":\"synthesize\",\"step_type\":\"LLM\",\"description\":\"Summarize findings\","
    "\"depends_on\":[\"retrieve-context\"],\"payload\":{}}]}\n";

const char* kDefaultPlanRevisionTemplate =
    "Revise the plan for goal: {{goal}}\n"
    "Existing Plan: {{existing_plan}}\n"
    "Failed Step Result: {{failed_step_result}}\n"
    "Use the same schema as plan generation: RETRIEVAL first, then LLM. No TOOL steps.\n"
    "Respond with JSON plan ONLY.\n";

} // namespace

// Default constructor uses default PromptConfig()
PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r)
    : PromptFactory(mem, r, PromptConfig()) {}

// Constructor with explicit config
PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r, const PromptConfig& cfg)
    : memory(mem), rag(r), config(cfg) {}

std::string PromptFactory::buildConversationPrompt(const std::string& user_input,
                                                   bool useExtendedSummary) {
    std::string defaultTemplate = 
        "{{system_prompt}}\n\n"
        "{{tool_list}}\n\n"
        "[Memory Context]\n{{memory_context}}\n\n"
        "{{conversation_history}}\n"
        "[User] {{user_input}}\n[Agent] ";

    std::string templateStr = loadTemplate("conversation_prompt.txt", defaultTemplate);

    std::unordered_map<std::string, std::string> subs;
    subs["{{system_prompt}}"] = config.systemPrompt;
    subs["{{tool_list}}"] = getToolList();
    subs["{{memory_context}}"] = getMemoryContext(useExtendedSummary);
    subs["{{conversation_history}}"] = getConversationHistory();
    subs["{{user_input}}"] = user_input;

    std::string result = applySubstitutions(templateStr, subs);
    return truncateToLimit(result, config.maxContextLength);
}


std::string PromptFactory::buildRagQueryPrompt(const std::string& query) {
    std::string defaultTemplate = 
        "You are a context retriever. Find relevant context for the following query:\n"
        "{{query}}\n";

    std::string templateStr = loadTemplate("rag_query_prompt.txt", defaultTemplate);

    std::unordered_map<std::string, std::string> subs;
    subs["{{query}}"] = query;

    std::string result = applySubstitutions(templateStr, subs);
    return truncateToLimit(result, config.maxContextLength);
}

std::string PromptFactory::buildPlanPrompt(const std::string& goal,
                                            const std::string& strategy_context,
                                            const std::string& past_experience) {
    std::string templateStr = loadTemplate("plan_generation.tmpl", kDefaultPlanGenerationTemplate);

    std::unordered_map<std::string, std::string> subs;
    subs["{{goal}}"] = goal;
    subs["{{strategy_context}}"] = strategy_context;
    subs["{{past_experience}}"] = past_experience;
    subs["{{available_tools}}"] = getToolList();

    std::string result = applySubstitutions(templateStr, subs);
    return truncateToLimit(result, config.maxContextLength);
}

std::string PromptFactory::buildRevisionPrompt(const std::string& goal,
                                                const std::string& existing_plan_json,
                                                const std::string& failed_step_result_json) {
    std::string templateStr = loadTemplate("plan_revision.tmpl", kDefaultPlanRevisionTemplate);

    std::unordered_map<std::string, std::string> subs;
    subs["{{goal}}"] = goal;
    subs["{{existing_plan}}"] = existing_plan_json;
    subs["{{failed_step_result}}"] = failed_step_result_json;

    std::string result = applySubstitutions(templateStr, subs);
    return truncateToLimit(result, config.maxContextLength);
}

void PromptFactory::ensureDefaultTemplatesExist() {
    FileHandler fh;
    const std::filesystem::path dir(fh.getAgentWorkspacePath("prompt_templates"));
    std::filesystem::create_directories(dir);

    auto writeIfMissing = [&](const char* filename, const char* content) {
        const std::filesystem::path path = dir / filename;
        if (std::filesystem::exists(path)) {
            return;
        }
        std::ofstream out(path);
        if (out.is_open()) {
            out << content;
        }
    };

    writeIfMissing("plan_generation.tmpl", kDefaultPlanGenerationTemplate);
    writeIfMissing("plan_revision.tmpl", kDefaultPlanRevisionTemplate);
}

std::string PromptFactory::truncateToLimit(const std::string& input, size_t maxLen) const {
    if (input.size() <= maxLen) return input;
    return input.substr(input.size() - maxLen); // Keep last maxLen chars
}

std::string PromptFactory::loadTemplate(const std::string& filename, const std::string& defaultValue) {
    FileHandler fh;
    std::string path = fh.getAgentWorkspacePath("prompt_templates/" + filename);
    
    std::ifstream in(path);
    if (!in.is_open()) {
        return defaultValue;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string PromptFactory::applySubstitutions(std::string templateStr, const std::unordered_map<std::string, std::string>& subs) {
    for (const auto& [placeholder, value] : subs) {
        size_t pos = 0;
        while ((pos = templateStr.find(placeholder, pos)) != std::string::npos) {
            templateStr.replace(pos, placeholder.length(), value);
            pos += value.length();
        }
    }
    return templateStr;
}

std::string PromptFactory::getToolList() {
    if (!config.enableTools) return "";

    auto tools = ToolRegistry::instance().getAvailableTools();
    if (tools.empty()) return "";

    std::ostringstream oss;
    oss << "[Available Tools]\n";
    oss << "If you need to call a tool, respond with: {\"tool_call\": {\"name\": \"tool_name\", \"input\": {...}}}\n";
    oss << "IMPORTANT: Some tools REQUIRE confirmation. For these tools, you MUST include \"confirmed\": true in the tool call 'input' if you are sure you want to proceed. If you omit it, the tool will fail with a request for confirmation.\n\n";
    
    for (const auto* tool : tools) {
        oss << "Tool: " << tool->name() << "\n";
        oss << "Requires Confirmation: " << (tool->requires_confirmation() ? "YES" : "NO") << "\n";
        oss << "Description: " << tool->description() << "\n";
        oss << "Input Schema: " << tool->input_schema().dump() << "\n\n";
    }
    return oss.str();
}

std::string PromptFactory::getMemoryContext(bool useExtendedSummary) {
    std::string memSummary = memory.getSummary(useExtendedSummary);
    const size_t maxMemLen = config.maxContextLength / 2;
    if (memSummary.length() > maxMemLen) {
        memSummary = memSummary.substr(memSummary.length() - maxMemLen);
    }
    return memSummary;
}

std::string PromptFactory::getConversationHistory() {
    auto convo = memory.getConversation();
    size_t start = convo.size() > config.maxRecentMessages
                     ? convo.size() - config.maxRecentMessages
                     : 0;

    const size_t maxConvoLen = config.maxContextLength / 2;
    std::ostringstream oss;
    size_t currentLen = 0;

    for (size_t i = start; i < convo.size(); i++) {
        std::ostringstream turn;
        if (config.includeRoleLabels) {
            turn << "[" << convo[i]["role"] << "] ";
        }
        turn << convo[i]["content"];
        if (config.includeTimestamps && convo[i].contains("timestamp")) {
            turn << " (" << convo[i]["timestamp"] << ")";
        }
        turn << config.conversationSeparator;

        if (currentLen + turn.str().length() > maxConvoLen) break;
        oss << turn.str();
        currentLen += turn.str().length();
    }
    return oss.str();
}
