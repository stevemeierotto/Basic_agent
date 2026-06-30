#include "../include/prompt_factory.h"
#include "../include/tools.h"
#include "../include/planner_injection_config.h"
#include "../include/goal_text_utils.h"
#include "../include/chat_prompt_config.h"
#include "file_handler.h"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <filesystem>
#include <vector>

namespace {

const char* kDefaultPlanGenerationTemplate =
    "Rules:\n"
    "- Generate a JSON plan using ONLY retrieval and synthesis.\n"
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
    "\"depends_on\":[\"retrieve-context\"],\"payload\":{}}]}\n"
    "Goal: {{goal}}\n"
    "{{strategy_context}}\n"
    "{{past_experience}}\n";

const char* kDefaultPlanRevisionTemplate =
    "Rules:\n"
    "- Revise the plan using ONLY retrieval and synthesis.\n"
    "- Respond with JSON only (no markdown fences).\n"
    "- Minimum 2 steps: RETRIEVAL first, then LLM with depends_on.\n"
    "- Do NOT emit TOOL steps.\n"
    "Schema:\n"
    "{\"plan\":[{\"step_id\":\"retrieve-context\",\"step_type\":\"RETRIEVAL\","
    "\"description\":\"Retrieve relevant corpus context\",\"payload\":{\"query\":\"...\",\"top_k\":5}},"
    "{\"step_id\":\"synthesize\",\"step_type\":\"LLM\",\"description\":\"Summarize findings\","
    "\"depends_on\":[\"retrieve-context\"],\"payload\":{}}]}\n"
    "Goal: {{goal}}\n"
    "Existing Plan: {{existing_plan}}\n"
    "Failed Step Result: {{failed_step_result}}\n";

} // namespace

namespace {

std::string extractSection(const std::string& text, const std::string& startMarker, const std::string& endMarker) {
    const std::size_t start = text.find(startMarker);
    if (start == std::string::npos) {
        return "";
    }
    const std::size_t contentStart = start + startMarker.size();
    const std::size_t end = endMarker.empty() ? std::string::npos : text.find(endMarker, contentStart);
    if (end == std::string::npos) {
        return text.substr(contentStart);
    }
    return text.substr(contentStart, end - contentStart);
}

std::string capCoreSection(const std::string& section, std::size_t maxChars) {
    if (section.size() <= maxChars) {
        return section;
    }
    return section.substr(0, maxChars);
}

std::string assemblePlannerPrompt(const std::string& rules,
                                   const std::string& schema,
                                   const std::string& goalBlock,
                                   const std::string& strategy_context,
                                   const std::string& past_experience,
                                   std::size_t totalBudget,
                                   Thoth::PlannerPromptMetrics* metrics) {
    const std::string rulesBlock = "Rules:\n" + rules;
    const std::string schemaBlock = "Schema:\n" + schema;

    std::ostringstream experience;
    if (!strategy_context.empty()) {
        experience << strategy_context;
        if (!past_experience.empty()) {
            experience << "\n";
        }
    }
    if (!past_experience.empty()) {
        experience << past_experience;
    }

    const std::size_t coreBytes = rulesBlock.size() + 2 + schemaBlock.size() + 2 + goalBlock.size() + 2;
    const std::size_t experienceBudget = coreBytes >= totalBudget ? 0 : (totalBudget - coreBytes);
    std::string experienceBlock = experience.str();
    bool experienceDropped = false;
    if (experienceBudget == 0 && !experienceBlock.empty()) {
        experienceBlock.clear();
        experienceDropped = true;
    } else if (experienceBlock.size() > experienceBudget) {
        experienceBlock = Thoth::capInjectionText(experienceBlock, experienceBudget);
    }

    std::ostringstream prompt;
    prompt << rulesBlock << "\n\n" << schemaBlock << "\n\n" << goalBlock;
    if (!experienceBlock.empty()) {
        prompt << "\n\n" << experienceBlock;
    }

    if (metrics) {
        metrics->rules_bytes = rulesBlock.size();
        metrics->schema_bytes = schemaBlock.size();
        metrics->goal_bytes = goalBlock.size();
        metrics->strategy_bytes = strategy_context.size();
        metrics->trajectory_bytes = past_experience.size();
        metrics->plan_reuse_bytes = 0;
        metrics->total_bytes = prompt.str().size();
        metrics->experience_dropped = experienceDropped;
    }

    return prompt.str();
}

} // namespace

// Default constructor uses default PromptConfig()
PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r)
    : PromptFactory(mem, r, PromptConfig()) {}

// Constructor with explicit config
PromptFactory::PromptFactory(Memory& mem, RAGPipeline& r, const PromptConfig& cfg)
    : memory(mem), rag(r), config(cfg) {}

std::string PromptFactory::buildConversationPrompt(const std::string& user_input,
                                                   bool useExtendedSummary,
                                                   Thoth::ConversationPromptMetrics* metrics) {
    ConversationBuildOptions options;
    options.includeTools = config.enableTools;
    options.grounded = false;
    return buildChatPrompt(user_input, "", useExtendedSummary, options, metrics);
}

std::string PromptFactory::assembleConversationSections(const std::string& user_input,
                                                        bool useExtendedSummary,
                                                        const ConversationBuildOptions& options,
                                                        std::size_t budget,
                                                        Thoth::ConversationPromptMetrics* metrics) {
    std::string systemPrompt = config.systemPrompt;
    if (systemPrompt.size() > Thoth::ChatPrompt::kMaxSystemPromptChars) {
        systemPrompt = systemPrompt.substr(0, Thoth::ChatPrompt::kMaxSystemPromptChars);
    }

    const std::string groundingRules =
        options.grounded ? std::string(Thoth::ChatPrompt::kGroundingRules) : "";
    const std::string toolList =
        (options.includeTools && config.enableTools) ? getToolList() : "";
    std::string memoryContext = getMemoryContext(useExtendedSummary);
    std::string conversationHistory = getConversationHistory();

    const std::string userBlock = "[User] " + user_input + "\n[Agent] ";

    const std::size_t coreBytes = groundingRules.size() + systemPrompt.size() + userBlock.size() + 8;

    auto trimFromStart = [](std::string& text, std::size_t targetSize) {
        if (text.size() <= targetSize) {
            return;
        }
        text = text.substr(text.size() - targetSize);
    };

    std::string droppedSection;
    bool toolsIncluded = false;
    if (coreBytes >= budget) {
        memoryContext.clear();
        conversationHistory.clear();
        droppedSection = "optional_sections";
    } else {
        std::size_t remaining = budget - coreBytes;

        if (!toolList.empty() && toolList.size() + 2 <= remaining) {
            toolsIncluded = true;
            remaining -= toolList.size() + 2;
        } else if (!toolList.empty()) {
            droppedSection = "tool_list";
        }

        if (!memoryContext.empty()) {
            const std::size_t memoryOverhead = std::string("[Memory Context]\n").size() + 2;
            const std::size_t memoryBudget =
                remaining > memoryOverhead ? remaining - memoryOverhead : 0;
            if (memoryContext.size() > memoryBudget) {
                if (droppedSection.empty()) {
                    droppedSection = "memory_context";
                }
                trimFromStart(memoryContext, memoryBudget);
            }
            if (memoryBudget > 0) {
                remaining = remaining > memoryOverhead + memoryContext.size()
                                ? remaining - memoryOverhead - memoryContext.size()
                                : 0;
            } else {
                memoryContext.clear();
            }
        }

        if (!conversationHistory.empty() && conversationHistory.size() > remaining) {
            if (droppedSection.empty()) {
                droppedSection = "conversation_history";
            }
            trimFromStart(conversationHistory, remaining);
        }
    }

    std::ostringstream assembled;
    if (!groundingRules.empty()) {
        assembled << groundingRules << "\n";
    }
    if (!systemPrompt.empty()) {
        assembled << systemPrompt << "\n\n";
    }
    if (toolsIncluded) {
        assembled << toolList << "\n";
    }
    if (!memoryContext.empty()) {
        assembled << "[Memory Context]\n" << memoryContext << "\n\n";
    }
    if (!conversationHistory.empty()) {
        assembled << conversationHistory;
    }
    assembled << userBlock;

    if (metrics) {
        metrics->system_prompt_chars = systemPrompt.size();
        metrics->grounding_rules_chars = groundingRules.size();
        metrics->tool_schema_chars = toolList.size();
        metrics->tools_included = toolsIncluded;
        metrics->tool_schema_chars_in_final = toolsIncluded ? toolList.size() : 0;
        metrics->memory_context_chars = memoryContext.size();
        metrics->conversation_history_chars = conversationHistory.size();
        metrics->user_input_chars = user_input.size();
        metrics->assembled_prompt_chars = assembled.str().size();
        metrics->truncated = !droppedSection.empty();
        metrics->truncated_section = droppedSection;
    }

    return assembled.str();
}

std::string PromptFactory::fitRagContextToBudget(const std::string& ragContext, std::size_t maxChars) {
    if (ragContext.size() <= maxChars) {
        return ragContext;
    }
    if (maxChars <= 32) {
        return ragContext.substr(0, maxChars);
    }
    return ragContext.substr(0, maxChars - 24) + "\n...(RAG truncated)...";
}

std::string PromptFactory::buildChatPrompt(const std::string& user_input,
                                           const std::string& ragContext,
                                           bool useExtendedSummary,
                                           const ConversationBuildOptions& options,
                                           Thoth::ConversationPromptMetrics* metrics) {
    ConversationBuildOptions effectiveOptions = options;
    if (effectiveOptions.grounded && ragContext.empty()) {
        effectiveOptions.grounded = false;
    }

    const std::size_t totalBudget = config.maxContextLength;
    const std::size_t ragHeaderChars =
        ragContext.empty()
            ? 0
            : std::char_traits<char>::length(Thoth::ChatPrompt::kRagContextHeader) +
                  std::char_traits<char>::length(Thoth::ChatPrompt::kUserQueryHeader) + 1;

    std::string systemPrompt = config.systemPrompt;
    if (systemPrompt.size() > Thoth::ChatPrompt::kMaxSystemPromptChars) {
        systemPrompt = systemPrompt.substr(0, Thoth::ChatPrompt::kMaxSystemPromptChars);
    }
    const std::string groundingRules =
        effectiveOptions.grounded ? std::string(Thoth::ChatPrompt::kGroundingRules) : "";
    const std::string userBlock = "[User] " + user_input + "\n[Agent] ";
    const std::size_t minCore = groundingRules.size() + systemPrompt.size() + userBlock.size() + 8;

    std::size_t ragBudget = 0;
    if (!ragContext.empty() && totalBudget > minCore + ragHeaderChars) {
        const std::size_t available = totalBudget - minCore - ragHeaderChars;
        ragBudget = std::min(ragContext.size(), available * 3 / 5);
    }

    const std::size_t convBudget =
        totalBudget > ragBudget + ragHeaderChars ? totalBudget - ragBudget - ragHeaderChars : minCore;

    const std::string conversationPart = assembleConversationSections(
        user_input, useExtendedSummary, effectiveOptions, convBudget, metrics);

    std::ostringstream finalPrompt;
    std::size_t ragChars = 0;
    if (!ragContext.empty()) {
        const std::string fittedRag = fitRagContextToBudget(ragContext, ragBudget);
        ragChars = fittedRag.size();
        finalPrompt << Thoth::ChatPrompt::kRagContextHeader << fittedRag << '\n'
                    << Thoth::ChatPrompt::kUserQueryHeader;
    }

    finalPrompt << conversationPart;
    const std::string result = finalPrompt.str();

    if (metrics) {
        metrics->rag_context_chars = ragChars;
        metrics->final_prompt_chars = result.size();
        if (ragContext.size() > ragChars) {
            metrics->truncated = true;
            metrics->truncated_section =
                metrics->truncated_section.empty() ? "rag_context" : metrics->truncated_section;
        }
    }

    return result;
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
                                            const std::string& past_experience,
                                            Thoth::PlannerPromptMetrics* metrics) {
    const std::string templateStr = loadTemplate("plan_generation.tmpl", kDefaultPlanGenerationTemplate);

    std::string rules = extractSection(templateStr, "Rules:\n", "Schema:\n");
    std::string schema = extractSection(templateStr, "Schema:\n", "Goal:");
    if (rules.empty()) {
        rules = extractSection(kDefaultPlanGenerationTemplate, "Rules:\n", "Schema:\n");
    }
    if (schema.empty()) {
        schema = extractSection(kDefaultPlanGenerationTemplate, "Schema:\n", "Goal:");
    }

    rules = capCoreSection(rules, Thoth::PlannerInjection::kMaxRulesChars);
    schema = capCoreSection(schema, Thoth::PlannerInjection::kMaxSchemaChars);

    const std::string cappedGoal = Thoth::capInjectionText(goal, Thoth::PlannerInjection::kMaxGoalChars);
    const std::string goalBlock = "Goal: " + cappedGoal;

    const std::string cappedStrategy =
        Thoth::capInjectionText(strategy_context, Thoth::PlannerInjection::kMaxStrategyContextChars);
    const std::string cappedPast =
        Thoth::capInjectionText(past_experience, Thoth::PlannerInjection::kMaxTrajectoryChars);

    const size_t budget = std::max(config.maxContextLength, Thoth::PlannerInjection::kMinPlanPromptBudget);
    return assemblePlannerPrompt(rules, schema, goalBlock, cappedStrategy, cappedPast, budget, metrics);
}

std::string PromptFactory::buildRevisionPrompt(const std::string& goal,
                                                const std::string& existing_plan_json,
                                                const std::string& failed_step_result_json,
                                                Thoth::PlannerPromptMetrics* metrics) {
    const std::string templateStr = loadTemplate("plan_revision.tmpl", kDefaultPlanRevisionTemplate);

    std::string rules = extractSection(templateStr, "Rules:\n", "Schema:\n");
    std::string schema = extractSection(templateStr, "Schema:\n", "Goal:");
    if (rules.empty()) {
        rules = extractSection(kDefaultPlanRevisionTemplate, "Rules:\n", "Schema:\n");
    }
    if (schema.empty()) {
        schema = extractSection(kDefaultPlanRevisionTemplate, "Schema:\n", "Goal:");
    }

    rules = capCoreSection(rules, Thoth::PlannerInjection::kMaxRulesChars);
    schema = capCoreSection(schema, Thoth::PlannerInjection::kMaxSchemaChars);

    const std::string cappedGoal = Thoth::capInjectionText(goal, Thoth::PlannerInjection::kMaxGoalChars);
    std::ostringstream goalBlock;
    goalBlock << "Goal: " << cappedGoal << "\n";
    goalBlock << "Existing Plan: " << existing_plan_json << "\n";
    goalBlock << "Failed Step Result: " << failed_step_result_json;

    const size_t budget = std::max(config.maxContextLength, Thoth::PlannerInjection::kMinPlanPromptBudget);
    return assemblePlannerPrompt(rules, schema, goalBlock.str(), "", "", budget, metrics);
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
    std::ostringstream memSummary;
    const auto warmRows = memory.getRecentWarmMemory(3);
    for (auto it = warmRows.rbegin(); it != warmRows.rend(); ++it) {
        if (!it->rendered_summary.empty()) {
            memSummary << "[Episodic Memory]\n" << it->rendered_summary << "\n\n";
        }
    }

    memSummary << memory.getSummary(useExtendedSummary);
    std::string combined = memSummary.str();
    const size_t maxMemLen = config.maxContextLength / 2;
    if (combined.length() > maxMemLen) {
        combined = combined.substr(combined.length() - maxMemLen);
    }
    return combined;
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
