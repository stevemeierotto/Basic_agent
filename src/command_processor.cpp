#include "command_processor.h"
#include "file_handler.h"
#include "index_manager.h"
#include "logger.h"
#include "tools.h"
#include "standard_execution_mode.h"
#include "scientific_execution_mode.h"
#include "decision_trace.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <mutex>
#include <sstream>
#include <regex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <json.hpp>

namespace fs = std::filesystem;

namespace {
std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::int64_t elapsedMs(std::int64_t startMs) {
    return nowMs() - startMs;
}

std::string benchmarkHistoryPath() {
    FileHandler fh;
    return fh.getAgentWorkspacePath("benchmark_history.jsonl");
}

std::vector<nlohmann::json> readBenchmarkHistory(std::size_t limit) {
    std::ifstream in(benchmarkHistoryPath());
    std::vector<nlohmann::json> entries;
    if (!in.is_open()) {
        return entries;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            entries.push_back(nlohmann::json::parse(line));
        } catch (...) {
        }
    }

    if (entries.size() > limit) {
        entries.erase(entries.begin(), entries.end() - static_cast<std::ptrdiff_t>(limit));
    }
    return entries;
}
}

static std::string backendToString(LLMBackend backend) {
    switch (backend) {
        case LLMBackend::Ollama:
            return "ollama";
        case LLMBackend::OpenAI:
            return "openai";
        default:
            return "unknown";
    }
}

CommandProcessor::CommandProcessor(Memory& mem, 
                                   RAGPipeline& ragPipeline, 
                                   LLMInterface& llmInterface,
                                   Config* cfg,
                                   std::shared_ptr<Thoth::ExecutiveController> ctrl)
    : memory(mem),
      rag(ragPipeline),
      llm(llmInterface),
      controller(ctrl),
      promptFactory(mem, ragPipeline),
    indexManager(ragPipeline.getIndexManager()),
    config(cfg)
{
    initializeCommands();
}

void CommandProcessor::syncPromptConfig() {
    try {
        if (!config) return;
        auto pCfg = promptFactory.getConfig();
        pCfg.enableTools = config->enable_tools;
        pCfg.maxContextLength = static_cast<size_t>(config->max_tokens * 4);
        promptFactory.setConfig(pCfg);
    } catch (...) {}
}

bool CommandProcessor::hasDangerousArgPattern(const std::string& args) const {
    static const std::regex dangerousPattern(R"((\|\||&&|;|`|\$\(|\n|\r))");
    return std::regex_search(args, dangerousPattern);
}

void CommandProcessor::handleSimilarityCommand(const std::string& args) {
    try {
        std::unordered_map<std::string, std::unique_ptr<ISimilarity>> options;
        options["dot"] = std::make_unique<DotProductSimilarity>();
        options["cosine"] = std::make_unique<CosineSimilarity>();
        options["euclidean"] = std::make_unique<EuclideanSimilarity>();
        options["jaccard"] = std::make_unique<JaccardSimilarity>();

        std::string chosen = toLower(trim(args));

        if (chosen.empty()) {
            std::cout << "Available similarity methods:\n";
            for (auto& [name, _] : options) std::cout << "  " << name << "\n";
            std::cout << "Enter choice: ";
            std::getline(std::cin, chosen);
            chosen = toLower(trim(chosen));
        }

        auto it = options.find(chosen);
        if (it == options.end()) {
            std::cout << "Unknown similarity: " << chosen << "\n";
            return;
        }

        rag.getIndexManager()->store.setSimilarity(std::move(it->second));
        std::cout << "Similarity set to " << chosen << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error setting similarity: " << e.what() << "\n";
    }
}


std::string CommandProcessor::processQuery(const std::string& input) {
    auto trace = traceLogger.startTrace("query", input.size());
    
    try {
        StructuredLogger::instance().log(
            LogLevel::Info,
            "command_processor",
            "query_start",
            "Processing query request",
            {{"input_length", input.size()}},
            trace.requestId);

        // Phase 1 Stabilization: Log routing decision
        bool goalActive = (controller && controller->get_state() != Thoth::ControllerState::IDLE);
        std::string mode = goalActive ? "PLAN_AWARE" : "CONVERSATIONAL";
        nlohmann::json indexes = goalActive ? nlohmann::json::array({"codebase_index"}) : nlohmann::json::array({"conversations_index"});
        
        StructuredLogger::instance().log(
            LogLevel::Info,
            "command_processor",
            "routing_decision",
            "Determining retrieval strategy",
            {
                {"goal_active", goalActive},
                {"routing_mode", mode},
                {"indexes_selected", indexes},
                {"reason", goalActive ? "active goal detected" : "no active goal detected"}
            },
            trace.requestId);

        traceLogger.addStage(
            trace,
            "intent",
            true,
            "classified as user_query",
            {{"is_command", false}});

        syncPromptConfig();
        ensureInitialized();

        // --- Sync Goal Context from Controller ---
        if (controller && controller->get_state() != Thoth::ControllerState::IDLE) {
            auto currentPlan = controller->get_current_plan();
            if (!currentPlan.plan_id.empty()) {
                std::string stepId = "";
                if (currentPlan.current_index < currentPlan.steps.size()) {
                    stepId = currentPlan.steps[currentPlan.current_index].step_id;
                }
                rag.setPlanContext(currentPlan.plan_id, stepId);
            }
        }

        if (input.size() > maxQueryLength) {
            traceLogger.addStage(
                trace,
                "input_validation",
                false,
                "query rejected by max length check",
                {{"max_query_length", maxQueryLength}});
            traceLogger.finishTrace(trace, false, "denied_input_too_large");
            traceLogger.writeTrace(trace);
            return "[Denied] input exceeds maximum allowed length.";
        }

        std::string queryDenyReason;
        if (!isQueryAllowed(queryDenyReason)) {
            traceLogger.addStage(
                trace,
                "policy",
                false,
                "query denied by policy",
                {{"reason", queryDenyReason}});
            traceLogger.finishTrace(trace, false, "denied_policy");
            traceLogger.writeTrace(trace);
            return "[Denied] " + queryDenyReason;
        }

        if (indexManager && indexManager->getChunks().empty()) {
            const auto generationStartMs = nowMs();
            std::string convPrompt = promptFactory.buildConversationPrompt(input);
            std::string response = llm.query(convPrompt);
            const auto generationLatencyMs = elapsedMs(generationStartMs);

            traceLogger.addStage(
                trace,
                "generation",
                true,
                "LLM response generated (no RAG)",
                {{"backend", backendToString(llm.getBackend())},
                 {"generation_latency_ms", generationLatencyMs}});

            std::string finalResponse = processToolCall(response, trace);

            try {
                memory.addMessage("user", input);
                memory.addMessage("assistant", finalResponse);
                memory.save();
                memory.updateSummary(input, finalResponse);
            } catch (...) {}

            traceLogger.finishTrace(trace, true, "query_completed_without_rag");
            traceLogger.writeTrace(trace);
            return finalResponse;
        }

        // 1. Retrieve context
        std::string activePlanId = "";
        std::string activeStepId = "";
        if (controller && controller->get_state() != Thoth::ControllerState::IDLE) {
            auto currentPlan = controller->get_current_plan();
            activePlanId = currentPlan.plan_id;
            if (currentPlan.current_index < currentPlan.steps.size()) {
                activeStepId = currentPlan.steps[currentPlan.current_index].step_id;
            }
            
            // Sync embeddings for GRAG math
            rag.setGoalEmbedding(controller->get_goal_embedding());
            rag.setCurrentEmbedding(controller->get_current_embedding());
        }

        std::vector<CodeChunk> contextChunks = rag.retrieveRelevant(input, {}, 5, trace.requestId, activePlanId, activeStepId);

        std::ostringstream contextStream;
        for (auto& c : contextChunks) {
            contextStream << c.code << "\n---\n";
        }
        std::string ragContext = contextStream.str();

        // 2. Build prompt
        std::string convPrompt = promptFactory.buildConversationPrompt(input);
        std::string finalPrompt;
        if (!ragContext.empty()) {
            finalPrompt = "[RAG Context]\n" + ragContext + "\n[User Query]\n" + convPrompt;
        } else {
            finalPrompt = convPrompt;
        }

        // 3. Query LLM
        const auto generationStartMs = nowMs();
        std::string response = llm.query(finalPrompt);
        const auto generationLatencyMs = elapsedMs(generationStartMs);

        traceLogger.addStage(
            trace,
            "generation",
            true,
            "LLM response generated",
            {{"backend", backendToString(llm.getBackend())},
             {"generation_latency_ms", generationLatencyMs}});

        std::string finalResponse = processToolCall(response, trace);

        // 4. Update memory
        try {
            memory.addMessage("user", input);
            memory.addMessage("assistant", finalResponse);
            memory.save();
            memory.updateSummary(input, finalResponse);
        } catch (...) {}

        traceLogger.finishTrace(trace, true, "query_completed");
        traceLogger.writeTrace(trace);
        return finalResponse;

    } catch (const std::exception& e) {
        traceLogger.finishTrace(trace, false, std::string("Query failed with exception: ") + e.what());
        traceLogger.writeTrace(trace);
        return std::string("[Error] Query failed: ") + e.what();
    } catch (...) {
        traceLogger.finishTrace(trace, false, "Query failed with unknown exception");
        traceLogger.writeTrace(trace);
        return "[Error] Query failed with an unknown error.";
    }
}

std::string CommandProcessor::processToolCall(const std::string& response, DecisionTrace& trace) {
    try {
        nlohmann::json responseJson = nlohmann::json::parse(response);
        if (responseJson.contains("tool_call") && responseJson["tool_call"].is_object()) {
            auto toolCall = responseJson["tool_call"];
            if (toolCall.contains("name") && toolCall["name"].is_string() &&
                toolCall.contains("input") && toolCall["input"].is_object()) {
                
                std::string toolName = toolCall["name"].get<std::string>();
                nlohmann::json toolInput = toolCall["input"];

                // NEW: Global Constraint Check (Standard Chat Loop)
                nlohmann::json checkPayload = toolInput;
                checkPayload["tool_name"] = toolName;
                
                // Map tool names to action types for ConstraintChecker
                std::string actionType = "tool_call";
                if (toolName == "code_modify") {
                    std::string op = toolInput.value("operation", "");
                    if (op == "read") actionType = "file_read";
                    else if (op == "apply_diff") actionType = "file_modify";
                } else if (toolName == "web_scrape") {
                    actionType = "network_request";
                } else if (toolName.find("gmail") == 0) {
                    actionType = "network_request";
                    checkPayload["url"] = "google.com"; // Generic for gmail tools
                }

                auto checkResult = constraint_checker_.check_action(actionType, checkPayload);
                if (!checkResult.allowed) {
                    traceLogger.addStage(
                        trace,
                        "tool_execution",
                        false,
                        "Tool BLOCKED by security policy: " + toolName,
                        {{"reason", checkResult.reason}});
                    
                    return nlohmann::json({
                        {"status", "error"},
                        {"error_message", "Action blocked by security policy: " + checkResult.reason}
                    }).dump();
                }

                const auto toolStartMs = nowMs();
                nlohmann::json toolResult = ToolRegistry::instance().executeTool(toolName, toolInput);
                const auto toolLatencyMs = elapsedMs(toolStartMs);

                std::string status = toolResult.value("status", "error");
                bool toolSuccess = (status == "success");

                traceLogger.addStage(
                    trace,
                    "tool_execution",
                    toolSuccess,
                    "Executed tool: " + toolName,
                    {{"latency_ms", toolLatencyMs}, {"status", status}});

                return toolResult.dump();
            }
        }
    } catch (...) {
    }
    return response;
}

std::string CommandProcessor::trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string CommandProcessor::lstripSlash(const std::string& s) {
    if (!s.empty() && s[0] == '/') return s.substr(1);
    return s;
}

std::string CommandProcessor::toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

bool CommandProcessor::startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

void CommandProcessor::runLoop() {
    std::cout << "Basic Chat Agent. Type /help for commands. Type exit or quit to leave.\n";
    std::string line;

    try {
        ensureInitialized();
        std::cout << "RAG system ready.\n";
    } catch (const std::exception& e) {
        std::cout << "Warning: RAG system initialization failed: " << e.what() << "\n";
    }

    while (true) {
        std::cout << "<USER> " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;
        }

        line = trim(line);
        if (line.empty()) continue;

        std::string low = toLower(line);
        if (low == "exit" || low == "quit" || low == "/exit" || low == "/quit") {
            std::cout << "Goodbye.\n";
            break;
        }
        handleCommand(line);
    }
}

void CommandProcessor::showConfig() const {
    try {
        if(config) config->printConfig();
        else std::cout << "No config connected.\n";
    } catch (...) {}
}

void CommandProcessor::setConfig(const std::string& key, const std::string& value) {
    try {
        if (config) {
            if (config->set(key, value)) {
                std::cout << "Updated " << key << " to " << value << "\n";
            } else {
                std::cout << "Failed to update key: " << key << "\n";
            }
        } else {
            std::cout << "No config connected.\n";
        }
    } catch (...) {}
}


void CommandProcessor::handleCommand(const std::string& input) {
    try {
        syncPromptConfig();
        ensureInitialized();
        if (!startsWith(input, "/")) {
            std::string response = processQuery(input);
            std::cout << "Assistant: " << response << "\n";
            try { memory.updateSummary(input, response); } catch (...) {}
            return;
        }

        auto [cmd, args] = parseCommand(input);
        auto trace = traceLogger.startTrace("command", input.size());

        traceLogger.addStage(
            trace,
            "intent",
            true,
            "classified as command",
            {{"command", cmd}});

        if (args.size() > maxCommandArgsLength) {
            traceLogger.finishTrace(trace, false, "denied_command_args_too_large");
            traceLogger.writeTrace(trace);
            std::cout << "[Denied] command arguments exceed maximum allowed length.\n";
            return;
        }

        std::string denyReason;
        if (!isCommandAllowed(cmd, args, denyReason)) {
            traceLogger.finishTrace(trace, false, "denied_policy");
            traceLogger.writeTrace(trace);
            std::cout << "[Denied] " << denyReason << "\n";
            return;
        }

        auto it = commandHandlers.find(cmd);
        if (it != commandHandlers.end()) {
            try {
                it->second(args);
                traceLogger.finishTrace(trace, true, "command_completed");
                traceLogger.writeTrace(trace);
            } catch (const std::exception& e) {
                traceLogger.finishTrace(trace, false, std::string("Command exception: ") + e.what());
                traceLogger.writeTrace(trace);
                std::cout << "Error executing command: " << e.what() << "\n";
            }
        } else {
            std::cout << "Unknown command '/" << cmd << "'. Try /help.\n";
            traceLogger.finishTrace(trace, false, "command_unknown");
            traceLogger.writeTrace(trace);
        }
    } catch (const std::exception& e) {
        std::cout << "Fatal error in command handler: " << e.what() << "\n";
    } catch (...) {
        std::cout << "Unknown fatal error in command handler.\n";
    }
}

bool CommandProcessor::isQueryAllowed(std::string& reason) const {
    if (!config) return true;
    if (llm.getBackend() == LLMBackend::OpenAI && !config->allow_network) {
        reason = "network access disabled by policy (allow_network=false).";
        return false;
    }
    return true;
}

bool CommandProcessor::isCommandAllowed(const std::string& cmd, const std::string& args, std::string& reason) const {
    if (!config) return true;
    if ((cmd == "rag" || cmd == "clear" || cmd == "reset" || cmd == "benchmark") && !config->allow_file_io) {
        reason = "file I/O disabled by policy (allow_file_io=false).";
        return false;
    }
    if (cmd == "backend") {
        const std::string backendArg = toLower(trim(args));
        if (backendArg == "openai" && !config->allow_network) {
            reason = "cannot switch to OpenAI: network access disabled by policy (allow_network=false).";
            return false;
        }
    }
    if (hasDangerousArgPattern(args)) {
        reason = "command arguments include blocked shell-like patterns.";
        return false;
    }
    return true;
}


std::pair<std::string, std::string> CommandProcessor::parseCommand(const std::string& input) {
    std::string stripped = input;
    if (!stripped.empty() && stripped[0] == '/')
        stripped.erase(0, 1);

    std::istringstream iss(stripped);
    std::string cmd;
    std::getline(iss, cmd, ' ');
    std::string args;
    std::getline(iss, args);
    return { toLower(cmd), trim(args) };
}

void CommandProcessor::showHelp() {
    std::cout <<
        "Built-ins:\n"
        "  /help               Show this help\n"
        "  /rag                Query knowledge with RAG\n"
        "  /clear              Clears agent's memory and summaries\n"
        "  /backend ollama     Switch to Ollama\n"
        "  /backend openai     Switch to OpenAI\n"
        "  /mode [std|sci]     Switch execution mode\n"
        "  /similarity         Switch Similarity\n"
        "  /benchmark ...      Run retrieval/index benchmarks\n"
        "  /config             Show config values\n"
        "  /set key value      Update config\n"
        "Also: type 'exit' or 'quit' to leave.\n";
}

void CommandProcessor::clearMemory() {
    try {
        memory.clear();
        memory.save();
        std::cout << "Memory cleared.\n";
    } catch (...) {}
}

void CommandProcessor::ensureInitialized() {
    if (!initialized) {
        try {
            FileHandler fh;
            indexManager->init(fh.getRagPath("rag_index.bin"));
            
            if (indexManager->getChunks().empty()) {
                std::string sandboxRag = fh.getAgentWorkspacePath("rag/");
                std::cerr << "[RAG] Index empty. Bootstrapping from sandbox: " << sandboxRag << "\n";
                indexManager->indexProject(sandboxRag);
                indexManager->saveIndex();
            }
            initialized = true;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] RAG initialization failed: " << e.what() << "\n";
            initialized = true;
        } catch (...) {
            std::cerr << "[ERROR] Unknown error during RAG initialization.\n";
            initialized = true;
        }
    }
}

void CommandProcessor::initializeCommands() {
    commandHandlers["help"] = [this](const std::string&) { showHelp(); };
    commandHandlers["h"] = commandHandlers["help"];
    commandHandlers["?"] = commandHandlers["help"];
    
    commandHandlers["clear"] = [this](const std::string&) { clearMemory(); };
    commandHandlers["reset"] = commandHandlers["clear"];
    commandHandlers["rag"] = [this](const std::string& args) { handleRag(args); };
    commandHandlers["backend"] = [this](const std::string& args) { handleBackend(args); };
    commandHandlers["mode"] = [this](const std::string& args) {
        if (!controller) return;
        std::string modeStr = toLower(trim(args));
        if (modeStr == "standard" || modeStr == "std") {
            controller->set_execution_mode(std::make_unique<Thoth::StandardExecutionMode>());
            std::cout << "[Controller] Switched to Standard mode.\n";
        } else if (modeStr == "scientific" || modeStr == "sci") {
            controller->set_execution_mode(std::make_unique<Thoth::ScientificExecutionMode>());
            std::cout << "[Controller] Switched to Scientific mode.\n";
        }
    };
    commandHandlers["similarity"] = [this](const std::string& args) { handleSimilarityCommand(args); };
    commandHandlers["config"] = [this](const std::string&) { showConfig(); };
    commandHandlers["benchmark"] = [this](const std::string& args) { handleBenchmark(args); };
    commandHandlers["goal"] = [this](const std::string& args) {
        if (!controller || args.empty()) return;
        controller->execute_goal(args);
    };
    commandHandlers["set"] = [this](const std::string& args) {
        std::istringstream iss(args);
        std::string k, v;
        iss >> k >> v;
        if (!k.empty() && !v.empty()) setConfig(k, v);
    };
}

void CommandProcessor::handleBenchmark(const std::string& args) {
    try {
        ensureInitialized();
        std::istringstream iss(args);
        std::string mode;
        iss >> mode;
        mode = toLower(trim(mode));

        if (mode == "index") {
            FileHandler fh;
            const auto start = nowMs();
            indexManager->indexProject(fh.getRagDirectory());
            indexManager->saveIndex();
            std::cout << "[Benchmark] Indexing took " << elapsedMs(start) << "ms\n";
        } else if (mode == "retrieve") {
            std::string q; std::getline(iss, q);
            const auto start = nowMs();
            rag.retrieveRelevant(q, {}, 5);
            std::cout << "[Benchmark] Retrieval took " << elapsedMs(start) << "ms\n";
        } else if (mode == "history") {
            auto entries = readBenchmarkHistory(10);
            if (entries.empty()) {
                std::cout << "[Benchmark] No history found.\n";
                return;
            }
            for (const auto& e : entries) {
                std::cout << " - " << e.dump() << "\n";
            }
        }
    } catch (...) {}
}

void CommandProcessor::handleRag(const std::string& args) {
    try {
        if (args.empty()) return;
        auto chunks = rag.retrieveRelevant(args, {}, 5);
        for (const auto& c : chunks) {
            std::cout << "File: " << c.fileName << "\n" << c.code << "\n---\n";
        }
    } catch (...) {}
}

void CommandProcessor::handleBackend(const std::string& args) {
    try {
        if (args == "ollama") llm.setBackend(LLMBackend::Ollama);
        else if (args == "openai") llm.setBackend(LLMBackend::OpenAI);
    } catch (...) {}
}
