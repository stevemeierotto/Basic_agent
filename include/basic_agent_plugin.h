#pragma once

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "memory.h"
#include "rag.h"
#include "command_processor.h"
#include "llm_interface.h"
#include "runtime_bootstrap.h"
#include "embedding_engine.h"
#include "executive_controller.h"
#include "default_planner.h"
#include "iplanner.h"
#include "prompt_factory.h"
#include "benchmark_environment.h"
#include "episode_event_channel.h"
#include "corpus_create.h"
#include "json.hpp"

#include <optional>

//#include "Plugin.h"  // Plugin interface

class BasicAgentPlugin {
public:
    BasicAgentPlugin();
    ~BasicAgentPlugin();

    // Event system for UI integration
    std::function<void(const ControllerEvent&)> onEvent;

    std::string processInput(const std::string& input,
                             const std::optional<std::string>& active_goal = std::nullopt,
                             const std::string& task_id = {},
                             const std::string& raw_capture_id = {});
    void setConversationMemory(
        const std::vector<std::pair<std::string, std::string>>& messages,
        const std::string& summary = "");
    void setConversationMemory(
        const std::vector<Memory::TimedMessage>& messages,
        const std::string& summary = "");
    void setRagFiles(const std::vector<std::string>& filePaths);
    void setSessionId(const std::string& sessionId);

    void checkResumablePlan();

    // --- Executive Control ---
    void pause() { if (controller) controller->pause(); }
    void resume() { if (controller) controller->resume(); }
    void abort() { if (controller) controller->abort(); }
    void executeGoal(const std::string& goal,
                     const Thoth::BenchmarkAttribution& benchmark = {},
                     const std::string& task_id = {}) {
        if (controller) {
            controller->execute_goal(goal, benchmark, task_id);
        }
    }

    /** MTCP — production revise_plan with a frozen plan. Empty when the planner is absent. */
    nlohmann::json revisePlanForMtcp(const nlohmann::json& plan_json,
                                     const nlohmann::json& failed_step_result);

    /** Headless TEST_SUITE: index sandbox rag/ only if empty, then skip full re-init. */
    void bootstrapSandboxIfEmpty();

    /** E1 D1: build benchmark inputs after setRagFiles (index + corpus ready). */
    Thoth::BenchmarkEnvironmentInputs buildTestSuiteBenchmarkInputs(bool fullTier,
                                                                    const std::string& corpusPath) const;
    Thoth::IndexEnvironment benchmarkIndexEnvironment() const;

    /** E2-D3-05: testing only — production episode channel after plugin init. */
    Thoth::InProcessEpisodeEventChannel* episodeEventChannelForTests() const;

    // --- Cognate UI Integration ---
    std::vector<Memory::CognateStrategyRecord> getAllStrategies() const;
    std::vector<Memory::CognateTrajectoryRecord> getAllTrajectories() const;
    std::vector<Memory::EpisodeStepRecord> getAllEpisodeSteps() const;
    std::vector<Memory::CognateExperimentRecord> getAllExperiments() const;
    bool saveExperiment(const Memory::CognateExperimentRecord& record);
    Memory::GraphStatistics getGraphStatistics() const;

    /** Phase 12A — Engine-owned graph statistics singleton resource. */
    nlohmann::json getGraphStatisticsResource() const;

    /** Phase 8 — Engine-owned corpus document list. */
    nlohmann::json listCorpusDocuments() const;

    /** Phase 9 — create corpus document (acceptance); indexing via INDEXING_* events. */
    nlohmann::json createCorpusDocument(const std::string& suggested_name,
                                        const std::string& content,
                                        const std::string& owner_context_id = "");

    /** ALP-C — extended create with hash/mtime/force_replace/dry_run. */
    nlohmann::json createCorpusDocument(const Thoth::CorpusCreate::CreateDocumentRequest& request);

    /** ALP amend — remove session↔document link (Local Note X). */
    nlohmann::json unlinkSessionDocument(const std::string& document_id,
                                         const std::string& session_id);

    /** TCB3 / §3.0 — active session for local ingest bind (v1 context key source). */
    std::string getActiveSessionId() const;

    /** Phase 10 — Engine-owned conversation authority. */
    nlohmann::json createConversationSession();
    nlohmann::json appendUserTurn(const std::string& session_id,
                                  const std::string& content,
                                  const std::optional<std::string>& active_goal = std::nullopt);
    nlohmann::json getConversationForSession(const std::string& session_id) const;
    nlohmann::json getConversationSummaryForSession(const std::string& session_id) const;

    /** Phase 11 — Engine-owned research resource collections. */
    nlohmann::json listStrategies() const;
    nlohmann::json listTrajectories() const;
    nlohmann::json listEpisodes() const;

    // --- Implement Plugin interface ---
    bool initialize()  {
        std::cerr << "[BasicAgentPlugin] initialize() called." << std::endl;
        return true;
    }

    void shutdown()  {
        std::cerr << "[BasicAgentPlugin] shutdown() called." << std::endl;

    }

    std::string name() const  {
        return "BasicAgentPlugin";
    }

    std::string execute(const std::string& input)  {
        // just call processInput
        return processInput(input);
    }

private:
    Thoth::RuntimeBootstrapGuard bootstrap_guard_;
    Config config;
    Memory memory;
    std::unique_ptr<EmbeddingEngine> embeddingEngine;
    IndexManager* indexManager;
    RAGPipeline rag;
    LLMInterface llm;
    CommandProcessor cmdProcessor;
    std::shared_ptr<IPlanner> planner;
    std::shared_ptr<PromptFactory> planner_prompt_factory_;
    std::shared_ptr<Thoth::ExecutiveController> controller;
    std::shared_ptr<Thoth::InProcessEpisodeEventChannel> episode_event_channel_;

    std::vector<std::string> lastRagFilePaths_;
    bool ragPathsNeedIndexing(const std::vector<std::string>& filePaths) const;
    void syncPlannerPromptConfig();
};

