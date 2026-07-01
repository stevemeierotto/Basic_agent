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
#include "env_loader.h"
#include "embedding_engine.h"
#include "executive_controller.h"
#include "default_planner.h"
#include "iplanner.h"
#include "prompt_factory.h"
#include "benchmark_environment.h"

//#include "Plugin.h"  // Plugin interface

class BasicAgentPlugin {
public:
    BasicAgentPlugin();
    ~BasicAgentPlugin();

    // Event system for UI integration
    std::function<void(const ControllerEvent&)> onEvent;

    std::string processInput(const std::string& input);
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
                     const Thoth::BenchmarkAttribution& benchmark = {}) {
        if (controller) {
            controller->execute_goal(goal, benchmark);
        }
    }

    /** Headless TEST_SUITE: index sandbox rag/ only if empty, then skip full re-init. */
    void bootstrapSandboxIfEmpty();

    /** E1 D1: build benchmark inputs after setRagFiles (index + corpus ready). */
    Thoth::BenchmarkEnvironmentInputs buildTestSuiteBenchmarkInputs(bool fullTier,
                                                                    const std::string& corpusPath) const;
    Thoth::IndexEnvironment benchmarkIndexEnvironment() const;

    // --- Cognate UI Integration ---
    std::vector<Memory::CognateStrategyRecord> getAllStrategies() const;
    std::vector<Memory::CognateTrajectoryRecord> getAllTrajectories() const;
    std::vector<Memory::EpisodeStepRecord> getAllEpisodeSteps() const;
    std::vector<Memory::CognateExperimentRecord> getAllExperiments() const;
    bool saveExperiment(const Memory::CognateExperimentRecord& record);
    Memory::GraphStatistics getGraphStatistics() const;

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

    std::vector<std::string> lastRagFilePaths_;
    bool ragPathsNeedIndexing(const std::vector<std::string>& filePaths) const;
    void syncPlannerPromptConfig();
};

