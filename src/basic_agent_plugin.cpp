#include "../include/basic_agent_plugin.h"
#include "../include/file_handler.h"
#include "logger.h"
#include "../include/similarity.h"
#include "../include/standard_execution_mode.h"
#include "../include/llm_planner.h"
#include "../include/fact_store.h"
#include "../include/planner_injection_config.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdlib>

BasicAgentPlugin::BasicAgentPlugin()
    : config(),
      memory(config),
      embeddingEngine(std::make_unique<EmbeddingEngine>(
          []() {
              const char* dev = std::getenv("THOTH_TEST_SUITE_DEV");
              if (dev && (std::string(dev) == "1" || std::string(dev) == "true")) {
                  return EmbeddingEngine::Method::TfIdf;
              }
              return EmbeddingEngine::Method::External;
          }(),
          &config)),
      indexManager(new IndexManager(embeddingEngine.get())),
      rag(std::move(embeddingEngine), indexManager, &config, &memory),
      cmdProcessor(memory, rag, llm, &config) 
{
    // Setup Planner
    auto memory_ptr = std::shared_ptr<Memory>(&memory, [](Memory*) {});
    auto rag_ptr = std::shared_ptr<RAGPipeline>(&rag, [](RAGPipeline*){});
    planner_prompt_factory_ = std::make_shared<PromptFactory>(memory, rag);
    planner = std::make_shared<LLMPlanner>(memory_ptr, rag_ptr, planner_prompt_factory_, &llm);

    // Initialize FactStore and ToolRegistry
    if (auto sqlite_repo = memory.getSQLiteRepo()) {
        auto fact_store = std::make_shared<Thoth::FactStore>(*sqlite_repo);
        ToolRegistry::instance().initialize(fact_store, &llm);
    }

    // Wrap the ToolRegistry singleton
    auto registry_ptr = std::shared_ptr<ToolRegistry>(&ToolRegistry::instance(), [](ToolRegistry*) {});

    controller = std::make_shared<Thoth::ExecutiveController>(planner, registry_ptr, rag_ptr, memory_ptr);
    cmdProcessor.setController(controller);
    controller->set_llm_interface(&llm);
    controller->set_config(&config);
    controller->set_max_reflections(config.max_reflections);
    
    // Set event callback
    auto cb = [this](const ControllerEvent& ev) {
        if (this->onEvent) this->onEvent(ev);
    };
    controller->set_event_callback(cb);
    rag.setEventCallback(cb);
    indexManager->setEventCallback(cb);

    FileHandler fileHandler;
    PromptFactory::ensureDefaultTemplatesExist();

    // --- Load config.json ---
    std::string configPath = fileHandler.getAgentWorkspacePath("config.json");
    if (std::filesystem::exists(configPath)) {
        if (config.loadFromJson(configPath)) {
            std::cerr << "[BasicAgentPlugin] Config loaded from: " << configPath << "\n";
        } else {
            std::cerr << "[BasicAgentPlugin] Error parsing config.json\n";
        }
    }

    // --- Load retrieval_config.json (Phase 5.1) ---
    std::string retConfigPath = fileHandler.getAgentWorkspacePath("retrieval_config.json");
    if (std::filesystem::exists(retConfigPath)) {
        if (config.loadRetrievalConfig(retConfigPath)) {
            std::cerr << "[BasicAgentPlugin] Retrieval config loaded from: " << retConfigPath << "\n";
        } else {
            std::cerr << "[BasicAgentPlugin] Error loading retrieval_config.json\n";
        }
    } else {
        // Save defaults if not exists
        config.saveRetrievalConfig(retConfigPath);
        std::cerr << "[BasicAgentPlugin] Created default retrieval_config.json\n";
    }

    // Set model based on config if available
    llm.setConfig(&config);
    ToolRegistry::instance().setConfig(&config);
    syncPlannerPromptConfig();
    cmdProcessor.syncPromptConfig();

    // --- Initialize RAG index ---
    const char* testIndexPath = std::getenv("THOTH_TEST_SUITE_INDEX");
    if (testIndexPath && *testIndexPath) {
        indexManager->init(testIndexPath);
        std::cerr << "[BasicAgentPlugin] TEST_SUITE index path: " << testIndexPath << "\n";
    } else {
        std::string ragIndexPath = fileHandler.getRagPath("rag_index.bin");
        if (std::filesystem::exists(ragIndexPath)) {
            indexManager->init(ragIndexPath);
            std::cerr << "[BasicAgentPlugin] RAG index loaded successfully.\n";
        } else {
            std::cerr << "[BasicAgentPlugin] RAG index path: " << ragIndexPath << "\n";
        }
    }
}

BasicAgentPlugin::~BasicAgentPlugin() {
    delete indexManager;
    std::cerr << "[BasicAgentPlugin] Destroyed.\n";
}

void BasicAgentPlugin::bootstrapSandboxIfEmpty() {
    if (!indexManager) return;
    if (indexManager->getChunks().empty()) {
        FileHandler fh;
        const std::string corpus = fh.getAgentWorkspacePath("rag/test_suite_corpus");
        std::filesystem::create_directories(corpus);
        std::cerr << "[BasicAgentPlugin] Indexing TEST_SUITE corpus: " << corpus << "\n";
        indexManager->indexProject(corpus);
        indexManager->saveIndex();
    }
    cmdProcessor.setInitialized(true);
}

std::string BasicAgentPlugin::processInput(const std::string& input) {
    if (input.empty()) return "";

    if (input[0] == '/') {
        return cmdProcessor.handleCommand(input);
    }

    return cmdProcessor.processQuery(input);
}

void BasicAgentPlugin::setConversationMemory(const std::vector<std::pair<std::string, std::string>>& messages,
                                             const std::string& summary) {
    memory.clear();
    for (const auto& msg : messages) {
        memory.addMessage(msg.first, msg.second);
    }
    if (!summary.empty()) {
        memory.updateSummary("Imported context", summary);
    }
}

bool BasicAgentPlugin::ragPathsNeedIndexing(const std::vector<std::string>& filePaths) const {
    if (!indexManager) {
        return false;
    }

    for (const auto& path : filePaths) {
        if (!std::filesystem::exists(path)) {
            continue;
        }
        if (std::filesystem::is_directory(path)) {
            try {
                for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
                    if (!entry.is_regular_file()) {
                        continue;
                    }
                    const std::string fullPath =
                        std::filesystem::absolute(entry.path()).lexically_normal().string();
                    if (indexManager->shouldReindexFile(fullPath)) {
                        return true;
                    }
                }
            } catch (...) {
                return true;
            }
            continue;
        }

        try {
            const std::string normalized =
                std::filesystem::absolute(path).lexically_normal().string();
            if (indexManager->shouldReindexFile(normalized)) {
                return true;
            }
        } catch (...) {
            return true;
        }
    }
    return false;
}

void BasicAgentPlugin::setRagFiles(const std::vector<std::string>& filePaths) {
    if (filePaths == lastRagFilePaths_ && !ragPathsNeedIndexing(filePaths)) {
        cmdProcessor.setInitialized(true);
        return;
    }

    for (const auto& path : filePaths) {
        if (std::filesystem::exists(path)) {
            if (std::filesystem::is_directory(path)) {
                indexManager->indexProject(path);
            } else {
                indexManager->indexFile(path);
            }
        }
    }
    indexManager->setActiveCorpusFiles(filePaths);
    lastRagFilePaths_ = filePaths;

    if (const char* cachePath = std::getenv("THOTH_TEST_SUITE_INDEX")) {
        if (cachePath[0] != '\0') {
            indexManager->saveIndex(cachePath);
        }
    }

    cmdProcessor.setInitialized(true);
}

void BasicAgentPlugin::setSessionId(const std::string& sessionId) {
    if (!sessionId.empty()) {
        memory.setActiveSessionId(sessionId);
    }
    if (controller) controller->set_session_id(sessionId);
    cmdProcessor.set_session_id(sessionId);
    if (indexManager) indexManager->setSessionId(sessionId);
}

void BasicAgentPlugin::checkResumablePlan() {
    if (!controller) return;
    auto plan = controller->get_resumable_plan();
    if (plan) {
        controller->resume_from_plan(*plan);
        
        ControllerEvent ev;
        ev.type = EventType::PLAN_CREATED;
        ev.session_id = controller->get_session_id();
        ev.plan_id = plan->plan_id;
        ev.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        ev.metadata = {{"plan", plan->to_json()}};
        if (onEvent) onEvent(ev);
    }
}

std::vector<Memory::CognateStrategyRecord> BasicAgentPlugin::getAllStrategies() const {
    return memory.getAllStrategies();
}

std::vector<Memory::CognateTrajectoryRecord> BasicAgentPlugin::getAllTrajectories() const {
    return memory.getAllTrajectories();
}

std::vector<Memory::EpisodeStepRecord> BasicAgentPlugin::getAllEpisodeSteps() const {
    return memory.getAllEpisodeSteps();
}

std::vector<Memory::CognateExperimentRecord> BasicAgentPlugin::getAllExperiments() const {
    return memory.getAllExperiments();
}

bool BasicAgentPlugin::saveExperiment(const Memory::CognateExperimentRecord& record) {
    return memory.saveExperiment(record);
}

Memory::GraphStatistics BasicAgentPlugin::getGraphStatistics() const {
    return memory.getGraphStatistics();
}

void BasicAgentPlugin::syncPlannerPromptConfig() {
    if (!planner_prompt_factory_) {
        return;
    }
    auto pCfg = planner_prompt_factory_->getConfig();
    pCfg.enableTools = config.enable_tools;
    const size_t fromTokens = static_cast<size_t>(config.max_tokens) * 4;
    pCfg.maxContextLength = std::max(fromTokens, Thoth::PlannerInjection::kMinPlanPromptBudget);
    planner_prompt_factory_->setConfig(pCfg);
}
