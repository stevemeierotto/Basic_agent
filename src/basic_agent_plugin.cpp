#include "../include/basic_agent_plugin.h"
#include "../include/file_handler.h"
#include "logger.h"
#include "../include/similarity.h"
#include "../include/standard_execution_mode.h"
#include "../include/llm_planner.h"
#include "../include/fact_store.h"
#include <filesystem>
#include <fstream>

BasicAgentPlugin::BasicAgentPlugin()
    : config(),
      memory(config),
      embeddingEngine(std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::External, &config)),
      indexManager(new IndexManager(embeddingEngine.get())),
      rag(std::move(embeddingEngine), indexManager, &config, &memory),
      cmdProcessor(memory, rag, llm, &config) 
{
    // Setup Planner
    auto memory_ptr = std::shared_ptr<Memory>(&memory, [](Memory*) {});
    auto rag_ptr = std::shared_ptr<RAGPipeline>(&rag, [](RAGPipeline*){});
    auto prompt_factory = std::make_shared<PromptFactory>(memory, rag);
    planner = std::make_shared<LLMPlanner>(memory_ptr, rag_ptr, prompt_factory, &llm);

    // Initialize FactStore and ToolRegistry
    if (auto sqlite_repo = memory.getSQLiteRepo()) {
        auto fact_store = std::make_shared<Thoth::FactStore>(*sqlite_repo);
        ToolRegistry::instance().initialize(fact_store, &llm);
    }

    // Wrap the ToolRegistry singleton
    auto registry_ptr = std::shared_ptr<ToolRegistry>(&ToolRegistry::instance(), [](ToolRegistry*) {});

    controller = std::make_shared<Thoth::ExecutiveController>(planner, registry_ptr, rag_ptr, memory_ptr);
    cmdProcessor.setController(controller);
    
    // Set event callback
    auto cb = [this](const ControllerEvent& ev) {
        if (this->onEvent) this->onEvent(ev);
    };
    controller->set_event_callback(cb);
    rag.setEventCallback(cb);

    FileHandler fileHandler;

    // --- Load config.json ---
    std::string configPath = fileHandler.getAgentWorkspacePath("config.json");
    if (std::filesystem::exists(configPath)) {
        std::ifstream f(configPath);
        if (f.is_open()) {
            try {
                nlohmann::json j = nlohmann::json::parse(f);
                config.loadFromJson(j);
                std::cerr << "[BasicAgentPlugin] Config loaded from: " << configPath << "\n";
            } catch (...) {
                std::cerr << "[BasicAgentPlugin] Error parsing config.json\n";
            }
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

    // --- Initialize RAG index ---
    std::string ragIndexPath = fileHandler.getRagPath("rag_index.bin");
    if (std::filesystem::exists(ragIndexPath)) {
        indexManager->init(ragIndexPath);
        std::cerr << "[BasicAgentPlugin] RAG index loaded successfully.\n";
    } else {
        std::cerr << "[BasicAgentPlugin] RAG index path: " << ragIndexPath << "\n";
    }
}

BasicAgentPlugin::~BasicAgentPlugin() {
    delete indexManager;
    std::cerr << "[BasicAgentPlugin] Destroyed.\n";
}

std::string BasicAgentPlugin::processInput(const std::string& input) {
    if (input.empty()) return "";

    if (input[0] == '/') {
        cmdProcessor.handleCommand(input);
        return "[Command Executed]";
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

void BasicAgentPlugin::setRagFiles(const std::vector<std::string>& filePaths) {
    for (const auto& path : filePaths) {
        if (std::filesystem::exists(path)) {
            if (std::filesystem::is_directory(path)) {
                indexManager->indexProject(path);
            } else {
                indexManager->indexFile(path);
            }
        }
    }
    cmdProcessor.setInitialized(true);
}
