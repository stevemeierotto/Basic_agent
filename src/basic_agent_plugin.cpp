#include "../include/basic_agent_plugin.h"
#include "../include/conversation_authority.h"
#include "../include/corpus_create.h"
#include "../include/alp_feature_flags.h"
#include "../include/corpus_documents.h"
#include "../include/engine_error.h"
#include "../include/file_handler.h"
#include "../include/research_resources.h"
#include "../include/graph_statistics.h"
#include "../include/runtime_bootstrap.h"
#include "logger.h"
#include "../include/similarity.h"
#include "../include/standard_execution_mode.h"
#include "../include/executive_controller.h"
#include "../include/llm_planner.h"
#include "../include/fact_store.h"
#include "../include/planner_injection_config.h"
#include "../include/benchmark_context.h"
#include "../include/ollama_snapshot.h"
#include "../include/episode_event_channel.h"
#include "../include/evaluation_subscriber.h"
#include "../include/replay_subscriber.h"
#include "../include/metrics_subscriber.h"
#include "../include/trace_subscriber.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>
#include <sstream>

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

    episode_event_channel_ = std::make_shared<Thoth::InProcessEpisodeEventChannel>();
    controller->set_episode_event_channel(episode_event_channel_.get());

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
    config.applyEnvironmentOverrides();

    Thoth::logResolvedRuntimeConfig(&config);
    Thoth::logEmbeddingStartupProbe(&config);

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

    if (config.enable_episodic_evaluation_publication && episode_event_channel_) {
        Thoth::setEvaluationSubscriberPipelineTelemetryEnabled(
            config.enable_episodic_pipeline_telemetry);
        Thoth::registerEvaluationSubscriber(*episode_event_channel_);
        std::cerr << "[BasicAgentPlugin] E2-C2 episode publication enabled (EvaluationSubscriber)\n";
    }

    if (config.enable_episode_replay_subscriber && episode_event_channel_) {
        Thoth::registerReplaySubscriber(*episode_event_channel_);
        std::cerr << "[BasicAgentPlugin] E2-D2 replay subscriber enabled (ReplaySubscriber)\n";
    }

    if (config.enable_metrics_subscriber && episode_event_channel_) {
        Thoth::registerMetricsSubscriber(*episode_event_channel_);
        std::cerr << "[BasicAgentPlugin] E2-D3 metrics subscriber enabled (MetricsSubscriber)\n";
    }

    if (config.enable_trace_subscriber && episode_event_channel_) {
        Thoth::registerTraceSubscriber(*episode_event_channel_);
        std::cerr << "[BasicAgentPlugin] E2-D3 trace subscriber enabled (TraceSubscriber)\n";
    }

    // Set model based on config if available
    llm.setConfig(&config);
    ToolRegistry::instance().setConfig(&config);
    syncPlannerPromptConfig();
    cmdProcessor.syncPromptConfig();
    memory.configureConsolidation(&llm, rag.engine.get());
    memory.setGoalActiveChecker([this]() {
        return controller
            && controller->get_state() != Thoth::ControllerState::IDLE;
    });
    memory.runStartupConsolidationDiscovery();

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

Thoth::InProcessEpisodeEventChannel* BasicAgentPlugin::episodeEventChannelForTests() const {
    return episode_event_channel_.get();
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
    std::string trimmed = input;
    auto b = trimmed.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    auto e = trimmed.find_last_not_of(" \t\r\n");
    trimmed = trimmed.substr(b, e - b + 1);
    if (trimmed.empty()) return "";

    if (trimmed[0] == '/') {
        return cmdProcessor.handleCommand(trimmed);
    }

    return cmdProcessor.processQuery(trimmed);
}

void BasicAgentPlugin::setConversationMemory(const std::vector<std::pair<std::string, std::string>>& messages,
                                             const std::string& summary) {
    std::vector<Memory::TimedMessage> timed;
    timed.reserve(messages.size());
    for (const auto& msg : messages) {
        timed.push_back({msg.first, msg.second, 0});
    }
    setConversationMemory(timed, summary);
}

void BasicAgentPlugin::setConversationMemory(const std::vector<Memory::TimedMessage>& messages,
                                             const std::string& summary) {
    memory.loadConversation(messages, summary);
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

nlohmann::json BasicAgentPlugin::listStrategies() const {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& s : getAllStrategies()) {
        nlohmann::json stepPattern = nlohmann::json::array();
        try {
            stepPattern = nlohmann::json::parse(s.step_pattern_json);
        } catch (...) {
        }
        items.push_back({
            {"strategy_id", s.strategy_id},
            {"description", s.description},
            {"step_pattern", stepPattern},
            {"success_rate", s.success_rate},
            {"created_at", s.created_at},
        });
    }
    return Thoth::ResearchResources::makeCollection(items);
}

nlohmann::json BasicAgentPlugin::listTrajectories() const {
    auto trajs = getAllTrajectories();
    std::sort(trajs.begin(), trajs.end(), [](const auto& a, const auto& b) {
        return a.created_at > b.created_at;
    });
    if (trajs.size() > 20) {
        trajs.resize(20);
    }

    nlohmann::json items = nlohmann::json::array();
    for (const auto& t : trajs) {
        nlohmann::json trajectory = nlohmann::json::object();
        try {
            trajectory = nlohmann::json::parse(t.trajectory_json);
        } catch (...) {
        }
        items.push_back({
            {"trajectory_id", t.trajectory_id},
            {"goal", t.goal},
            {"trajectory", trajectory},
            {"success_score", t.success_score},
            {"created_at", t.created_at},
            {"usage_count", t.usage_count},
            {"tier", t.tier},
        });
    }
    return Thoth::ResearchResources::makeCollection(items);
}

nlohmann::json BasicAgentPlugin::listEpisodes() const {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& s : getAllEpisodeSteps()) {
        items.push_back({
            {"episode_id", s.episode_id},
            {"goal_id", s.goal_id},
            {"step_index", s.step_index},
            {"state_summary", s.state_summary},
            {"action_taken", s.action_taken},
            {"result_status", s.result_status},
            {"timestamp_ms", s.timestamp_ms},
        });
    }
    return Thoth::ResearchResources::makeCollection(items);
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

nlohmann::json BasicAgentPlugin::getGraphStatisticsResource() const {
    const auto stats = getGraphStatistics();
    const int64_t generated_at =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    nlohmann::json session_id = nullptr;
    if (controller) {
        const std::string sid = controller->get_session_id();
        if (!sid.empty()) {
            session_id = sid;
        }
    }
    return Thoth::GraphStatistics::makeResponse(
        Thoth::GraphStatistics::makeStatisticsPayload(stats.total_nodes,
                                                    stats.total_edges,
                                                    stats.avg_edge_weight,
                                                    stats.max_edge_weight,
                                                    stats.min_edge_weight,
                                                    stats.total_success_count,
                                                    stats.total_failure_count),
        generated_at,
        session_id);
}

nlohmann::json BasicAgentPlugin::listCorpusDocuments() const {
    if (!indexManager) {
        return Thoth::CorpusDocuments::emptyV1List();
    }
    FileHandler fh;
    return indexManager->listCorpusDocuments(fh.getRagDirectory());
}

std::string BasicAgentPlugin::getActiveSessionId() const {
    return memory.getActiveSessionId();
}

nlohmann::json BasicAgentPlugin::createCorpusDocument(const std::string& suggested_name,
                                                      const std::string& content,
                                                      const std::string& owner_context_id) {
    Thoth::CorpusCreate::CreateDocumentRequest request;
    request.suggested_name = suggested_name;
    request.content = content;
    request.owner_context_id = owner_context_id;
    return createCorpusDocument(request);
}

nlohmann::json BasicAgentPlugin::createCorpusDocument(
    const Thoth::CorpusCreate::CreateDocumentRequest& request) {
    if (!indexManager) {
        throw Thoth::EngineException(
            Thoth::EngineError::engineBusy("Index manager not initialized."));
    }

    IndexManager::CreateCorpusDocumentOptions options;
    options.content_hash = request.content_hash;
    options.local_source_mtime_sec = request.local_source_mtime_sec;
    options.force_replace = request.force_replace;
    options.dry_run = request.dry_run;

    FileHandler fh;
    const auto outcome = indexManager->createCorpusDocument(fh.getRagDirectory(),
                                                            request.suggested_name,
                                                            request.content,
                                                            request.owner_context_id,
                                                            options);
    if (!outcome.ok) {
        if (outcome.machine_code == "revision_in_flight") {
            nlohmann::json details{{"document_id", outcome.document_id}};
            throw Thoth::EngineException(Thoth::EngineError::conflict(
                "revision_in_flight", outcome.error, std::move(details)));
        }
        if (outcome.machine_code == "content_conflict") {
            nlohmann::json details{{"document_id", outcome.document_id}};
            throw Thoth::EngineException(Thoth::EngineError::conflict(
                "content_conflict", outcome.error, std::move(details)));
        }
        if (outcome.machine_code == "alp_misconfigured") {
            throw Thoth::EngineException(
                Thoth::EngineError::engineBusy(outcome.error));
        }
        throw Thoth::EngineException(Thoth::EngineError::invalidRequest(outcome.error));
    }

    if (request.dry_run) {
        return Thoth::CorpusCreate::makeDryRunResponse(
            outcome.action, outcome.document_id, outcome.document_name, outcome.action);
    }

    if (Thoth::AlpFeatureFlags::alpCreateAllowed()) {
        return Thoth::CorpusCreate::makeAlpAcceptedResponse(outcome.document_id,
                                                            outcome.document_name,
                                                            outcome.revision_id,
                                                            outcome.action);
    }
    return Thoth::CorpusCreate::makeAcceptedResponse(outcome.document_id, outcome.document_name);
}

namespace {

std::string newConversationSessionId() {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    std::random_device rd;
    std::uniform_int_distribution<std::uint32_t> dist;
    std::ostringstream out;
    out << "session-" << ms << "-" << std::hex << dist(rd);
    return out.str();
}

} // namespace

nlohmann::json BasicAgentPlugin::createConversationSession() {
    return Thoth::ConversationAuthority::makeCreateSessionResponse(newConversationSessionId());
}

nlohmann::json BasicAgentPlugin::appendUserTurn(const std::string& session_id,
                                                const std::string& content) {
    if (session_id.empty()) {
        throw Thoth::EngineException(
            Thoth::EngineError::invalidRequest("session_id must not be empty."));
    }
    if (content.empty()) {
        throw Thoth::EngineException(
            Thoth::EngineError::invalidRequest("content must not be empty."));
    }

    setSessionId(session_id);
    const std::string assistant_text = processInput(content);

    const auto messages = memory.getTimedMessages(session_id);
    if (messages.empty()) {
        throw Thoth::EngineException(
            Thoth::EngineError::internalError("Conversation store missing assistant turn."));
    }

    const Memory::TimedMessage& last = messages.back();
    if (last.role != "assistant") {
        throw Thoth::EngineException(
            Thoth::EngineError::internalError("Expected assistant turn after append."));
    }

    return Thoth::ConversationAuthority::makeAppendTurnResponse(
        session_id,
        Thoth::ConversationAuthority::makeMessage(last.role, last.content, last.timestamp_ms));
}

nlohmann::json BasicAgentPlugin::getConversationForSession(const std::string& session_id) const {
    if (session_id.empty()) {
        throw Thoth::EngineException(
            Thoth::EngineError::invalidRequest("session_id must not be empty."));
    }
    nlohmann::json messages = nlohmann::json::array();
    for (const auto& msg : memory.getTimedMessages(session_id)) {
        messages.push_back(
            Thoth::ConversationAuthority::makeMessage(msg.role, msg.content, msg.timestamp_ms));
    }
    return Thoth::ConversationAuthority::makeConversationResponse(session_id, messages);
}

nlohmann::json BasicAgentPlugin::getConversationSummaryForSession(
    const std::string& session_id) const {
    if (session_id.empty()) {
        throw Thoth::EngineException(
            Thoth::EngineError::invalidRequest("session_id must not be empty."));
    }
    return Thoth::ConversationAuthority::makeSummaryResponse(
        session_id, memory.getSummaryForSession(session_id));
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

Thoth::BenchmarkEnvironmentInputs BasicAgentPlugin::buildTestSuiteBenchmarkInputs(
    bool fullTier, const std::string& corpusPath) const {
    Thoth::BenchmarkEnvironmentInputs inputs;
    inputs.harness = "test_suite";
    inputs.tier = fullTier ? Thoth::BenchmarkTier::FULL : Thoth::BenchmarkTier::DEV;
    inputs.model.llm_model = fullTier ? config.llm_model : "mock";
    inputs.model.embedding_model = config.embedding_model;
    if (rag.engine) {
        switch (rag.engine->getMethod()) {
            case EmbeddingEngine::Method::TfIdf:
                inputs.model.embedding_method = "TfIdf";
                break;
            case EmbeddingEngine::Method::External:
                inputs.model.embedding_method = "External";
                break;
            case EmbeddingEngine::Method::Simple:
                inputs.model.embedding_method = "Simple";
                break;
            case EmbeddingEngine::Method::WordHash:
                inputs.model.embedding_method = "WordHash";
                break;
        }
        inputs.model.embedding_dimension = rag.engine->getDimension();
        inputs.model.embedding_internal_version = rag.engine->getInternalVersion();
    }
    inputs.corpus_paths = {corpusPath};
    inputs.corpus_mode = Thoth::CorpusFingerprintMode::FAST;
    if (indexManager) {
        inputs.corpus_chunk_count = static_cast<int>(indexManager->getChunks().size());
    }
    inputs.thoth_env_flags = Thoth::collectThothEnvFlags();
    if (fullTier) {
        inputs.ollama_reachable = Thoth::isOllamaReachable();
        if (inputs.ollama_reachable) {
            if (auto snap = Thoth::fetchOllamaSnapshot()) {
                inputs.ollama = *snap;
            }
        }
    }
    return inputs;
}

Thoth::IndexEnvironment BasicAgentPlugin::benchmarkIndexEnvironment() const {
    Thoth::IndexEnvironment index;
    if (!indexManager || !rag.engine) {
        return index;
    }
    index.rag_index_header = {
        {"model_name", rag.engine->getModelName()},
        {"embedding_dimension", rag.engine->getDimension()},
        {"embedding_version", rag.engine->getInternalVersion()},
        {"chunk_count", static_cast<int>(indexManager->getChunks().size())},
    };
    return index;
}
