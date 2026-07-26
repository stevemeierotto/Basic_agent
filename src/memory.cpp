#include "../include/memory.h"
#include "../include/grag_scorer.h"
#include "../include/plan_reuse_config.h"
#include "../include/planner_injection_config.h"
#include "../include/sqlite_memory_repository.h"
#include "../include/memory_consolidation_config.h"
#include "../include/embedding_engine.h"
#include "../include/llm_interface.h"
#include "../include/clock.h"
#include "../include/decision_trace.h"
#include "../include/file_handler.h"
#include <iostream>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <filesystem>

using json = nlohmann::json;

namespace fs = std::filesystem;

Memory::Memory(const Config& config)
    : config_(&config),
      clock_(Thoth::makeSystemClock()) {
    FileHandler fh;
    const std::string dbPath =
        (fs::path(fh.getAgentWorkspacePath()) / "memory.db").string();
    repo = Thoth::MemoryRepositoryFactory::createRepository(config, dbPath);
    
    if (!repo) {
        std::cerr << "[Memory] CRITICAL: Failed to initialize repository backend.\n";
    } else if (config_) {
        const auto policy = Thoth::PruningPolicy::fromConfig(*config_);
        pruner = std::make_unique<Thoth::MemoryPruner>(*repo, policy, nullptr, nullptr, clock_);
    }

    // Phase 4: Embedding Migration
    migrateEmbeddings();
    
    // Phase 7.7: Load index
    loadTrajectoryIndex();
}

Memory::~Memory() = default;

void Memory::setActiveSessionId(const std::string& sessionId) {
    if (sessionId.empty()) {
        return;
    }
    {
        std::unique_lock lock(mtx);
        activeSessionId = sessionId;
    }
    onSessionActivated(sessionId);
}

void Memory::onSessionActivated(const std::string& sessionId) {
    if (!pruner) {
        return;
    }

    // Activating a session is a deliberate state change; give consolidation a
    // fresh attempt even if it had backed off after earlier failures.
    pruner->resetConsolidationBackoff(sessionId);

    bool markedStale = false;
    {
        std::shared_lock lock(mtx);
        markedStale = stale_session_ids_.count(sessionId) > 0;
    }

    const auto decision = pruner->evaluatePolicy(sessionId);
    if (!markedStale && !decision.shouldConsolidate()) {
        return;
    }

    consolidateIfNeeded(sessionId);

    {
        std::unique_lock lock(mtx);
        stale_session_ids_.erase(sessionId);
    }
}

std::string Memory::getActiveSessionId() const {
    std::shared_lock lock(mtx);
    return activeSessionId;
}

void Memory::migrateEmbeddings() {
    std::unique_lock lock(mtx);
    if (!repo) return;

    // Check current version
    std::string current_v_str = repo->getMeta("embedding_schema_version");
    int current_v = current_v_str.empty() ? 1 : std::stoi(current_v_str);

    if (current_v < 2) {
        std::cout << "[Memory] Phase 4: Migrating embedding schema from v1 to v2...\n";
        repo->clearPlanEmbeddings(1);
        repo->setMeta("embedding_schema_version", "2");
        std::cout << "[Memory] Migration complete. All v1 embeddings invalidated.\n";
    }
}

int64_t Memory::currentTimeMs() const {
    return clock_ ? clock_->nowMs() : Thoth::makeSystemClock()->nowMs();
}

void Memory::addMessage(const std::string& role, const std::string& content) {
    addMessageWithTimestamp(role, content, 0, true);
}

void Memory::addMessageWithTimestamp(const std::string& role,
                                     const std::string& content,
                                     int64_t timestamp_ms,
                                     bool triggerConsolidation) {
    std::string sessionForConsolidate;
    {
        std::unique_lock lock(mtx);
        if (!repo) return;

        repo->createSession(activeSessionId, currentTimeMs());

        Thoth::MessageRecord msg;
        msg.role = role;
        msg.content = content;
        msg.timestamp_ms = timestamp_ms > 0 ? timestamp_ms : currentTimeMs();

        repo->appendMessage(activeSessionId, msg);
        if (triggerConsolidation) {
            sessionForConsolidate = activeSessionId;
        }
    }

    if (!sessionForConsolidate.empty()) {
        consolidateIfNeeded(sessionForConsolidate);
    }
}

void Memory::loadConversation(const std::vector<TimedMessage>& messages,
                              const std::string& summary) {
    clear();
    for (const auto& message : messages) {
        addMessageWithTimestamp(message.role, message.content, message.timestamp_ms, false);
    }
    if (!summary.empty()) {
        updateSummary("Imported context", summary);
    }
    std::string sessionId;
    {
        std::shared_lock lock(mtx);
        sessionId = activeSessionId;
    }
    if (!sessionId.empty()) {
        consolidateIfNeeded(sessionId);
    }
}

std::vector<json> Memory::getConversation() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    auto messages = repo->getMessages(activeSessionId);
    std::vector<json> result;
    for (const auto& m : messages) {
        result.push_back({{"role", m.role}, {"content", m.content}});
    }
    return result;
}

std::vector<Memory::TimedMessage> Memory::getTimedMessages(const std::string& sessionId) const {
    std::shared_lock lock(mtx);
    if (!repo) {
        return {};
    }
    std::vector<TimedMessage> result;
    for (const auto& record : repo->getMessages(sessionId)) {
        result.push_back({record.role, record.content, record.timestamp_ms});
    }
    return result;
}

std::string Memory::getSummaryForSession(const std::string& sessionId,
                                         bool useExtended) const {
    std::shared_lock lock(mtx);
    if (!repo) {
        return {};
    }
    return repo->getSummary(sessionId, useExtended ? "extended" : "short");
}

void Memory::clear() {
    std::unique_lock lock(mtx);
    if (repo) repo->clearMessages(activeSessionId);
}

std::vector<Memory::CognateTrajectoryRecord> Memory::retrieveSimilarTrajectories(const std::vector<float>& target_embedding, int limit) const {
    std::shared_lock lock(mtx);
    if (!repo || target_embedding.empty()) return {};

    std::vector<std::pair<std::string, float>> ranked_ids;

    for (const auto& entry : trajectory_index_) {
        // Boost successful plans
        float score = GragScorer::cosine_similarity(target_embedding, entry.goal_embedding);
        if (entry.success_score >= 0.8f) score += 0.1f;
        
        ranked_ids.push_back({entry.trajectory_id, score});
    }

    std::sort(ranked_ids.begin(), ranked_ids.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::vector<CognateTrajectoryRecord> results;
    int count = std::min((int)ranked_ids.size(), limit);
    for (int i = 0; i < count; ++i) {
        if (ranked_ids[i].second < Thoth::PlannerTrajectory::kMinSimilarityFloor) {
            break;
        }
        auto t = repo->loadTrajectory(ranked_ids[i].first);
        if (t) results.push_back(std::move(*t));
    }

    return results;
}

void Memory::processMemoryAging() {
    std::unique_lock lock(mtx);
    if (!repo) return;

    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
    
    constexpr int64_t ONE_DAY_MS = 1000LL * 60 * 60 * 24;
    const int64_t THIRTY_DAYS_MS = 30LL * ONE_DAY_MS;

    bool index_changed = false;

    // 1. Trajectory Aging
    for (auto& entry : trajectory_index_) {
        int old_tier = entry.tier;
        int64_t age = now_ms - entry.created_at;

        if (age < ONE_DAY_MS) {
            entry.tier = TIER_HOT;
        }
        else if (entry.success_score >= 0.8f || entry.usage_count > 5) {
            entry.tier = TIER_WARM;
        }
        else {
            entry.tier = TIER_COLD;
        }

        if (entry.tier != old_tier) {
            index_changed = true;
            auto traj = repo->loadTrajectory(entry.trajectory_id);
            if (traj) {
                traj->tier = entry.tier;
                repo->saveTrajectory(*traj);
            }
        }
    }

    // 2. Graph Memory Aging (Phase 5.6)
    auto edges = repo->getAllEdges();
    for (auto& edge : edges) {
        // Global Decay
        edge.weight *= 0.995f;

        // Frequency-Based (Dormancy) Penalty
        if (now_ms - edge.last_used_ms > THIRTY_DAYS_MS) {
            edge.weight *= 0.97f;
        }

        // Pruning
        if (edge.weight < 0.02f) {
            repo->deleteEdge(edge.from_id, edge.to_id);
        } else {
            repo->addEdge(edge); // Update weight
        }
    }

    if (index_changed) {
        std::cout << "[Memory] Aging process completed. Tiers updated.\n";
    }
}

void Memory::loadTrajectoryIndex() {
    std::unique_lock lock(mtx);
    if (!repo) return;

    trajectory_index_.clear();
    auto all = repo->getAllTrajectories();
    for (const auto& t : all) {
        TrajectoryIndexEntry entry;
        entry.trajectory_id = t.trajectory_id;
        entry.goal_embedding = t.embedding;
        entry.success_score = t.success_score;
        entry.created_at = t.created_at;
        entry.usage_count = t.usage_count;
        entry.tier = t.tier;
        trajectory_index_.push_back(std::move(entry));
    }
}

void Memory::updateTrajectoryIndex(const CognateTrajectoryRecord& record) {
    std::unique_lock lock(mtx);
    
    // Find existing or add new
    auto it = std::find_if(trajectory_index_.begin(), trajectory_index_.end(), [&](const auto& e) {
        return e.trajectory_id == record.trajectory_id;
    });

    if (it != trajectory_index_.end()) {
        it->goal_embedding = record.embedding;
        it->success_score = record.success_score;
        it->usage_count = record.usage_count;
        it->tier = record.tier;
    } else {
        TrajectoryIndexEntry entry;
        entry.trajectory_id = record.trajectory_id;
        entry.goal_embedding = record.embedding;
        entry.success_score = record.success_score;
        entry.created_at = record.created_at;
        entry.usage_count = record.usage_count;
        entry.tier = record.tier;
        trajectory_index_.push_back(std::move(entry));
    }
}

std::vector<Memory::TrajectoryIndexEntry> Memory::getTrajectoryIndex() const {
    std::shared_lock lock(mtx);
    return trajectory_index_;
}

void Memory::addNode(const Node& node) {
    std::unique_lock lock(mtx);
    if (repo) repo->addNode(node);
}

void Memory::addEdge(const Edge& edge) {
    std::unique_lock lock(mtx);
    if (repo) repo->addEdge(edge);
}

std::vector<Memory::Node> Memory::getNodesByType(const std::string& type) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getNodesByType(type);
}

std::vector<Memory::Edge> Memory::getEdgesFrom(const std::string& nodeId) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getEdgesFrom(nodeId);
}

Memory::GraphStatistics Memory::getGraphStatistics() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getGraphStatistics();
}

std::string Memory::calculateContentHash(const std::string& text) {
    std::size_t h = std::hash<std::string>{}(text);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%zx", h);
    return std::string(buf);
}

void Memory::storeEpisodeStep(const EpisodeStepRecord& step) {
    std::unique_lock lock(mtx);
    if (repo) repo->storeEpisodeStep(step);
}

std::vector<Memory::EpisodeStepRecord> Memory::getRecentEpisodeSteps(const std::string& goal_id, int n) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getRecentEpisodeSteps(goal_id, n);
}

std::vector<Memory::EpisodeStepRecord> Memory::getAllEpisodeSteps() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getAllEpisodeSteps();
}

std::shared_ptr<Thoth::MemoryRepository> Memory::getRepo() const {
    return std::shared_ptr<Thoth::MemoryRepository>(repo.get(), [](Thoth::MemoryRepository*){});
}

std::vector<Thoth::MemoryRepository::ArchivedTurnRecord> Memory::getArchivedTurns() const {
    std::shared_lock lock(mtx);
    if (!pruner) return {};
    return pruner->restore(activeSessionId);
}

void Memory::configureConsolidation(LLMInterface* llm,
                                    EmbeddingEngine* embeddingEngine,
                                    std::shared_ptr<Thoth::Clock> clock) {
    std::unique_lock lock(mtx);
    if (!repo || !config_) return;
    if (clock) {
        clock_ = std::move(clock);
    } else if (!clock_) {
        clock_ = Thoth::makeSystemClock();
    }
    const auto policy = Thoth::PruningPolicy::fromConfig(*config_);
    pruner = std::make_unique<Thoth::MemoryPruner>(*repo, policy, llm, embeddingEngine, clock_);
}

Thoth::ConsolidationDecision Memory::evaluateConsolidationPolicy(const std::string& sessionId) const {
    std::shared_lock lock(mtx);
    if (!pruner) {
        return {};
    }
    return pruner->evaluatePolicy(sessionId);
}

bool Memory::isSessionMarkedStale(const std::string& sessionId) const {
    std::shared_lock lock(mtx);
    return stale_session_ids_.count(sessionId) > 0;
}

void Memory::setGoalActiveChecker(std::function<bool()> checker) {
    std::unique_lock lock(mtx);
    goal_active_checker_ = std::move(checker);
}

Thoth::ConsolidationStatus Memory::getConsolidationStatus(const std::string& sessionId) const {
    std::shared_lock lock(mtx);
    const std::string resolved = sessionId.empty() ? activeSessionId : sessionId;
    if (!pruner) {
        Thoth::ConsolidationStatus status;
        status.session_id = resolved;
        return status;
    }
    const bool goal_active = goal_active_checker_ ? goal_active_checker_() : false;
    const bool marked_stale = stale_session_ids_.count(resolved) > 0;
    return pruner->buildStatus(resolved, marked_stale, goal_active);
}

Thoth::ConsolidationResult Memory::runConsolidation(const std::string& sessionId,
                                                    const Thoth::ConsolidationRequest& request) {
    std::unique_lock lock(mtx);
    if (!pruner) {
        Thoth::ConsolidationResult result;
        result.blocked = true;
        result.block_reason = "Consolidation not configured.";
        return result;
    }

    const std::string resolved = sessionId.empty() ? activeSessionId : sessionId;
    const bool goal_active = goal_active_checker_ ? goal_active_checker_() : false;

    if (request.source == Thoth::ConsolidationSource::MANUAL
        && goal_active
        && !request.allow_during_goal) {
        Thoth::ConsolidationResult result;
        result.source = request.source;
        result.blocked = true;
        result.block_reason =
            "Goal in progress. Consolidation blocked until goal completes. Use --unsafe to override.";
        result.decision = pruner->evaluatePolicy(resolved);
        result.remaining_hot = result.decision.hot_count;
        result.total_archived = 0;
        result.batches_completed = 0;
        result.final_decision = result.decision;
        return result;
    }

    auto result = pruner->runConsolidation(resolved, request);
    if (result.archived > 0 || result.deferred) {
        stale_session_ids_.erase(resolved);
    }
    return result;
}

Thoth::RestoreResult Memory::runRestore(const std::string& sessionId,
                                        const Thoth::RestoreRequest& request) {
    std::unique_lock lock(mtx);
    if (!pruner) {
        Thoth::RestoreResult result;
        result.mode = request.mode;
        result.blocked = true;
        result.block_reason = "Restore not configured.";
        return result;
    }

    const std::string resolved = sessionId.empty() ? activeSessionId : sessionId;
    const bool goal_active = goal_active_checker_ ? goal_active_checker_() : false;

    if (request.mode == Thoth::RestoreMode::REHYDRATE
        && goal_active
        && !request.allow_during_goal) {
        Thoth::RestoreResult result;
        result.mode = request.mode;
        result.blocked = true;
        result.block_reason =
            "Goal in progress. Rehydrate blocked until goal completes. Use --unsafe to override.";
        return result;
    }

    return pruner->restore(resolved, request);
}

void Memory::runStartupConsolidationDiscovery() {
    if (!pruner || !repo) {
        return;
    }

    std::vector<std::string> sessionIds;
    std::string activeSession;
    {
        std::shared_lock lock(mtx);
        sessionIds = repo->getAllSessionIds();
        activeSession = activeSessionId;
    }

    std::vector<std::string> markedStale;
    for (const auto& sessionId : sessionIds) {
        const auto decision = pruner->evaluatePolicy(sessionId);
        if (!decision.shouldConsolidate()) {
            continue;
        }
        if (sessionId == activeSession) {
            continue;
        }
        {
            std::unique_lock lock(mtx);
            stale_session_ids_.insert(sessionId);
        }
        markedStale.push_back(sessionId);
    }

    if (!markedStale.empty()) {
        DecisionTraceLogger logger;
        DecisionTrace trace = logger.startTrace("memory_consolidation", static_cast<int>(markedStale.size()));
        logger.addStage(trace, "stale_sessions_discovered", true,
                        "Discovered stale sessions (no LLM work during discovery)", {
            {"stale_session_count", static_cast<int>(markedStale.size())},
            {"stale_session_ids", markedStale},
            {"active_session_id", activeSession}
        });
        logger.finishTrace(trace, true, "Stale session discovery complete");
        logger.writeTrace(trace);
    }

    if (!activeSession.empty()) {
        const auto decision = pruner->evaluatePolicy(activeSession);
        if (decision.shouldConsolidate()) {
            consolidateIfNeeded(activeSession);
        }
    }
}

void Memory::consolidateIfNeeded(const std::string& sessionId) {
    if (!pruner || sessionId.empty()) {
        return;
    }

    const auto result = pruner->consolidateIfNeeded(sessionId);
    if (result.archived > 0 || result.deferred) {
        const auto labels = Thoth::consolidationReasonsToStrings(result.decision.reasons);
        std::cerr << "[Memory] Consolidated " << result.archived << " turn(s) for session "
                  << sessionId;
        if (!labels.empty() && labels[0] != "NONE") {
            std::cerr << " (reasons:";
            for (const auto& label : labels) {
                if (label != "NONE") {
                    std::cerr << ' ' << label;
                }
            }
            std::cerr << ')';
        }
        if (result.deferred) {
            std::cerr << (result.archived > 0
                ? " [deferred: batch cap reached]"
                : " [deferred: no forward progress — consolidation failing]");
        }
        std::cerr << '\n';
    }
}

std::vector<Thoth::MemoryRepository::WarmMemoryRecord> Memory::getRecentWarmMemory(int limit) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getRecentWarmMemory(activeSessionId, limit);
}

std::vector<Thoth::MemoryRepository::WarmMemoryRecord> Memory::searchWarmMemory(
    const std::vector<float>& queryEmbedding, int limit) const {
    std::shared_lock lock(mtx);
    if (!repo || queryEmbedding.empty()) return {};
    return repo->searchWarmMemoryByEmbedding(
        activeSessionId,
        Thoth::MemoryScope::SESSION,
        queryEmbedding,
        limit,
        Thoth::MemoryConsolidation::kWarmMemoryEmbeddingVersion);
}

std::vector<Thoth::MemoryRepository::WarmMemoryRecord> Memory::searchWarmMemoryAllSessions(
    const std::vector<float>& queryEmbedding, int limit) const {
    std::shared_lock lock(mtx);
    if (!repo || queryEmbedding.empty()) return {};
    return repo->searchWarmMemoryByEmbedding(
        "",
        Thoth::MemoryScope::SESSION,
        queryEmbedding,
        limit,
        Thoth::MemoryConsolidation::kWarmMemoryEmbeddingVersion);
}

void Memory::markDirty() const {}
void Memory::saveIfNeeded() const {}
void Memory::flushInternal() const {}
void Memory::load() {}
void Memory::save() const {}
void Memory::flush() const {}

Thoth::SQLiteMemoryRepository* Memory::getSQLiteRepo() const {
    return dynamic_cast<Thoth::SQLiteMemoryRepository*>(repo.get());
}

// These were missing but declared in header
void Memory::addMessages(const std::vector<std::pair<std::string, std::string>>& messages) {
    for (const auto& m : messages) addMessage(m.first, m.second);
}

void Memory::setSummary(const std::string& summary) {
    std::unique_lock lock(mtx);
    if (repo) repo->storeSummary(activeSessionId, "short", summary);
}

std::string Memory::getSummary(bool useExtended) const {
    std::shared_lock lock(mtx);
    if (!repo) return "";
    return repo->getSummary(activeSessionId, useExtended ? "extended" : "short");
}

void Memory::updateSummary(const std::string& goal, const std::string& response) {
    std::unique_lock lock(mtx);
    if (!repo) return;
    
    std::string newSummary = "Last Goal: " + goal + "\nLast Result: " + response;
    repo->storeSummary(activeSessionId, "short", newSummary);
}

bool Memory::setMeta(const std::string& key, const std::string& value) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->setMeta(key, value);
}

std::string Memory::getMeta(const std::string& key) const {
    std::shared_lock lock(mtx);
    if (!repo) return "";
    return repo->getMeta(key);
}

bool Memory::storePlanEmbedding(const std::string& planId, const std::string& type, const std::vector<float>& embedding) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->storePlanEmbedding(planId, type, embedding, 2);
}

std::vector<float> Memory::getPlanEmbedding(const std::string& planId, const std::string& type) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getPlanEmbedding(planId, type, 2);
}

bool Memory::clearPlanEmbeddings(int version) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->clearPlanEmbeddings(version);
}

void Memory::storePastPlan(const PastPlanRecord& plan) {
    std::unique_lock lock(mtx);
    if (repo) repo->storePastPlan(plan, 2);
}

std::vector<Memory::PastPlanRecord> Memory::getAllPastPlans() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getAllPastPlans(2);
}

void Memory::storeStepMetric(const Thoth::MemoryRepository::StepMetricRecord& metric) {
    std::unique_lock lock(mtx);
    if (repo) repo->storeStepMetric(metric);
}

void Memory::storeActivePlan(const Thoth::MemoryRepository::ActivePlanRecord& plan) {
    std::unique_lock lock(mtx);
    if (repo) repo->storeActivePlan(plan);
}

std::optional<Thoth::MemoryRepository::ActivePlanRecord> Memory::getActivePlan(const std::string& session_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->getActivePlan(session_id);
}

void Memory::deleteActivePlan(const std::string& plan_id) {
    std::unique_lock lock(mtx);
    if (repo) repo->deleteActivePlan(plan_id);
}

bool Memory::saveCognatePlan(const CognatePlanRecord& record) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->saveCognatePlan(record);
}

std::optional<Memory::CognatePlanRecord> Memory::loadCognatePlan(const std::string& plan_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->loadCognatePlan(plan_id);
}

std::vector<Memory::PastPlanRecord> Memory::retrieveSimilarPlans(
    const std::vector<float>& target_embedding,
    int limit) const {
    std::shared_lock lock(mtx);
    if (!repo || target_embedding.empty() || limit <= 0) return {};

    std::vector<std::pair<PastPlanRecord, float>> ranked;

    for (const auto& plan : repo->getAllPastPlans(2)) {
        if (plan.goal_embedding.empty()) continue;
        if (plan.success_score < Thoth::PlanReuse::kMinSuccessScore) continue;

        float score = GragScorer::cosine_similarity(target_embedding, plan.goal_embedding);
        if (plan.success_score >= Thoth::PlanReuse::kSuccessBoostThreshold) {
            score += Thoth::PlanReuse::kSuccessBoost;
        }
        ranked.push_back({plan, score});
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::vector<PastPlanRecord> results;
    const int count = std::min(static_cast<int>(ranked.size()), limit);
    results.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        if (ranked[static_cast<std::size_t>(i)].second < Thoth::PlanReuse::kMinSimilarityFloor) {
            break;
        }
        results.push_back(std::move(ranked[static_cast<std::size_t>(i)].first));
    }
    return results;
}

bool Memory::saveTrajectory(const CognateTrajectoryRecord& record) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    bool ok = repo->saveTrajectory(record);
    if (ok) {
        lock.unlock(); // Update index after release
        updateTrajectoryIndex(record);
    }
    return ok;
}

std::optional<Memory::CognateTrajectoryRecord> Memory::loadTrajectory(const std::string& trajectory_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->loadTrajectory(trajectory_id);
}

std::vector<Memory::CognateTrajectoryRecord> Memory::getAllTrajectories() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getAllTrajectories();
}

bool Memory::saveStrategy(const CognateStrategyRecord& record) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->saveStrategy(record);
}

std::optional<Memory::CognateStrategyRecord> Memory::loadStrategy(const std::string& strategy_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->loadStrategy(strategy_id);
}

std::vector<Memory::CognateStrategyRecord> Memory::getAllStrategies() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getAllStrategies();
}

bool Memory::saveExperiment(const CognateExperimentRecord& record) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->saveExperiment(record);
}

std::optional<Memory::CognateExperimentRecord> Memory::loadExperiment(const std::string& experiment_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->loadExperiment(experiment_id);
}

std::vector<Memory::CognateExperimentRecord> Memory::getAllExperiments() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getAllExperiments();
}

bool Memory::saveProblemState(const ProblemStateRecord& record) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->saveProblemState(record);
}

std::optional<Memory::ProblemStateRecord> Memory::loadProblemState(const std::string& problem_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->loadProblemState(problem_id);
}

std::optional<Memory::ProblemStateRecord> Memory::getLatestProblemState(const std::string& goal_id) const {
    std::shared_lock lock(mtx);
    if (!repo) return std::nullopt;
    return repo->getLatestProblemState(goal_id);
}
