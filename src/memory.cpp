#include "../include/memory.h"
#include "../include/grag_scorer.h"
#include "../include/sqlite_memory_repository.h"
#include <iostream>
#include <algorithm>
#include <ctime>

using json = nlohmann::json;

Memory::Memory(const Config& config) {
    repo = Thoth::MemoryRepositoryFactory::createRepository(config);
    if (repo) {
        pruner = std::make_unique<Thoth::MemoryPruner>(*repo);
    }
    load();
    loadTrajectoryIndex(); // Phase 7.7
    migrateEmbeddings(); // Step 2: Check for version migration on startup
}

Memory::~Memory() {
    save();
}

void Memory::markDirty() const {}
void Memory::saveIfNeeded() const { save(); }
void Memory::flushInternal() const { flush(); }

void Memory::load() {
    std::unique_lock lock(mtx);
    if (!repo) return;

    data = json::object();
    auto messages = repo->getMessages(activeSessionId);
    
    data["history"] = json::array();
    for (const auto& msg : messages) {
        data["history"].push_back({
            {"role", msg.role},
            {"content", msg.content},
            {"timestamp", msg.timestamp_ms}
        });
    }

    data["short_summary"] = repo->getSummary(activeSessionId, "short");
    data["extended_summary"] = repo->getSummary(activeSessionId, "extended");
}

void Memory::save() const {
    flush();
}

void Memory::flush() const {
    // With real-time persistence in addMessage, flush is mostly a no-op for history
}

void Memory::addMessage(const std::string& role, const std::string& content) {
    std::unique_lock lock(mtx);
    int64_t now = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    
    if (repo) {
        repo->createSession(activeSessionId, now);
        repo->appendMessage(activeSessionId, {role, content, now});
        
        if (pruner) {
            int archived = pruner->prune(activeSessionId);
            if (archived > 0) {
                // Refresh cache from DB if pruning happened
                lock.unlock();
                load();
                return;
            }
        }
    }

    data["history"].push_back({
        {"role", role},
        {"content", content},
        {"timestamp", now}
    });
}

void Memory::addMessages(const std::vector<std::pair<std::string, std::string>>& messages) {
    for (const auto& m : messages) {
        addMessage(m.first, m.second);
    }
}

std::vector<json> Memory::getConversation() const {
    std::shared_lock lock(mtx);
    std::vector<json> history;
    if (data.contains("history") && data["history"].is_array()) {
        for (const auto& msg : data["history"]) {
            history.push_back(msg);
        }
    }
    return history;
}

void Memory::clear() {
    std::unique_lock lock(mtx);
    if (repo) {
        repo->clearMessages(activeSessionId);
    }
    data["history"] = json::array();
}

void Memory::setSummary(const std::string& summary) {
    std::unique_lock lock(mtx);
    data["short_summary"] = summary;
    if (repo) {
        repo->storeSummary(activeSessionId, "short", summary);
    }
}

std::string Memory::getSummary(bool useExtended) const {
    std::shared_lock lock(mtx);
    if (useExtended) {
        return data.value("extended_summary", "");
    }
    return data.value("short_summary", "");
}

void Memory::updateSummary(const std::string& goal, const std::string& response) {
    std::unique_lock lock(mtx);
    std::string newSummary = "Last Goal: " + goal + "\nResult: " + response;
    data["short_summary"] = newSummary;
    if (repo) {
        repo->storeSummary(activeSessionId, "short", newSummary);
    }
}

bool Memory::setMeta(const std::string& key, const std::string& value) {
    std::unique_lock lock(mtx);
    if (repo) return repo->setMeta(key, value);
    return false;
}

std::string Memory::getMeta(const std::string& key) const {
    std::shared_lock lock(mtx);
    if (repo) return repo->getMeta(key);
    return "";
}

bool Memory::storePlanEmbedding(const std::string& planId, const std::string& type, const std::vector<float>& embedding) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    
    // Step 3 — Tag new embeddings on write:
    std::string versionStr = repo->getMeta("embedding_schema_version");
    int version = versionStr.empty() ? 1 : std::stoi(versionStr);
    
    return repo->storePlanEmbedding(planId, type, embedding, version);
}

std::vector<float> Memory::getPlanEmbedding(const std::string& planId, const std::string& type) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    
    std::string versionStr = repo->getMeta("embedding_schema_version");
    int version = versionStr.empty() ? 1 : std::stoi(versionStr);
    
    return repo->getPlanEmbedding(planId, type, version);
}

bool Memory::clearPlanEmbeddings(int version) {
    std::unique_lock lock(mtx);
    if (repo) return repo->clearPlanEmbeddings(version);
    return false;
}

void Memory::storePastPlan(const PastPlanRecord& plan) {
    std::unique_lock lock(mtx);
    if (!repo) return;

    std::string versionStr = repo->getMeta("embedding_schema_version");
    int version = versionStr.empty() ? 2 : std::stoi(versionStr);

    repo->storePastPlan(plan, version);
}
std::vector<Memory::PastPlanRecord> Memory::getAllPastPlans() const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    
    std::string versionStr = repo->getMeta("embedding_schema_version");
    int version = versionStr.empty() ? 2 : std::stoi(versionStr);

    return repo->getAllPastPlans(version);
}

void Memory::storeStepMetric(const Thoth::MemoryRepository::StepMetricRecord& metric) {
    std::unique_lock lock(mtx);
    if (repo) {
        repo->storeStepMetric(metric);
    }
}

void Memory::storeActivePlan(const Thoth::MemoryRepository::ActivePlanRecord& plan) {
    std::unique_lock lock(mtx);
    if (repo) repo->storeActivePlan(plan);
}

std::optional<Thoth::MemoryRepository::ActivePlanRecord> Memory::getActivePlan() const {
    std::shared_lock lock(mtx);
    if (repo) return repo->getActivePlan();
    return std::nullopt;
}

void Memory::deleteActivePlan(const std::string& plan_id) {
    std::unique_lock lock(mtx);
    if (repo) repo->deleteActivePlan(plan_id);
}

bool Memory::saveCognatePlan(const CognatePlanRecord& record) {
    std::unique_lock lock(mtx);
    if (repo) return repo->saveCognatePlan(record);
    return false;
}

std::optional<Memory::CognatePlanRecord> Memory::loadCognatePlan(const std::string& plan_id) const {
    std::shared_lock lock(mtx);
    if (repo) return repo->loadCognatePlan(plan_id);
    return std::nullopt;
}

std::vector<Memory::CognatePlanRecord> Memory::retrieveSimilarPlans(const std::vector<float>& target_embedding, int limit) const {
    std::shared_lock lock(mtx);
    if (!repo || target_embedding.empty()) return {};

    auto all_plans = repo->getAllCognatePlans();
    if (all_plans.empty()) return {};

    std::vector<std::pair<CognatePlanRecord, float>> scored_plans;
    for (auto& plan : all_plans) {
        if (plan.embedding.empty()) continue;

        float sim = GragScorer::cosine_similarity(target_embedding, plan.embedding);
        
        // Ranking factor: 70% similarity, 30% success score
        float score = (0.7f * sim) + (0.3f * plan.success_score);
        scored_plans.push_back({std::move(plan), score});
    }

    std::sort(scored_plans.begin(), scored_plans.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::vector<CognatePlanRecord> results;
    int count = std::min((int)scored_plans.size(), limit);
    for (int i = 0; i < count; ++i) {
        results.push_back(std::move(scored_plans[i].first));
    }

    return results;
}

bool Memory::saveTrajectory(const CognateTrajectoryRecord& record) {
    std::unique_lock lock(mtx);
    if (repo) {
        bool success = repo->saveTrajectory(record);
        if (success) {
            // Drop lock before calling update which will re-acquire
            lock.unlock();
            updateTrajectoryIndex(record);
            return true;
        }
        return false;
    }
    return false;
}

std::optional<Memory::CognateTrajectoryRecord> Memory::loadTrajectory(const std::string& trajectory_id) const {
    std::shared_lock lock(mtx);
    if (repo) return repo->loadTrajectory(trajectory_id);
    return std::nullopt;
}

std::vector<Memory::CognateTrajectoryRecord> Memory::getAllTrajectories() const {
    std::shared_lock lock(mtx);
    if (repo) return repo->getAllTrajectories();
    return {};
}

bool Memory::saveStrategy(const CognateStrategyRecord& record) {
    std::unique_lock lock(mtx);
    if (repo) return repo->saveStrategy(record);
    return false;
}

std::optional<Memory::CognateStrategyRecord> Memory::loadStrategy(const std::string& strategy_id) const {
    std::shared_lock lock(mtx);
    if (repo) return repo->loadStrategy(strategy_id);
    return std::nullopt;
}

std::vector<Memory::CognateStrategyRecord> Memory::getAllStrategies() const {
    std::shared_lock lock(mtx);
    if (repo) return repo->getAllStrategies();
    return {};
}

std::vector<Memory::CognateTrajectoryRecord> Memory::retrieveSimilarTrajectories(const std::vector<float>& target_embedding, int limit) const {
    std::shared_lock lock(mtx);
    if (!repo || target_embedding.empty()) return {};

    if (trajectory_index_.empty()) return {};

    // Ranking formula: retrieval_score = embedding_similarity + success_weight + recency_weight + usage_weight
    // We'll normalize these weights for Phase 7.7
    
    std::vector<std::pair<std::string, float>> ranked_ids;
    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

    for (const auto& entry : trajectory_index_) {
        if (entry.goal_embedding.empty()) continue;

        float sim = GragScorer::cosine_similarity(target_embedding, entry.goal_embedding);
        
        // Recency score (0 to 1, based on 7-day decay)
        float days_old = static_cast<float>(now - entry.created_at) / (1000.0f * 60 * 60 * 24);
        float recency = std::exp(-days_old / 7.0f); 

        // Usage score (logarithmic boost)
        float usage = std::log1p(static_cast<float>(entry.usage_count)) / 5.0f;

        // Combined score (weights can be tuned)
        float score = (0.5f * sim) + (0.2f * entry.success_score) + (0.2f * recency) + (0.1f * usage);
        
        // Phase 7.8: Tier-based filtering/penalty
        if (entry.tier == TIER_COLD) {
            score *= 0.5f; // Significant penalty for COLD memory
        } else if (entry.tier == TIER_WARM) {
            score *= 1.2f; // Slight boost for proven successful WARM memory
        }
        
        ranked_ids.push_back({entry.trajectory_id, score});
    }

    std::sort(ranked_ids.begin(), ranked_ids.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::vector<CognateTrajectoryRecord> results;
    int count = std::min((int)ranked_ids.size(), limit);
    for (int i = 0; i < count; ++i) {
        auto t = repo->loadTrajectory(ranked_ids[i].first);
        if (t) results.push_back(std::move(*t));
    }

    return results;
}

void Memory::processMemoryAging() {
    std::unique_lock lock(mtx);
    if (!repo) return;

    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
    
    constexpr int64_t ONE_DAY_MS = 1000LL * 60 * 60 * 24;

    bool index_changed = false;

    for (auto& entry : trajectory_index_) {
        int old_tier = entry.tier;
        int64_t age = now - entry.created_at;

        // 1. HOT: Recent trajectories (< 24h)
        if (age < ONE_DAY_MS) {
            entry.tier = TIER_HOT;
        }
        // 2. WARM: Proven successful (score >= 0.8) or frequently used
        else if (entry.success_score >= 0.8f || entry.usage_count > 5) {
            entry.tier = TIER_WARM;
        }
        // 3. COLD: Everything else
        else {
            entry.tier = TIER_COLD;
        }

        if (entry.tier != old_tier) {
            index_changed = true;
            // Sync to DB
            auto traj = repo->loadTrajectory(entry.trajectory_id);
            if (traj) {
                traj->tier = entry.tier;
                repo->saveTrajectory(*traj);
            }
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

bool Memory::addNode(const Node& node) {    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->addNode(node);
}

bool Memory::addEdge(const Edge& edge) {
    std::unique_lock lock(mtx);
    if (!repo) return false;
    return repo->addEdge(edge);
}

std::vector<Memory::Node> Memory::getNodesByType(const std::string& type) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getNodesByType(type);
}

std::vector<Thoth::MemoryRepository::Edge> Memory::getEdgesFrom(const std::string& nodeId) const {
    std::shared_lock lock(mtx);
    if (!repo) return {};
    return repo->getEdgesFrom(nodeId);
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

std::shared_ptr<Thoth::MemoryRepository> Memory::getRepo() const {
    // We can't return repo directly as it is a unique_ptr.
    // This is a design conflict in the requested change.
    // I will use a shared_ptr from a raw pointer with a custom no-op deleter
    // since Memory owns the lifecycle of repo.
    return std::shared_ptr<Thoth::MemoryRepository>(repo.get(), [](Thoth::MemoryRepository*){});
}

std::vector<Thoth::MemoryRepository::ArchivedTurnRecord> Memory::getArchivedTurns() const {
    std::shared_lock lock(mtx);
    if (!pruner) return {};
    return pruner->restore(activeSessionId);
}

Thoth::SQLiteMemoryRepository* Memory::getSQLiteRepo() const {
    return dynamic_cast<Thoth::SQLiteMemoryRepository*>(repo.get());
}

void Memory::migrateEmbeddings() {
    std::unique_lock lock(mtx);
    if (!repo) return;

    std::string versionStr = repo->getMeta("embedding_schema_version");
    if (versionStr == "1") {
        std::cout << "[Memory] Phase 4: Migrating embedding schema from v1 to v2...\n";
        
        // Step 2: Clear all stored goal/state embeddings from the DB
        repo->clearPlanEmbeddings(1);
        
        // Sets version to '2'
        repo->setMeta("embedding_schema_version", "2");
        
        std::cout << "[Memory] Migration complete. All v1 embeddings invalidated.\n";
    } else if (versionStr.empty()) {
        // Fresh DB or legacy without tag
        repo->setMeta("embedding_schema_version", "2");
    }
}

void Memory::printSummaries() const {
    std::shared_lock lock(mtx);
    std::cout << "Short Summary: " << data.value("short_summary", "(empty)") << "\n";
    std::cout << "Extended Summary: " << data.value("extended_summary", "(empty)") << "\n";
}
