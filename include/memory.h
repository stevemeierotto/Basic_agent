/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * Uses either Ollama local models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include <json.hpp>
#include <string>
#include <vector>
#include <shared_mutex>
#include <chrono>
#include <memory>
#include "memory_repository_factory.h"
#include "config.h"
#include "strategy.h"
#include "memory_pruner.h"

namespace Thoth { class SQLiteMemoryRepository; }

using json = nlohmann::json;

class Memory {
public:
    // Trajectory Index (Phase 7.7)
    struct TrajectoryIndexEntry {
        std::string trajectory_id;
        std::vector<float> goal_embedding;
        float success_score;
        int64_t created_at;
        int usage_count;
        int tier;
    };

    // Constructor / Destructor
    explicit Memory(const Config& config);
    ~Memory();

    void markDirty() const; // Deprecated, but keeping for interface compatibility
    void saveIfNeeded() const; // Deprecated
    void flushInternal() const; // Deprecated

    // Persistence

    void load();            // loads from disk (overwrites memory)
    void save() const;      // flushes to disk if dirty
    void flush() const;     // unconditional save, clears dirty flag

    // Conversation
    void addMessage(const std::string& role, const std::string& content);
    void addMessages(const std::vector<std::pair<std::string, std::string>>& messages);
    std::vector<json> getConversation() const;
    void clear();

    // Summaries
    void setSummary(const std::string& summary);  // sets short_summary only
    std::string getSummary(bool useExtended = false) const;
    void updateSummary(const std::string& goal, const std::string& response);

    // Metadata
    bool setMeta(const std::string& key, const std::string& value);
    std::string getMeta(const std::string& key) const;

    // Plan Embeddings (Goal/State)
    bool storePlanEmbedding(const std::string& planId, const std::string& type, const std::vector<float>& embedding);
    std::vector<float> getPlanEmbedding(const std::string& planId, const std::string& type) const;
    bool clearPlanEmbeddings(int version);

    // Plan History Reuse (Phase 5)
    using PastPlanRecord = Thoth::MemoryRepository::PastPlanRecord;
    void storePastPlan(const PastPlanRecord& plan);
    std::vector<PastPlanRecord> getAllPastPlans() const;

    void storeStepMetric(const Thoth::MemoryRepository::StepMetricRecord& metric);

    void storeActivePlan(const Thoth::MemoryRepository::ActivePlanRecord& plan);
    std::optional<Thoth::MemoryRepository::ActivePlanRecord> getActivePlan() const;
    void deleteActivePlan(const std::string& plan_id);

    // Cognate Plan Persistence (Phase 2.1)
    using CognatePlanRecord = Thoth::MemoryRepository::CognatePlanRecord;
    bool saveCognatePlan(const CognatePlanRecord& record);
    std::optional<CognatePlanRecord> loadCognatePlan(const std::string& plan_id) const;
    std::vector<CognatePlanRecord> retrieveSimilarPlans(const std::vector<float>& target_embedding, int limit) const;

    // Cognate Trajectory Persistence (Phase 7.2)
    using CognateTrajectoryRecord = Thoth::MemoryRepository::CognateTrajectoryRecord;
    bool saveTrajectory(const CognateTrajectoryRecord& record);
    std::optional<CognateTrajectoryRecord> loadTrajectory(const std::string& trajectory_id) const;
    std::vector<CognateTrajectoryRecord> getAllTrajectories() const;
    std::vector<CognateTrajectoryRecord> retrieveSimilarTrajectories(const std::vector<float>& target_embedding, int limit) const;

    // Cognate Strategy Extraction (Phase 8.2)
    using Strategy = Thoth::Strategy;
    using CognateStrategyRecord = Thoth::MemoryRepository::CognateStrategyRecord;
    bool saveStrategy(const CognateStrategyRecord& record);
    std::optional<CognateStrategyRecord> loadStrategy(const std::string& strategy_id) const;
    std::vector<CognateStrategyRecord> getAllStrategies() const;

    // Trajectory Index (Phase 7.7)
    void loadTrajectoryIndex();
    void updateTrajectoryIndex(const CognateTrajectoryRecord& record);
    std::vector<TrajectoryIndexEntry> getTrajectoryIndex() const;
    void processMemoryAging(); // Phase 7.8

    // Episode Steps (Phase 5, Step 5.5)
    using EpisodeStepRecord = Thoth::MemoryRepository::EpisodeStepRecord;
    void storeEpisodeStep(const EpisodeStepRecord& step);
    std::vector<EpisodeStepRecord> getRecentEpisodeSteps(const std::string& goal_id, int n) const;

    std::shared_ptr<Thoth::MemoryRepository> getRepo() const;

    // Tier Constants
    static constexpr int TIER_HOT = 0;
    static constexpr int TIER_WARM = 1;
    static constexpr int TIER_COLD = 2;

    // Graph Memory (Phase 8)
    using Node = Thoth::MemoryRepository::Node;
    using Edge = Thoth::MemoryRepository::Edge;
    bool addNode(const Node& node);
    bool addEdge(const Edge& edge);
    std::vector<Node> getNodesByType(const std::string& type) const;
    std::vector<Edge> getEdgesFrom(const std::string& nodeId) const;

    // Pruning (Phase 4, Step 4.2)
    std::vector<Thoth::MemoryRepository::ArchivedTurnRecord> getArchivedTurns() const;

    // Fact Store Access (Phase 4, Step 4.3)
    Thoth::SQLiteMemoryRepository* getSQLiteRepo() const;

    // Debug helpers
    void printSummaries() const;

private:
    void migrateEmbeddings(); // Step 2 migration path

    std::unique_ptr<Thoth::MemoryRepository> repo;
    std::unique_ptr<Thoth::MemoryPruner> pruner;
    std::string activeSessionId = "default_session";
    std::string configPath;
    
    // In-memory cache to maintain existing performance for rapid reads
    json data;

    mutable std::shared_mutex mtx;     // protects data and repository access
    std::vector<TrajectoryIndexEntry> trajectory_index_; // Phase 7.7
};
