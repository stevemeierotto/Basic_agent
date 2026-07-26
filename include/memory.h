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
#include <unordered_set>
#include "memory_repository_factory.h"
#include "config.h"
#include "strategy.h"
#include "memory_pruner.h"
#include "memory_pruning_config.h"
#include "plan_reuse_config.h"
#include "consolidation_policy.h"
#include "consolidation_api.h"
#include "restore_api.h"
#include "clock.h"
#include <functional>

class LLMInterface;
class EmbeddingEngine;

namespace Thoth { class SQLiteMemoryRepository; class Clock; }

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

    // Session scoping (must be set before conversation writes)
    void setActiveSessionId(const std::string& sessionId);
    std::string getActiveSessionId() const;

    // Conversation
    struct TimedMessage {
        std::string role;
        std::string content;
        int64_t timestamp_ms = 0;
    };

    void addMessage(const std::string& role, const std::string& content);
    void addMessageWithTimestamp(const std::string& role,
                                 const std::string& content,
                                 int64_t timestamp_ms,
                                 bool triggerConsolidation = true);
    void addMessages(const std::vector<std::pair<std::string, std::string>>& messages);
    /** Replace hot tier; preserves timestamps; consolidates once at end. */
    void loadConversation(const std::vector<TimedMessage>& messages,
                          const std::string& summary = "");
    std::vector<json> getConversation() const;
    std::vector<TimedMessage> getTimedMessages(const std::string& sessionId) const;
    std::string getSummaryForSession(const std::string& sessionId,
                                     bool useExtended = false) const;
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
    std::optional<Thoth::MemoryRepository::ActivePlanRecord> getActivePlan(const std::string& session_id) const;
    void deleteActivePlan(const std::string& plan_id);

    // Cognate Plan Persistence (Phase 2.1)
    using CognatePlanRecord = Thoth::MemoryRepository::CognatePlanRecord;
    bool saveCognatePlan(const CognatePlanRecord& record);
    std::optional<CognatePlanRecord> loadCognatePlan(const std::string& plan_id) const;
    std::vector<PastPlanRecord> retrieveSimilarPlans(
        const std::vector<float>& target_embedding,
        int limit = Thoth::PlanReuse::kDefaultRetrieveLimit) const;

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

    // Cognate Experiments
    using CognateExperimentRecord = Thoth::MemoryRepository::CognateExperimentRecord;
    bool saveExperiment(const CognateExperimentRecord& record);
    std::optional<CognateExperimentRecord> loadExperiment(const std::string& experiment_id) const;
    std::vector<CognateExperimentRecord> getAllExperiments() const;

    // Problem State (Cognate V2, Phase 1.1)
    using ProblemStateRecord = Thoth::MemoryRepository::ProblemStateRecord;
    bool saveProblemState(const ProblemStateRecord& record);
    std::optional<ProblemStateRecord> loadProblemState(const std::string& problem_id) const;
    std::optional<ProblemStateRecord> getLatestProblemState(const std::string& goal_id) const;

    // Trajectory Index (Phase 7.7)
    void loadTrajectoryIndex();
    void updateTrajectoryIndex(const CognateTrajectoryRecord& record);
    std::vector<TrajectoryIndexEntry> getTrajectoryIndex() const;
    void processMemoryAging(); // Phase 7.8

    // Episode Steps (Phase 5, Step 5.5)
    using EpisodeStepRecord = Thoth::MemoryRepository::EpisodeStepRecord;
    void storeEpisodeStep(const EpisodeStepRecord& step);
    std::vector<EpisodeStepRecord> getRecentEpisodeSteps(const std::string& goal_id, int n) const;
    std::vector<EpisodeStepRecord> getAllEpisodeSteps() const;

    std::shared_ptr<Thoth::MemoryRepository> getRepo() const;

    // Tier Constants
    static constexpr int TIER_HOT = 0;
    static constexpr int TIER_WARM = 1;
    static constexpr int TIER_COLD = 2;

    // Graph Memory (Phase 5.6 / 8)
    using Node = Thoth::MemoryRepository::Node;
    using Edge = Thoth::MemoryRepository::Edge;
    using GraphStatistics = Thoth::MemoryRepository::GraphStatistics;
    void addNode(const Node& node);
    void addEdge(const Edge& edge);
    std::vector<Node> getNodesByType(const std::string& type) const;
    std::vector<Edge> getEdgesFrom(const std::string& nodeId) const;
    GraphStatistics getGraphStatistics() const;

    static std::string calculateContentHash(const std::string& text);


    // Pruning / consolidation (Phase 4, Step 4.2 / M2)
    void configureConsolidation(LLMInterface* llm,
                                EmbeddingEngine* embeddingEngine,
                                std::shared_ptr<Thoth::Clock> clock = nullptr);
    /** Discovery only — no LLM for inactive sessions; consolidates active if stale. */
    void runStartupConsolidationDiscovery();
    Thoth::ConsolidationDecision evaluateConsolidationPolicy(const std::string& sessionId) const;
    /** Immutable consolidation snapshot (M3). */
    Thoth::ConsolidationStatus getConsolidationStatus(const std::string& sessionId) const;
    /** Manual or automatic consolidation entry (M3). */
    Thoth::ConsolidationResult runConsolidation(const std::string& sessionId,
                                                const Thoth::ConsolidationRequest& request);
    /** M4 ranged restore (replay / rehydrate). */
    Thoth::RestoreResult runRestore(const std::string& sessionId,
                                    const Thoth::RestoreRequest& request);
    void setGoalActiveChecker(std::function<bool()> checker);
    bool isSessionMarkedStale(const std::string& sessionId) const;
    std::vector<Thoth::MemoryRepository::ArchivedTurnRecord> getArchivedTurns() const;
    std::vector<Thoth::MemoryRepository::WarmMemoryRecord> getRecentWarmMemory(int limit = 5) const;
    std::vector<Thoth::MemoryRepository::WarmMemoryRecord> searchWarmMemory(
        const std::vector<float>& queryEmbedding, int limit = 5) const;
    /** Goal execution: cosine search across all sessions (E2 / cross-session episodic recall). */
    std::vector<Thoth::MemoryRepository::WarmMemoryRecord> searchWarmMemoryAllSessions(
        const std::vector<float>& queryEmbedding, int limit = 5) const;

    // Fact Store Access (Phase 4, Step 4.3)
    Thoth::SQLiteMemoryRepository* getSQLiteRepo() const;

    // Debug helpers
    void printSummaries() const;

private:
    void migrateEmbeddings(); // Step 2 migration path
    void consolidateIfNeeded(const std::string& sessionId);
    void onSessionActivated(const std::string& sessionId);
    int64_t currentTimeMs() const;

    std::unique_ptr<Thoth::MemoryRepository> repo;
    std::unique_ptr<Thoth::MemoryPruner> pruner;
    const Config* config_ = nullptr;
    std::shared_ptr<Thoth::Clock> clock_;
    std::unordered_set<std::string> stale_session_ids_;
    std::function<bool()> goal_active_checker_;
    std::string activeSessionId = "default_session";
    std::string configPath;
    
    // In-memory cache to maintain existing performance for rapid reads
    json data;

    mutable std::shared_mutex mtx;     // protects data and repository access
    std::vector<TrajectoryIndexEntry> trajectory_index_; // Phase 7.7
};
