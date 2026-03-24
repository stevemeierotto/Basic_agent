#pragma once
#include "memory_repository.h"
#include <memory>
#include <string>

namespace Thoth {

/**
 * @brief SQLite-backed implementation of episodic memory.
 * Uses PIMPL to keep SQLite headers out of the interface.
 */
class SQLiteMemoryRepository : public MemoryRepository {
public:
    explicit SQLiteMemoryRepository(const std::string& dbPath);
    ~SQLiteMemoryRepository() override;

    // Transaction Control
    bool beginTransaction() override;
    bool commit() override;
    bool rollback() override;

    // Session Management
    bool createSession(const std::string& sessionId, int64_t createdAtMs) override;
    std::optional<SessionRecord> getSession(const std::string& sessionId) override;
    std::vector<std::string> getAllSessionIds() override;

    // Message Management
    bool appendMessage(const std::string& sessionId, const MessageRecord& msg) override;
    std::vector<MessageRecord> getMessages(const std::string& sessionId) override;
    bool clearMessages(const std::string& sessionId) override;

    // Pruning and Archival
    bool archiveMessages(const std::string& sessionId, int count, int summaryVersion) override;
    std::vector<ArchivedTurnRecord> getArchivedMessages(const std::string& sessionId) override;
    int getHotMessageCount(const std::string& sessionId) override;

    // Structured Fact Store
    bool upsertFact(const FactRecord& fact) override;
    std::optional<FactRecord> getFact(const std::string& key) override;
    std::vector<FactRecord> searchFacts(const std::string& query) override;
    bool deleteFact(const std::string& key) override;

    // Summary Management
    bool storeSummary(const std::string& sessionId, const std::string& type, const std::string& content) override;
    std::string getSummary(const std::string& sessionId, const std::string& type) override;

    // Metadata Management
    bool setMeta(const std::string& key, const std::string& value) override;
    std::string getMeta(const std::string& key) override;

    // Embedding Management (Goal/State)
    bool storePlanEmbedding(const std::string& planId, const std::string& type, const std::vector<float>& embedding, int version) override;
    std::vector<float> getPlanEmbedding(const std::string& planId, const std::string& type, int version) override;
    bool clearPlanEmbeddings(int version) override;

    // Plan History Reuse (Phase 5)
    bool storePastPlan(const PastPlanRecord& plan, int version) override;
    std::vector<PastPlanRecord> getAllPastPlans(int version) override;

    // Graph Memory (Phase 8)
    bool addNode(const Node& node) override;
    bool addEdge(const Edge& edge) override;
    std::vector<Node> getNodesByType(const std::string& type) override;
    std::vector<Edge> getEdgesFrom(const std::string& nodeId) override;
    std::vector<Edge> getAllEdges() override;
    bool deleteEdge(const std::string& from_id, const std::string& to_id) override;
    GraphStatistics getGraphStatistics() override;

    virtual bool storeStepMetric(const StepMetricRecord& metric) override;

    bool storeActivePlan(const ActivePlanRecord& plan) override;
    std::optional<ActivePlanRecord> getActivePlan(const std::string& session_id) override;
    bool deleteActivePlan(const std::string& plan_id) override;

    // Cognate Phase 2.1
    bool saveCognatePlan(const CognatePlanRecord& record) override;
    std::optional<CognatePlanRecord> loadCognatePlan(const std::string& plan_id) override;
    std::vector<CognatePlanRecord> getAllCognatePlans() override;

    // Cognate Phase 7.2
    bool saveTrajectory(const CognateTrajectoryRecord& record) override;
    std::optional<CognateTrajectoryRecord> loadTrajectory(const std::string& trajectory_id) override;
    std::vector<CognateTrajectoryRecord> getAllTrajectories() override;

    // Cognate Phase 8.2
    bool saveStrategy(const CognateStrategyRecord& record) override;
    std::optional<CognateStrategyRecord> loadStrategy(const std::string& strategy_id) override;
    std::vector<CognateStrategyRecord> getAllStrategies() override;

    // Episode Steps (Phase 5, Step 5.5)
    bool storeEpisodeStep(const EpisodeStepRecord& step) override;
    std::vector<EpisodeStepRecord> getRecentEpisodeSteps(const std::string& goal_id, int n) override;

    // Cognate Experiments
    bool saveExperiment(const CognateExperimentRecord& record) override;
    std::optional<CognateExperimentRecord> loadExperiment(const std::string& experiment_id) override;
    std::vector<CognateExperimentRecord> getAllExperiments() override;

private:
    struct DBHandle;
    std::unique_ptr<DBHandle> db_;
};

} // namespace Thoth
