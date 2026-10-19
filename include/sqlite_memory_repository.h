#pragma once
#include "memory_repository.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

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
    std::vector<ArchivedTurnRecord> getArchivedMessages(
        const std::string& sessionId,
        const RestoreRange& range) override;
    RehydrateBatchResult rehydrateArchivedMessages(
        const std::string& sessionId,
        const RestoreRange& range) override;
    int getHotMessageCount(const std::string& sessionId) override;
    std::vector<MessageRecord> getOldestMessages(const std::string& sessionId, int count) override;
    std::optional<int64_t> getOldestHotMessageTimestamp(const std::string& sessionId) override;
    bool consolidateSessionBatch(const MemoryConsolidationRequest& request) override;
    std::vector<WarmMemoryRecord> getRecentWarmMemory(const std::string& sessionId, int limit) override;
    std::vector<WarmMemoryRecord> searchWarmMemoryByEmbedding(
        const std::string& sessionId,
        MemoryScope scope,
        const std::vector<float>& queryEmbedding,
        int limit,
        int embeddingVersion) override;

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

    /**
     * Test-only hook for the reliability harness.
     *
     * The callback is inactive unless a test installs one. It runs only after
     * an autocommit INSERT OR REPLACE into active_plans has reached
     * SQLITE_DONE and sqlite3_get_autocommit confirms the connection is no
     * longer inside a transaction, and only after sqlite3_finalize has
     * destroyed the statement. commitStepCheckpoint invokes it after COMMIT
     * returns and the connection is back in autocommit, so the active plan
     * and its episode row are durable before the callback. The callback is
     * not invoked when the statement fails, and it is not invoked while the
     * connection is still inside an explicit transaction.
     *
     * The repository mutex that guards this pointer is not held during the
     * callback. Memory::storeActivePlan or Memory::commitStepCheckpoint may
     * still hold its own mutex, so the callback must not re-enter Memory or
     * this SQLite connection. Exceptions
     * from the callback are caught and do not change the true/false result of
     * the committed write.
     */
    using ActivePlanCommitObserver = std::function<void(const ActivePlanRecord&)>;
    void setActivePlanCommitObserverForTests(ActivePlanCommitObserver observer);

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
    std::vector<EpisodeStepRecord> getAllEpisodeSteps() override;
    EpisodeIdentityReport inspectEpisodeIdentity() override;
    bool commitStepCheckpoint(const ActivePlanRecord& plan,
                              const std::optional<CognatePlanRecord>& cognate,
                              const std::optional<EpisodeStepRecord>& episode) override;
    bool enqueueCognitiveMetrics(const std::string& plan_id,
                                 const std::string& payload_json) override;
    std::vector<CognitiveMetricsOutboxRow> pendingCognitiveMetrics() override;
    bool markCognitiveMetricsFlushed(const std::string& metrics_id) override;

    // Cognate Experiments
    bool saveExperiment(const CognateExperimentRecord& record) override;
    std::optional<CognateExperimentRecord> loadExperiment(const std::string& experiment_id) override;
    std::vector<CognateExperimentRecord> getAllExperiments() override;

    // Problem State (Cognate V2, Phase 1.1)
    bool saveProblemState(const ProblemStateRecord& record) override;
    std::optional<ProblemStateRecord> loadProblemState(const std::string& problem_id) override;
    std::optional<ProblemStateRecord> getLatestProblemState(const std::string& goal_id) override;

private:
    struct DBHandle;
    std::unique_ptr<DBHandle> db_;
    std::vector<WarmMemoryRecord> getAllRecentWarmMemory(int limit);
    mutable std::mutex active_plan_observer_mu_;
    ActivePlanCommitObserver active_plan_observer_;
    // Guards transaction_held_ and transaction_owner_. Never hold this across
    // sqlite3_exec: the authorizer callback takes it.
    std::mutex transaction_mu_;
    bool transaction_held_ = false;
    std::thread::id transaction_owner_{};
    static int authorizeConnection(void* user, int action, const char* arg1, const char* arg2,
                                   const char* db_name, const char* trigger);
    void clearTransactionOwnership();
    void configureEpisodeIdentity();
    bool episodeObservationExists(const EpisodeStepRecord& step);
    bool observation_unique_ = false;
};

} // namespace Thoth
