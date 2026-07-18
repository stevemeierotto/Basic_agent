#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include "memory_scope.h"

namespace Thoth {

/** Timestamp range for cold-archive query / rehydrate (M4). */
struct RestoreRange {
    std::optional<int64_t> start_ms;
    std::optional<int64_t> end_ms;

    /** True when both bounds set and start_ms > end_ms. */
    bool isInvalid() const {
        return start_ms.has_value() && end_ms.has_value() && *start_ms > *end_ms;
    }
};

/** Outcome of a rehydrate batch transaction (M4). */
struct RehydrateBatchResult {
    int matched = 0;
    int restored = 0;
    int skipped_dup = 0;
    bool ok = true;
};

struct MessageRecord {
    std::string role;
    std::string content;
    int64_t timestamp_ms;
};

struct SessionRecord {
    std::string session_id;
    int64_t created_at_ms;
    int64_t updated_at_ms;
};

/**
 * @brief Pure virtual interface for memory persistence.
 * Models domain operations (sessions, messages, summaries) rather than storage mechanics.
 */
class MemoryRepository {
public:
    virtual ~MemoryRepository() = default;

    // Transaction Control
    virtual bool beginTransaction() = 0;
    virtual bool commit() = 0;
    virtual bool rollback() = 0;

    // Session Management
    virtual bool createSession(const std::string& sessionId, int64_t createdAtMs) = 0;
    virtual std::optional<SessionRecord> getSession(const std::string& sessionId) = 0;
    virtual std::vector<std::string> getAllSessionIds() = 0;

    // Message Management
    virtual bool appendMessage(const std::string& sessionId, const MessageRecord& msg) = 0;
    virtual std::vector<MessageRecord> getMessages(const std::string& sessionId) = 0;
    virtual bool clearMessages(const std::string& sessionId) = 0;

    // Pruning and Archival (Phase 4, Step 4.2)
    struct ArchivedTurnRecord {
        std::string archive_id;
        std::string session_id;
        int64_t original_timestamp_ms;
        std::string role;
        std::string content;
        std::string metadata_json;
        int64_t archived_at_ms;
        int summary_version;
    };

    virtual bool archiveMessages(const std::string& sessionId, int count, int summaryVersion) = 0;
    virtual std::vector<ArchivedTurnRecord> getArchivedMessages(const std::string& sessionId) = 0;
    /** Ranged cold query; empty bounds = full session. Stable order: ts ASC, archive_id ASC. */
    virtual std::vector<ArchivedTurnRecord> getArchivedMessages(
        const std::string& sessionId,
        const RestoreRange& range) = 0;
    /**
     * Copy matched cold turns into hot messages (transactional). Cold unchanged.
     * Duplicate invariant: (timestamp_ms, role, content).
     */
    virtual RehydrateBatchResult rehydrateArchivedMessages(
        const std::string& sessionId,
        const RestoreRange& range) = 0;
    virtual int getHotMessageCount(const std::string& sessionId) = 0;
    virtual std::vector<MessageRecord> getOldestMessages(const std::string& sessionId, int count) = 0;
    virtual std::optional<int64_t> getOldestHotMessageTimestamp(const std::string& sessionId) = 0;

    struct WarmMemoryRecord {
        std::string id;
        std::string session_id;
        MemoryScope scope = MemoryScope::SESSION;
        std::string episodic_payload;
        std::string rendered_summary;
        float importance = 0.5f;
        float novelty = 0.5f;
        float confidence = 1.0f;
        int covered_turn_start = 0;
        int covered_turn_end = 0;
        int64_t covered_ts_start = 0;
        int64_t covered_ts_end = 0;
        std::string parent_archive_ids_json;
        std::string derived_from_hash;
        int summary_version = 1;
        std::string prompt_version;
        std::string llm_model;
        bool summary_missing = false;
        int64_t created_at_ms = 0;
        std::vector<float> embedding;
        int embedding_version = 1;
    };

    struct MemoryConsolidationRequest {
        std::string session_id;
        std::vector<MessageRecord> messages_to_archive;
        std::optional<WarmMemoryRecord> warm;
    };

    /** Atomic: warm + embedding + archive + delete hot. Caller must embed before invoke. */
    virtual bool consolidateSessionBatch(const MemoryConsolidationRequest& request) = 0;
    virtual std::vector<WarmMemoryRecord> getRecentWarmMemory(const std::string& sessionId, int limit) = 0;
    virtual std::vector<WarmMemoryRecord> searchWarmMemoryByEmbedding(
        const std::string& sessionId,
        MemoryScope scope,
        const std::vector<float>& queryEmbedding,
        int limit,
        int embeddingVersion) = 0;

    // Structured Fact Store (Phase 4, Step 4.3)
    struct FactRecord {
        std::string key;
        std::string value;
        float confidence;
        std::string source;
        int64_t last_updated_ms;
    };

    virtual bool upsertFact(const FactRecord& fact) = 0;
    virtual std::optional<FactRecord> getFact(const std::string& key) = 0;
    virtual std::vector<FactRecord> searchFacts(const std::string& query) = 0;
    virtual bool deleteFact(const std::string& key) = 0;

    // Summary Management
    virtual bool storeSummary(const std::string& sessionId, const std::string& type, const std::string& content) = 0;
    virtual std::string getSummary(const std::string& sessionId, const std::string& type) = 0;

    // Metadata Management
    virtual bool setMeta(const std::string& key, const std::string& value) = 0;
    virtual std::string getMeta(const std::string& key) = 0;

    // Embedding Management (Goal/State)
    virtual bool storePlanEmbedding(const std::string& planId, const std::string& type, const std::vector<float>& embedding, int version) = 0;
    virtual std::vector<float> getPlanEmbedding(const std::string& planId, const std::string& type, int version) = 0;
    virtual bool clearPlanEmbeddings(int version) = 0;

    // Plan History Reuse (Phase 5)
    struct PastPlanRecord {
        std::string plan_id;
        std::string goal;
        std::string outline; // structured JSON
        float success_score;
        int64_t duration_ms;
        int failure_count;
        std::vector<float> goal_embedding;
    };

    virtual bool storePastPlan(const PastPlanRecord& plan, int version) = 0;
    virtual std::vector<PastPlanRecord> getAllPastPlans(int version) = 0;

    // Graph Memory (Phase 5.6)
    struct Node {
        std::string id;         // Content-based hash (SHA256)
        std::string file_path;
        std::string symbol;
        std::string type;
    };

    struct Edge {
        std::string from_id;
        std::string to_id;
        float weight;
        int success_count;
        int failure_count;
        int64_t last_used_ms;
    };

    virtual bool addNode(const Node& node) = 0;
    virtual bool addEdge(const Edge& edge) = 0;
    virtual std::vector<Node> getNodesByType(const std::string& type) = 0;
    virtual std::vector<Edge> getEdgesFrom(const std::string& nodeId) = 0;
    virtual std::vector<Edge> getAllEdges() = 0;
    virtual bool deleteEdge(const std::string& from_id, const std::string& to_id) = 0;
    
    // Graph Statistics (Adaptive Graph Memory)
    struct GraphStatistics {
        int total_nodes = 0;
        int total_edges = 0;
        float avg_edge_weight = 0.0f;
        float max_edge_weight = 0.0f;
        float min_edge_weight = 0.0f;
        int total_success_count = 0;
        int total_failure_count = 0;
    };
    virtual GraphStatistics getGraphStatistics() = 0;

    // Execution Metrics (Phase 1, Step 1.4)
    struct StepMetricRecord {
        std::string step_id;
        std::string plan_id;
        std::string tool_name;
        int64_t latency_ms;
        int retry_count;
        std::string status;
        int64_t timestamp_ms;
    };

    virtual bool storeStepMetric(const StepMetricRecord& metric) = 0;

    // Plan Persistence (Phase 1, Step 1.6)
    struct ActivePlanRecord {
        std::string plan_id;
        std::string session_id;
        std::string goal;
        std::string steps_json;
        int current_index;
        std::string controller_state;
        int64_t created_at_ms;
        int64_t updated_at_ms;
    };

    virtual bool storeActivePlan(const ActivePlanRecord& plan) = 0;
    virtual std::optional<ActivePlanRecord> getActivePlan(const std::string& session_id) = 0; // Filter by session
    virtual bool deleteActivePlan(const std::string& plan_id) = 0;

    // Cognate Plan Persistence (Phase 2.1)
    struct CognatePlanRecord {
        std::string plan_id;
        std::string goal;
        std::string plan_json;
        int status;
        float success_score;
        std::vector<float> embedding; // Phase 3.1
        int64_t created_at;
        int64_t updated_at;
    };

    virtual bool saveCognatePlan(const CognatePlanRecord& record) = 0;
    virtual std::optional<CognatePlanRecord> loadCognatePlan(const std::string& plan_id) = 0;
    virtual std::vector<CognatePlanRecord> getAllCognatePlans() = 0;

    // Cognate Trajectory Persistence (Phase 7.2)
    struct CognateTrajectoryRecord {
        std::string trajectory_id;
        std::string goal;
        std::string trajectory_json;
        float success_score;
        std::vector<float> embedding;
        int64_t created_at;
        int usage_count = 0; // Phase 7.7
        int tier = 0;        // Phase 7.7 (0: HOT, 1: WARM, 2: COLD)
    };

    virtual bool saveTrajectory(const CognateTrajectoryRecord& record) = 0;
    virtual std::optional<CognateTrajectoryRecord> loadTrajectory(const std::string& trajectory_id) = 0;
    virtual std::vector<CognateTrajectoryRecord> getAllTrajectories() = 0;

    // Cognate Strategy Extraction (Phase 8.2)
    struct CognateStrategyRecord {
        std::string strategy_id;
        std::string description;
        std::string step_pattern_json;
        float success_rate;
        int64_t created_at;
    };

    virtual bool saveStrategy(const CognateStrategyRecord& record) = 0;
    virtual std::optional<CognateStrategyRecord> loadStrategy(const std::string& strategy_id) = 0;
    virtual std::vector<CognateStrategyRecord> getAllStrategies() = 0;

    // Episode Steps (Phase 5, Step 5.5)
    struct EpisodeStepRecord {
        std::string episode_id;
        std::string goal_id;
        int step_index;
        std::string state_summary;
        std::string action_taken;
        std::string result_status;
        std::vector<float> embedding;
        int64_t timestamp_ms;
    };

    virtual bool storeEpisodeStep(const EpisodeStepRecord& step) = 0;
    virtual std::vector<EpisodeStepRecord> getRecentEpisodeSteps(const std::string& goal_id, int n) = 0;
    virtual std::vector<EpisodeStepRecord> getAllEpisodeSteps() = 0;

    // Cognate Experiments (Scientific Mode Integration)
    struct CognateExperimentRecord {
        std::string experiment_id;
        std::string name;
        std::string hypothesis;
        std::string configuration_json;
        std::string results_json;
        int64_t created_at;
        std::string status;
    };

    virtual bool saveExperiment(const CognateExperimentRecord& record) = 0;
    virtual std::optional<CognateExperimentRecord> loadExperiment(const std::string& experiment_id) = 0;
    virtual std::vector<CognateExperimentRecord> getAllExperiments() = 0;

    // Problem State (Cognate V2, Phase 1.1)
    struct ProblemStateRecord {
        std::string problem_id;
        std::string goal_id;
        std::string state_json; // Serialized ProblemState struct
        int iteration_count;
        float confidence_score;
        int64_t created_at;
        int64_t updated_at;
    };

    virtual bool saveProblemState(const ProblemStateRecord& record) = 0;
    virtual std::optional<ProblemStateRecord> loadProblemState(const std::string& problem_id) = 0;
    virtual std::optional<ProblemStateRecord> getLatestProblemState(const std::string& goal_id) = 0;
};

} // namespace Thoth
