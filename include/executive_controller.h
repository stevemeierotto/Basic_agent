/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <future>
#include "plan.h"
#include "trajectory.h"
#include "problem_state.h"
#include "iplanner.h"
#include "strategy_engine.h"
#include "iexecution_mode.h"
#include "controller_event.h"
#include "tools.h"
#include "rag.h"
#include "workflow_engine.h"
#include "standard_execution_mode.h"
#include "json.hpp"
#include "decision_trace.h"
#include "constraint_checker.h"
#include "trajectory_builder.h"
#include "reflection_utils.h"
#include "benchmark_environment.h"
#include "episode_events.h"
#include "episodic_learning_eval.h"

class LLMInterface;
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>

class Memory;
class LLMInterface;
class Config;

namespace Thoth {

class GraphRefiner;

enum class ControllerState {
    IDLE,
    PLANNING,
    EXECUTING_STEP,
    OBSERVING_RESULT,
    REVISING_PLAN,
    SCIENTIFIC_MODE,
    COMPLETED,
    ABORTED,
    FAILED
};

class ExecutiveController {
public:
    explicit ExecutiveController(
        std::shared_ptr<IPlanner> planner,
        std::shared_ptr<ToolRegistry> tool_registry,
        std::shared_ptr<RAGPipeline> rag,
        std::shared_ptr<Memory> memory
    );
    ~ExecutiveController();

    // Primary entry point — drives the full goal to completion
    std::string execute_goal(const std::string& goal,
                             const BenchmarkAttribution& benchmark = {},
                             const std::string& mtcp_task_id = {});

    /** E2 evaluation harness only — inject STRICT kernel context for RETRIEVAL dispatch (A4). */
    void set_e2_strict_eval_context(const SealedEpisodeInjectionLog* episode_log,
                                    const E2EvalConfig* eval_config);
    void clear_e2_strict_eval_context();

    // Pause/resume/abort support
    void pause();
    void resume();
    void abort();
    bool is_running() const;

    // Resume from a persisted plan (loaded from SQLite or JSON)
    void resume_from_plan(const Plan& plan);

    // Check for an interrupted plan in SQLite (Step 1.6)
    std::optional<Plan> get_resumable_plan() const;

    // Observability — subscribe to lifecycle events
    void set_event_callback(EventCallback callback);
    void set_session_id(const std::string& session_id) { 
        std::lock_guard<std::mutex> lock(mutex_);
        session_id_ = session_id; 
    }

    /** E2-C2: optional episode publication channel (execution domain only). */
    void set_episode_event_channel(Thoth::IEpisodeEventChannel* channel);

    // Mode switching (called internally, but exposed for testing)
    void set_execution_mode(std::unique_ptr<IExecutionMode> mode);
    void set_workflow_engine(std::shared_ptr<WorkflowEngine> engine) { 
        std::lock_guard<std::mutex> lock(mutex_);
        workflow_engine_ = engine; 
    }
    void set_llm_interface(LLMInterface* llm);
    void set_config(Config* cfg);

    /** C3: reflection replan limit (0 disables). Overrides default and env when set explicitly. */
    void set_max_reflections(int value);
    int get_max_reflections() const;
    int get_reflection_count() const;

    // State inspection (for UI highlighting and trace logging)
    ControllerState get_state() const;
    Plan get_current_plan() const;
    std::string get_current_plan_id() const;
    std::string get_session_id() const { 
        std::lock_guard<std::mutex> lock(mutex_); 
        return session_id_; 
    }
    int get_current_step_index() const;

    // Helpers exposed for ExecutionModes
    nlohmann::json dispatch_step(PlanStep& step);
    void transition_to(ControllerState new_state);
    void emit_event(EventType type, const std::string& step_id = "", const nlohmann::json& metadata = nlohmann::json::object());
    void handle_step_completion(const Thoth::StepResult& result);
    void decide_transition();
    
    // Wiring Requirement: Update embeddings
    void update_goal_embedding(const std::string& goal);
    void update_current_embedding();
    void update_trajectory_embedding();
    void clear_embeddings();

    // Problem State (Cognate V2)
    void update_problem_state(const ProblemState& state);
    ProblemState get_problem_state() const;

    // Getters for GRAG integration
    std::vector<float> get_goal_embedding() const { 
        std::lock_guard<std::mutex> lock(mutex_); 
        return goal_embedding_; 
    }
    std::vector<float> get_current_embedding() const { 
        std::lock_guard<std::mutex> lock(mutex_); 
        return current_embedding_; 
    }
    std::vector<float> get_trajectory_embedding() const { 
        std::lock_guard<std::mutex> lock(mutex_); 
        return trajectory_embedding_; 
    }

protected:
    friend class StandardExecutionMode;
    friend class ScientificExecutionMode;
    friend struct ExecutiveControllerP0TestAccess;
    std::string state_to_name(ControllerState state) const;
    void transition_to_unlocked(ControllerState new_state);
    void update_goal_embedding_unlocked(const std::string& goal);
    void update_current_embedding_unlocked();
    /** C7: single embedBatch call for goal G and current-state C vectors. */
    void refresh_goal_state_embeddings_unlocked();
    void update_trajectory_embedding_unlocked();
    void clear_embeddings_unlocked();
    nlohmann::json store_plan_history(float success_score);
    bool reinforce_plan_graph();
    bool commit_history_aging_unlocked();
    void continue_pending_reflection();
    void capture_working_trajectory_unlocked(Plan& plan) const;
    void bind_active_sets_to_plan_unlocked(const std::string& plan_id);
    void clear_owned_active_sets_unlocked();
    void sync_retrieval_sets_into_plan_unlocked();
    void adopt_resumed_retrieval_sets_unlocked();
    void record_trajectory_step(const PlanStep& step, const Thoth::StepResult& result);
    float calculate_trajectory_score(bool plan_completed_successfully);
    bool persist_current_plan_unlocked();
    bool commit_completed_step_unlocked(int episode_index);
    void backfill_durable_episode_steps_unlocked(const std::vector<int>& durable_indices);
    /** Resume-only. Terminal plans are left unchanged. Returns true when an
     *  orphaned RUNNING step on an unfinished plan must fail closed. */
    bool reconcileOrphanedRunningSteps_unlocked(std::vector<std::string>& requeued_step_ids,
                                                nlohmann::json& failure_metadata);
    void persist_problem_state_unlocked();

    // The main loop — runs until COMPLETED, ABORTED, or FAILED
    void run_loop();

    // Called each iteration — updates C embedding when GRAG is integrated
    void evaluate_state();

    void log_to_trace(const ControllerEvent& event);

    std::string build_plan_reuse_context(const std::vector<Memory::PastPlanRecord>& plans) const;
    nlohmann::json log_plan_reuse_injection(const std::vector<Memory::PastPlanRecord>& plans, const std::string& source);
    nlohmann::json log_plan_history_persisted(float success_score, const Memory::PastPlanRecord& past_record);
    nlohmann::json log_reflection_replan(float score, int reflection_cycle);

    void reset_goal_metrics_unlocked();
    void sync_goal_metrics_checkpoint_unlocked();
    void restore_goal_metrics_unlocked();
    void record_step_metrics_unlocked(const PlanStep& step, const StepResult& result);
    void sync_planning_tokens_unlocked();
    nlohmann::json goal_cognitive_metrics_payload_unlocked(const std::string& outcome,
                                                           float trajectory_score);
    bool flush_cognitive_metrics_outbox_unlocked();
    void emit_goal_cognitive_metrics_unlocked(const std::string& outcome, float trajectory_score);
    void publish_episode_completed_unlocked(bool goal_succeeded, float trajectory_score);

    int countActiveRetrievals_unlocked() const;
    const PlanStep* findStepById_unlocked(const std::string& step_id) const;
    PlanStep* findStepById_unlocked(const std::string& step_id);
    void attachEmbeddingSnapshot_unlocked(StepExecutionContext& ctx) const;
    bool isRetrievalPrefetchCandidate_unlocked(const PlanStep& step) const;
    bool isPrefetchStillValid_unlocked(const PlanStep& step) const;
    void invalidatePrefetchForStep_unlocked(const std::string& failed_step_id);
    void collectPrefetchResults_unlocked();
    int maxParallelRetrieval_unlocked() const;
    bool retrievalPrefetchEnabled_unlocked() const;

    std::shared_ptr<IPlanner> planner_;
    std::shared_ptr<ToolRegistry> tool_registry_;
    std::shared_ptr<RAGPipeline> rag_;
    std::shared_ptr<Memory> memory_;
    std::shared_ptr<WorkflowEngine> workflow_engine_;
    std::shared_ptr<StrategyEngine> strategy_engine_;
    std::shared_ptr<StepMetricsRepository> metrics_repo_;
    std::shared_ptr<TrajectoryBuilder> trajectory_builder_;
    std::shared_ptr<GraphRefiner> graph_refiner_;
    std::unique_ptr<IExecutionMode> execution_mode_;
    ConstraintChecker constraint_checker_;
    EventCallback event_callback_;
    LLMInterface* llm_interface_ = nullptr;
    std::string mtcp_task_id_;

    Plan current_plan_;
    Trajectory current_trajectory_;
    ProblemState current_problem_state_;
    
    // Phase 5.6: Track active chunks per step for causal linking
    struct ActiveStepSet {
        std::string step_id;
        std::vector<std::string> chunk_hashes;
    };
    std::vector<ActiveStepSet> active_sets_;
    // Plan id that owns active_sets_. Empty means the vector is not owned.
    std::string active_sets_plan_id_;
    // When false, persist must not replace the plan's retrieval_sets from memory.
    // Set for a consumed, terminal, or legacy finished plan whose sets must not
    // be rewritten from an empty in-memory list.
    bool retrieval_sets_memory_authoritative_ = true;

    struct RetrievalCaptureResult {
        ActiveStepSet set;
        bool unusable = false;
    };
    RetrievalCaptureResult capture_retrieval_set_unlocked(const std::string& step_id,
                                                          const nlohmann::json& result,
                                                          bool register_nodes);
    bool plan_has_unfinished_steps_unlocked() const;
    void restore_legacy_retrieval_sets_unlocked();
    void log_retrieval_set_diagnostic(const char* event, const std::string& detail) const;

    int revisions_count_ = 0;
    int reflection_count_ = 0;
    bool reflection_handoff_ = false;
    int max_reflections_ = Reflection::kMaxReflections;
    std::string reflection_skip_reason_;
    std::string session_id_;
    bool plan_reused_ = false;
    BenchmarkAttribution benchmark_attribution_;
    Thoth::IEpisodeEventChannel* episode_event_channel_ = nullptr;

    std::int64_t goal_started_at_ms_ = 0;
    std::string c64_window_id_;
    std::string c64_protocol_version_;
    std::string c64_metric_schema_version_;
    std::string c64_environment_schema_version_;
    std::string c64_cohort_fingerprint_;
    std::int64_t planning_time_ms_ = 0;
    std::int64_t retrieval_time_ms_ = 0;
    std::int64_t llm_synthesis_time_ms_ = 0;
    int retrieved_chunk_count_ = 0;
    int synthesis_prompt_chars_ = 0;
    bool synthesis_context_truncated_ = false;
    std::int64_t planning_tokens_ = 0;
    std::int64_t planning_tokens_at_resume_ = 0;
    std::int64_t token_base_prompt_ = 0;
    std::int64_t token_base_completion_ = 0;
    std::int64_t token_base_total_ = 0;
    bool metrics_checkpoint_gap_ = false;
    bool metrics_legacy_resume_ = false;
    bool cognitive_metrics_emitted_ = false;
    float last_grag_alpha_ = 0.0f;
    std::string last_grag_routing_mode_;
    float final_trajectory_score_ = 0.0f;

    ControllerState state_ = ControllerState::IDLE;
    bool paused_ = false;
    bool running_ = false;
    std::atomic<bool> stop_requested_{false};
    std::unique_ptr<std::thread> loop_thread_;
    mutable std::mutex mutex_;
    DecisionTraceLogger trace_logger_;
    std::vector<std::future<StepResult>> active_step_futures_;
    std::vector<std::string> active_step_ids_;

    Config* config_ = nullptr;
    std::unordered_map<std::string, StepResult> prefetch_cache_;
    std::vector<std::future<StepResult>> prefetch_futures_;
    std::vector<std::string> prefetch_step_ids_;
    int retrieval_prefetch_hits_ = 0;

    // GRAG embeddings
    std::vector<float> goal_embedding_;    // G

    // Updated after each STEP_COMPLETED. Represents where the plan
    // currently is in semantic space relative to the goal.
    // NOT updated on STEP_FAILED — state is unchanged on failure.
    std::vector<float> current_embedding_; // C

    // Phase 5.5: Trajectory embedding (T)
    std::vector<float> trajectory_embedding_;

    /** E2 A4: arm-scoped STRICT eval context (null outside eval harness). */
    const SealedEpisodeInjectionLog* e2_strict_episode_log_ = nullptr;
    const E2EvalConfig* e2_eval_config_ = nullptr;
};

} // namespace Thoth
