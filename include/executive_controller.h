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
#include "graph_refiner.h"
#include <thread>
#include <atomic>
#include <mutex>

class Memory;

namespace Thoth {

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
    std::string execute_goal(const std::string& goal);

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

    // Mode switching (called internally, but exposed for testing)
    void set_execution_mode(std::unique_ptr<IExecutionMode> mode);
    void set_workflow_engine(std::shared_ptr<WorkflowEngine> engine) { 
        std::lock_guard<std::mutex> lock(mutex_);
        workflow_engine_ = engine; 
    }

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
    std::string state_to_name(ControllerState state) const;
    void transition_to_unlocked(ControllerState new_state);
    void update_goal_embedding_unlocked(const std::string& goal);
    void update_current_embedding_unlocked();
    void update_trajectory_embedding_unlocked();
    void clear_embeddings_unlocked();
    void store_plan_history(float success_score);
    void reinforce_plan_graph();
    void record_trajectory_step(const PlanStep& step, const Thoth::StepResult& result);
    float calculate_trajectory_score();
    void persist_current_plan_unlocked();
    void persist_problem_state_unlocked();

    // The main loop — runs until COMPLETED, ABORTED, or FAILED
    void run_loop();

    // Called each iteration — updates C embedding when GRAG is integrated
    void evaluate_state();

    void log_to_trace(const ControllerEvent& event);

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

    Plan current_plan_;
    Trajectory current_trajectory_;
    ProblemState current_problem_state_;
    
    // Phase 5.6: Track active chunks per step for causal linking
    struct ActiveStepSet {
        std::string step_id;
        std::vector<std::string> chunk_hashes;
    };
    std::vector<ActiveStepSet> active_sets_;
    int revisions_count_ = 0;
    int reflection_count_ = 0;
    const int MAX_REFLECTIONS = 2;
    std::string session_id_;
    bool plan_reused_ = false;
    ControllerState state_ = ControllerState::IDLE;
    bool paused_ = false;
    bool running_ = false;
    std::atomic<bool> stop_requested_{false};
    std::unique_ptr<std::thread> loop_thread_;
    mutable std::mutex mutex_;
    DecisionTraceLogger trace_logger_;
    std::vector<std::future<StepResult>> active_step_futures_;
    std::vector<std::string> active_step_ids_;

    // GRAG embeddings
    std::vector<float> goal_embedding_;    // G

    // Updated after each STEP_COMPLETED. Represents where the plan
    // currently is in semantic space relative to the goal.
    // NOT updated on STEP_FAILED — state is unchanged on failure.
    std::vector<float> current_embedding_; // C

    // Phase 5.5: Trajectory embedding (T)
    std::vector<float> trajectory_embedding_;
};

} // namespace Thoth
