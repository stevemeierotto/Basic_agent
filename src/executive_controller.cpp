/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Implementation (Parallel Engine v1.0)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/executive_controller.h"
#include "../include/experience_provenance.h"
#include "../include/generation_call.h"
#include "../include/llm_interface.h"
#include "../include/logger.h"
#include "../include/grag_scorer.h"
#include "../include/memory.h"
#include "../include/plan_reuse_config.h"
#include "../include/grag_metrics.h"
#include "../include/step_metrics_repository.h"
#include "../include/file_handler.h"
#include "../include/goal_text_utils.h"
#include "../include/c64_window_attribution.h"
#include "../include/cognitive_metrics.h"
#include "../include/cm1_telemetry.h"
#include "../include/runtime_latency_config.h"
#include "../include/config.h"
#include "../include/reflection_utils.h"
#include "../include/llm_timeout_policy.h"
#include "../include/graph_refiner.h"
#include "../include/grag_diagnostics.h"
#include "../include/index_manager.h"
#include "../include/rag.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <csignal>
#include <filesystem>
#include <unistd.h>

namespace Thoth {

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static bool payloadNamesTool(const nlohmann::json& payload) {
    if (!payload.is_object()) {
        return false;
    }
    if (!payload.value("tool", std::string()).empty()) {
        return true;
    }
    return !payload.value("tool_name", std::string()).empty();
}

static bool isOrdinaryReadOnlyRetrieval(const PlanStep& step) {
    if (step.type != StepType::RETRIEVAL) {
        return false;
    }
    if (step.result.is_object() || step.result.is_array()) {
        if (!step.result.empty()) {
            return false;
        }
    } else if (step.result.is_string()) {
        if (!step.result.get<std::string>().empty()) {
            return false;
        }
    } else if (!step.result.is_null()) {
        return false;
    }
    if (!step.payload.is_null() && !step.payload.is_object()) {
        return false;
    }
    if (payloadNamesTool(step.payload)) {
        return false;
    }
    if (step.tool.is_object()) {
        if (!step.tool.empty()) {
            return false;
        }
    } else if (!step.tool.is_null()) {
        return false;
    }
    return true;
}

static bool stepIsInFlight(const std::string& step_id,
                    const std::vector<std::string>& active_ids,
                    const std::vector<std::string>& prefetch_ids) {
    return std::find(active_ids.begin(), active_ids.end(), step_id) != active_ids.end()
        || std::find(prefetch_ids.begin(), prefetch_ids.end(), step_id) != prefetch_ids.end();
}

static void rememberPriorRunning(PlanStep& step) {
    constexpr const char* kMarker = "[resume_prior_status=RUNNING]";
    if (step.reasoning.find(kMarker) == std::string::npos) {
        if (!step.reasoning.empty()) {
            step.reasoning += " ";
        }
        step.reasoning += kMarker;
    }
}

static bool planStatusIsTerminal(PlanStatus status) {
    return status == PlanStatus::COMPLETED
        || status == PlanStatus::FAILED
        || status == PlanStatus::ABORTED;
}

static std::optional<Memory::EpisodeStepRecord> episodeFromDurableStep(const Plan& plan, int index) {
    if (plan.plan_id.empty() || index < 0 ||
        static_cast<std::size_t>(index) >= plan.steps.size()) {
        return std::nullopt;
    }
    const PlanStep& step = plan.steps[static_cast<std::size_t>(index)];
    if (step.status != StepStatus::SUCCESS && step.status != StepStatus::FAILED) {
        return std::nullopt;
    }
    Memory::EpisodeStepRecord episode;
    episode.episode_id = "traj-" + plan.plan_id;
    episode.goal_id = plan.plan_id;
    episode.step_index = index;
    episode.state_summary = step.description;
    episode.action_taken = step.tool.dump();
    episode.result_status = step.status == StepStatus::SUCCESS ? "SUCCESS" : "FAILED";
    episode.timestamp_ms = step.completed_at_ms;
    return episode;
}

static void cognatePlanRecordFromPlan(const Plan& plan, MemoryRepository::CognatePlanRecord& record) {
    record.plan_id = plan.plan_id;
    record.goal = plan.goal;
    record.plan_json = plan.to_json().dump();
    record.status = static_cast<int>(plan.status);
    record.success_score = 0.0f;
    record.created_at = plan.created_at_ms;
    record.updated_at = plan.updated_at_ms;
}

static StepExecutionContext buildStepExecutionContext(const Plan& plan) {
    StepExecutionContext ctx;
    auto [cleanGoal, _] = splitPlanReuseInjection(plan.goal);
    ctx.goal = cleanGoal;
    for (const auto& step : plan.steps) {
        if (step.status != StepStatus::SUCCESS || step.result.is_null()) {
            continue;
        }
        ctx.prior_steps.push_back({
            step.step_id,
            static_cast<int>(step.type),
            step.description,
            step.result
        });
    }
    return ctx;
}

PlanStep* ExecutiveController::findStepById_unlocked(const std::string& step_id) {
    for (auto& step : current_plan_.steps) {
        if (step.step_id == step_id) {
            return &step;
        }
    }
    return nullptr;
}

const PlanStep* ExecutiveController::findStepById_unlocked(const std::string& step_id) const {
    for (const auto& step : current_plan_.steps) {
        if (step.step_id == step_id) {
            return &step;
        }
    }
    return nullptr;
}

void ExecutiveController::attachEmbeddingSnapshot_unlocked(StepExecutionContext& ctx) const {
    ctx.goal_embedding = goal_embedding_;
    ctx.current_embedding = current_embedding_;
    ctx.trajectory_embedding = trajectory_embedding_;
}

int ExecutiveController::maxParallelRetrieval_unlocked() const {
    if (config_ && config_->max_parallel_retrieval > 0) {
        return config_->max_parallel_retrieval;
    }
    return Thoth::RuntimeLatency::kDefaultMaxParallelRetrieval;
}

bool ExecutiveController::retrievalPrefetchEnabled_unlocked() const {
    if (config_) {
        return config_->enable_retrieval_prefetch;
    }
    return Thoth::RuntimeLatency::kDefaultEnableRetrievalPrefetch;
}

int ExecutiveController::countActiveRetrievals_unlocked() const {
    int count = 0;
    for (const auto& step_id : active_step_ids_) {
        if (const PlanStep* step = findStepById_unlocked(step_id)) {
            if (step->type == StepType::RETRIEVAL) {
                ++count;
            }
        }
    }
    count += static_cast<int>(prefetch_step_ids_.size());
    return count;
}

bool ExecutiveController::isRetrievalPrefetchCandidate_unlocked(const PlanStep& step) const {
    if (step.type != StepType::RETRIEVAL || step.status != StepStatus::PENDING) {
        return false;
    }
    if (prefetch_cache_.count(step.step_id) > 0) {
        return false;
    }
    if (std::find(active_step_ids_.begin(), active_step_ids_.end(), step.step_id) != active_step_ids_.end()) {
        return false;
    }
    if (std::find(prefetch_step_ids_.begin(), prefetch_step_ids_.end(), step.step_id) != prefetch_step_ids_.end()) {
        return false;
    }
    if (step.depends_on.empty()) {
        return false;
    }

    int running_unmet = 0;
    for (const auto& dep_id : step.depends_on) {
        const PlanStep* dep = findStepById_unlocked(dep_id);
        if (!dep) {
            continue;
        }
        if (dep->status == StepStatus::SUCCESS || dep->status == StepStatus::FAILED) {
            continue;
        }
        if (dep->status == StepStatus::RUNNING) {
            ++running_unmet;
            continue;
        }
        return false;
    }
    return running_unmet == 1;
}

bool ExecutiveController::isPrefetchStillValid_unlocked(const PlanStep& step) const {
    for (const auto& dep_id : step.depends_on) {
        const PlanStep* dep = findStepById_unlocked(dep_id);
        if (!dep) {
            continue;
        }
        if (dep->status == StepStatus::FAILED) {
            return false;
        }
    }
    return true;
}

void ExecutiveController::invalidatePrefetchForStep_unlocked(const std::string& failed_step_id) {
    for (auto it = prefetch_cache_.begin(); it != prefetch_cache_.end();) {
        const PlanStep* step = findStepById_unlocked(it->first);
        if (!step) {
            it = prefetch_cache_.erase(it);
            continue;
        }
        const auto& deps = step->depends_on;
        if (std::find(deps.begin(), deps.end(), failed_step_id) != deps.end()) {
            it = prefetch_cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void ExecutiveController::collectPrefetchResults_unlocked() {
    auto fut_it = prefetch_futures_.begin();
    auto id_it = prefetch_step_ids_.begin();

    while (fut_it != prefetch_futures_.end()) {
        if (fut_it->valid() && fut_it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            StepResult result;
            try {
                result = fut_it->get();
            } catch (const std::exception& e) {
                result.step_id = *id_it;
                result.success = false;
                result.error_message = std::string("Prefetch future exception: ") + e.what();
            }

            const PlanStep* step = findStepById_unlocked(result.step_id);
            if (step && isPrefetchStillValid_unlocked(*step)) {
                prefetch_cache_[result.step_id] = result;
            }

            fut_it = prefetch_futures_.erase(fut_it);
            id_it = prefetch_step_ids_.erase(id_it);
        } else {
            ++fut_it;
            ++id_it;
        }
    }
}

ExecutiveController::ExecutiveController(
    std::shared_ptr<IPlanner> planner,
    std::shared_ptr<ToolRegistry> tool_registry,
    std::shared_ptr<RAGPipeline> rag,
    std::shared_ptr<Memory> memory
) : planner_(planner), tool_registry_(tool_registry), rag_(rag), memory_(memory) {
    if (const char* envMax = std::getenv("THOTH_MAX_REFLECTIONS")) {
        try {
            max_reflections_ = std::max(0, std::stoi(envMax));
        } catch (...) {
        }
    }
    metrics_repo_ = std::make_shared<Thoth::StepMetricsRepository>("");
    workflow_engine_ = std::make_shared<Thoth::WorkflowEngine>(tool_registry, rag, memory, metrics_repo_);
    strategy_engine_ = std::make_shared<Thoth::StrategyEngine>(memory);
    if (rag && memory) {
        trajectory_builder_ = std::make_shared<Thoth::TrajectoryBuilder>(memory->getRepo(), rag->engine.get());
        graph_refiner_ = std::make_shared<Thoth::GraphRefiner>(memory);
    }
    execution_mode_ = std::make_unique<Thoth::StandardExecutionMode>();
    transition_to_unlocked(ControllerState::IDLE);
}

ExecutiveController::~ExecutiveController() {
    std::unique_ptr<std::thread> thread_to_join;
    std::vector<std::future<StepResult>> step_futures;
    std::vector<std::future<StepResult>> preload_futures;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
        thread_to_join = std::move(loop_thread_);
        step_futures = std::move(active_step_futures_);
        preload_futures = std::move(prefetch_futures_);
        active_step_ids_.clear();
        prefetch_step_ids_.clear();
    }
    if (thread_to_join && thread_to_join->joinable()) {
        thread_to_join->join();
    }
    for (auto& fut : step_futures) {
        if (fut.valid()) {
            try {
                fut.wait();
            } catch (...) {
            }
        }
    }
    for (auto& fut : preload_futures) {
        if (fut.valid()) {
            try {
                fut.wait();
            } catch (...) {
            }
        }
    }
}

void ExecutiveController::set_llm_interface(LLMInterface* llm) {
    std::lock_guard<std::mutex> lock(mutex_);
    llm_interface_ = llm;
    if (workflow_engine_) {
        workflow_engine_->setLLMInterface(llm);
    }
}

void ExecutiveController::set_config(Config* cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = cfg;
    if (workflow_engine_) {
        workflow_engine_->setConfig(cfg);
    }
}

void ExecutiveController::set_episode_event_channel(Thoth::IEpisodeEventChannel* channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    episode_event_channel_ = channel;
}

void ExecutiveController::publish_episode_completed_unlocked(bool goal_succeeded,
                                                             float trajectory_score) {
    if (!config_ || !config_->enable_episodic_evaluation_publication) {
        return;
    }
    if (!episode_event_channel_) {
        return;
    }
    Thoth::EpisodeCompleted event;
    event.plan_id = current_plan_.plan_id;
    event.goal = current_plan_.goal;
    event.terminal_state = goal_succeeded ? "COMPLETED" : "FAILED";
    event.final_success_score = trajectory_score;
    event.completed_at_ms = nowMs();
    event.run_id = benchmark_attribution_.run_id;
    event.env_hash = benchmark_attribution_.env_hash;
    event.plan_snapshot = current_plan_.to_json();
    event.trajectory_snapshot = current_trajectory_.to_json();
    try {
        episode_event_channel_->publish(event);
    } catch (...) {
        // Best-effort publication — must not affect execution outcome.
    }
}

void ExecutiveController::set_max_reflections(int value) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_reflections_ = std::max(0, value);
}

int ExecutiveController::get_max_reflections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_reflections_;
}

int ExecutiveController::get_reflection_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reflection_count_;
}

void ExecutiveController::set_e2_strict_eval_context(const SealedEpisodeInjectionLog* episode_log,
                                                   const E2EvalConfig* eval_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    e2_strict_episode_log_ = episode_log;
    e2_eval_config_ = eval_config;
    if (rag_) {
        rag_->setActiveE2EvalConfig(eval_config);
    }
}

void ExecutiveController::clear_e2_strict_eval_context() {
    std::lock_guard<std::mutex> lock(mutex_);
    e2_strict_episode_log_ = nullptr;
    e2_eval_config_ = nullptr;
    if (rag_) {
        rag_->setActiveE2EvalConfig(nullptr);
    }
}

/** Stamp the goal session onto planner logs for this thread only, then restore. */
class PlannerLogSessionGuard {
public:
    explicit PlannerLogSessionGuard(const std::string& sessionId) {
        auto& logger = StructuredLogger::instance();
        previousRequestId_ = logger.currentRequestId();
        previousSessionId_ = logger.currentSessionId();
        logger.setContext(previousRequestId_, sessionId);
    }

    ~PlannerLogSessionGuard() {
        StructuredLogger::instance().setContext(previousRequestId_, previousSessionId_);
    }

    PlannerLogSessionGuard(const PlannerLogSessionGuard&) = delete;
    PlannerLogSessionGuard& operator=(const PlannerLogSessionGuard&) = delete;

private:
    std::string previousRequestId_;
    std::string previousSessionId_;
};

std::string ExecutiveController::execute_goal(const std::string& goal,
                                              const BenchmarkAttribution& benchmark,
                                              const std::string& mtcp_task_id) {
    // Join any prior loop outside the lock (same pattern as ~ExecutiveController).
    // Never unlock while a std::lock_guard still owns the mutex.
    {
        std::unique_ptr<std::thread> thread_to_join;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (loop_thread_) {
                stop_requested_ = true;
                thread_to_join = std::move(loop_thread_);
            }
        }
        if (thread_to_join && thread_to_join->joinable()) {
            thread_to_join->join();
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = false;
        flush_cognitive_metrics_outbox_unlocked();
        revisions_count_ = 0;
        reflection_count_ = 0;
        plan_reused_ = false;
        benchmark_attribution_ = benchmark;
        mtcp_task_id_ = mtcp_task_id;
        reset_goal_metrics_unlocked();
        transition_to_unlocked(ControllerState::PLANNING);
        // Drop the previous goal's retrieval sets before this goal retrieves.
        active_sets_.clear();
        active_sets_plan_id_.clear();
        retrieval_sets_memory_authoritative_ = true;

        current_plan_ = Plan();
        current_plan_.updated_at_ms = nowMs();
        persist_current_plan_unlocked();
    }

    emit_event(EventType::STATE_CHANGED);

    nlohmann::json plan_reuse_meta;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        update_goal_embedding_unlocked(goal);

        // Phase 5: Plan History Reuse logic
        std::string enhanced_goal = goal;
        if (memory_ && rag_) {
            auto past_plans = memory_->retrieveSimilarPlans(goal_embedding_, Thoth::PlanReuse::kDefaultRetrieveLimit);
            if (!past_plans.empty()) {
                enhanced_goal = goal + build_plan_reuse_context(past_plans);
                plan_reused_ = true;
                plan_reuse_meta = log_plan_reuse_injection(past_plans, "execute_goal");
            }
        }

        const auto planStart = nowMs();
        {
            PlannerLogSessionGuard sessionGuard(session_id_);
            Thoth::GenerationCallContext scopeContext;
            scopeContext.task_id = mtcp_task_id_;
            scopeContext.session_id = session_id_;
            scopeContext.call_type = "plan";
            Thoth::GenerationCallScope scope(scopeContext);
            current_plan_ = planner_->create_plan(enhanced_goal);
        }
        planning_time_ms_ += nowMs() - planStart;
        sync_planning_tokens_unlocked();
        current_plan_.created_at_ms = nowMs();
        current_plan_.updated_at_ms = current_plan_.created_at_ms;
        bind_active_sets_to_plan_unlocked(current_plan_.plan_id);
        
        // Initialize Trajectory (Phase 7.3)
        current_trajectory_ = Trajectory();
        current_trajectory_.trajectory_id = "traj-" + current_plan_.plan_id;
        current_trajectory_.goal = goal;
        current_trajectory_.plan_initial = current_plan_;
        current_trajectory_.created_at = current_plan_.created_at_ms;
        current_trajectory_.embedding = goal_embedding_;

        refresh_goal_state_embeddings_unlocked();
        persist_current_plan_unlocked();

        state_ = ControllerState::IDLE;
        running_ = true;
    }

    // Announce the plan before the loop can emit STEP_STARTED. The Plan
    // surface creates its rows from PLAN_CREATED; a step event that arrives
    // first has nowhere to land.
    if (!plan_reuse_meta.is_null()) {
        emit_event(EventType::PLAN_REUSE_INJECTION, "", plan_reuse_meta);
    }
    emit_event(EventType::PLAN_CREATED);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        loop_thread_ = std::make_unique<std::thread>([this]() {
            run_loop();
        });
    }
    return "GOAL ACCEPTED: " + goal;
}

void ExecutiveController::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = true;
    current_plan_.updated_at_ms = nowMs();
    persist_current_plan_unlocked();
}

void ExecutiveController::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = false;
    current_plan_.updated_at_ms = nowMs();
    persist_current_plan_unlocked();
}

void ExecutiveController::abort() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != ControllerState::COMPLETED && state_ != ControllerState::FAILED && state_ != ControllerState::ABORTED) {
        transition_to_unlocked(ControllerState::ABORTED);
        current_plan_.status = PlanStatus::ABORTED;
        clear_owned_active_sets_unlocked();
        stop_requested_ = true;
        current_plan_.updated_at_ms = nowMs();
        emit_goal_cognitive_metrics_unlocked("aborted", final_trajectory_score_);
        persist_current_plan_unlocked();
        
        // Releasing lock to emit event (prevents deadlocks)
        lock.unlock();
        emit_event(EventType::PLAN_ABORTED);
    }
}

bool ExecutiveController::is_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

void ExecutiveController::resume_from_plan(const Plan& plan) {
    {
        std::unique_ptr<std::thread> thread_to_join;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (loop_thread_) {
                stop_requested_ = true;
                thread_to_join = std::move(loop_thread_);
            }
        }
        if (thread_to_join && thread_to_join->joinable()) {
            thread_to_join->join();
        }
    }

    nlohmann::json failureMetadata;
    std::vector<std::string> requeued;
    bool failClosed = false;
    bool startLoop = false;
    bool continueReflection = false;
    bool finishHistoryTail = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = false;
        flush_cognitive_metrics_outbox_unlocked();

        current_plan_ = plan;
        current_plan_.updated_at_ms = nowMs();
        benchmark_attribution_ = {};

        std::vector<int> durable_episode_steps;
        durable_episode_steps.reserve(current_plan_.steps.size());
        for (int index = 0; index < static_cast<int>(current_plan_.steps.size()); ++index) {
            const StepStatus status = current_plan_.steps[static_cast<std::size_t>(index)].status;
            if (status == StepStatus::SUCCESS || status == StepStatus::FAILED) {
                durable_episode_steps.push_back(index);
            }
        }

        // A stored terminal plan is left unchanged, including any orphaned
        // RUNNING step. Reconciliation applies only to an unfinished plan.
        if (!planStatusIsTerminal(current_plan_.status)) {
            failClosed = reconcileOrphanedRunningSteps_unlocked(requeued, failureMetadata);
        }

        update_goal_embedding_unlocked(current_plan_.goal);
        refresh_goal_state_embeddings_unlocked();
        if (current_plan_.working_trajectory_present) {
            current_trajectory_ = Trajectory::from_json(current_plan_.working_trajectory);
            revisions_count_ = current_plan_.working_revisions_count;
            current_plan_.working_trajectory_present = false;
            current_plan_.working_trajectory = nlohmann::json::object();
            current_plan_.working_revisions_count = 0;
        } else {
            current_trajectory_ = Trajectory();
            revisions_count_ = 0;
        }
        restore_goal_metrics_unlocked();
        backfill_durable_episode_steps_unlocked(durable_episode_steps);

        if (failClosed || current_plan_.status == PlanStatus::FAILED) {
            state_ = ControllerState::FAILED;
            running_ = false;
        } else if (current_plan_.status == PlanStatus::ABORTED) {
            state_ = ControllerState::ABORTED;
            running_ = false;
        } else if (current_plan_.status == PlanStatus::COMPLETED) {
            state_ = ControllerState::COMPLETED;
            running_ = false;
        } else if (current_plan_.reflection_pending) {
            state_ = ControllerState::IDLE;
            running_ = false;
            continueReflection = true;
        } else {
            state_ = ControllerState::IDLE;
            running_ = true;
            startLoop = true;
        }
        adopt_resumed_retrieval_sets_unlocked();
        persist_current_plan_unlocked();
        finishHistoryTail = !failClosed &&
            planStatusIsTerminal(current_plan_.status) &&
            current_plan_.graph_reinforced;
    }

    if (failClosed) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            bool all_successful = !current_plan_.steps.empty();
            for (const auto& step : current_plan_.steps) {
                if (step.status != StepStatus::SUCCESS) {
                    all_successful = false;
                    break;
                }
            }
            reflection_skip_reason_ = "interrupted_step_not_replayed";
            const float score = calculate_trajectory_score(all_successful);
            emit_goal_cognitive_metrics_unlocked("failed", score);
        }
        emit_event(EventType::PLAN_FAILED, "", failureMetadata);
        return;
    }
    for (const auto& stepId : requeued) {
        emit_event(EventType::STATE_CHANGED, stepId, {
            {"resume_reconciliation", "retrieval_requeued"},
            {"prior_status", "RUNNING"},
            {"success", true}
        });
    }
    if (continueReflection) {
        continue_pending_reflection();
        return;
    }
    if (startLoop) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stop_requested_ && !loop_thread_) {
            loop_thread_ = std::make_unique<std::thread>([this]() {
                run_loop();
            });
        }
    }
    if (finishHistoryTail) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!current_plan_.history_aging_done) {
            if (commit_history_aging_unlocked() && memory_) {
                memory_->deleteActivePlan(current_plan_.plan_id);
            }
        } else if (memory_) {
            memory_->deleteActivePlan(current_plan_.plan_id);
        }
    }
}

bool ExecutiveController::reconcileOrphanedRunningSteps_unlocked(std::vector<std::string>& requeued_step_ids,
                                                                nlohmann::json& failure_metadata) {
    if (planStatusIsTerminal(current_plan_.status)) {
        return false;
    }
    std::vector<PlanStep*> orphans;
    for (auto& step : current_plan_.steps) {
        if (step.status != StepStatus::RUNNING) {
            continue;
        }
        if (stepIsInFlight(step.step_id, active_step_ids_, prefetch_step_ids_)) {
            continue;
        }
        orphans.push_back(&step);
    }
    if (orphans.empty()) {
        return false;
    }

    bool unsafe = false;
    for (const PlanStep* step : orphans) {
        if (!isOrdinaryReadOnlyRetrieval(*step)) {
            unsafe = true;
            break;
        }
    }

    if (unsafe) {
        failure_metadata = {
            {"error", "interrupted_step_not_replayed"},
            {"prior_status", "RUNNING"},
            {"steps", nlohmann::json::array()}
        };
        for (PlanStep* step : orphans) {
            rememberPriorRunning(*step);
            step->result = {
                {"error", "interrupted_step_not_replayed"},
                {"prior_status", "RUNNING"},
                {"step_type", static_cast<int>(step->type)}
            };
            step->status = StepStatus::FAILED;
            step->completed_at_ms = nowMs();
            failure_metadata["steps"].push_back({
                {"step_id", step->step_id},
                {"step_type", static_cast<int>(step->type)},
                {"prior_status", "RUNNING"},
                {"error", "interrupted_step_not_replayed"}
            });
        }
        current_plan_.status = PlanStatus::FAILED;
        current_plan_.updated_at_ms = nowMs();
        return true;
    }

    for (PlanStep* step : orphans) {
        rememberPriorRunning(*step);
        step->status = StepStatus::PENDING;
        step->started_at_ms = 0;
        step->completed_at_ms = 0;
        requeued_step_ids.push_back(step->step_id);
    }
    current_plan_.updated_at_ms = nowMs();
    return false;
}

std::optional<Plan> ExecutiveController::get_resumable_plan() const {
    if (!memory_) return std::nullopt;
    
    std::string sid;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sid = session_id_;
    }
    
    auto rec = memory_->getActivePlan(sid);
    if (!rec) return std::nullopt;

    try {
        Plan plan = Plan::from_json(nlohmann::json::parse(rec->steps_json));
        plan.plan_id = rec->plan_id;
        plan.goal = rec->goal;
        plan.current_index = static_cast<size_t>(rec->current_index);
        plan.created_at_ms = rec->created_at_ms;
        plan.updated_at_ms = rec->updated_at_ms;
        return plan;
    } catch (...) {
        return std::nullopt;
    }
}

void ExecutiveController::set_event_callback(EventCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    event_callback_ = callback;
}

void ExecutiveController::set_execution_mode(std::unique_ptr<IExecutionMode> mode) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution_mode_ = std::move(mode);
        current_plan_.updated_at_ms = nowMs();
        persist_current_plan_unlocked();
        persist_problem_state_unlocked();
    }
    emit_event(EventType::MODE_SWITCHED);
}

void ExecutiveController::update_problem_state(const ProblemState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_problem_state_ = state;
    current_problem_state_.updated_at = nowMs();
    persist_problem_state_unlocked();
}

ProblemState ExecutiveController::get_problem_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_problem_state_;
}

void ExecutiveController::persist_problem_state_unlocked() {
    if (!memory_ || current_problem_state_.problem_id.empty()) return;

    MemoryRepository::ProblemStateRecord rec;
    rec.problem_id = current_problem_state_.problem_id;
    rec.goal_id = current_plan_.plan_id; // Goal ID usually maps to plan_id
    rec.state_json = current_problem_state_.to_json().dump();
    rec.iteration_count = current_problem_state_.iteration_count;
    rec.confidence_score = current_problem_state_.confidence_score;
    rec.created_at = current_problem_state_.created_at;
    rec.updated_at = current_problem_state_.updated_at;
    
    memory_->saveProblemState(rec);
}

ControllerState ExecutiveController::get_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

Plan ExecutiveController::get_current_plan() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_plan_;
}

std::string ExecutiveController::get_current_plan_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_plan_.plan_id;
}

int ExecutiveController::get_current_step_index() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(current_plan_.current_index);
}

void ExecutiveController::run_loop() {
    while (!stop_requested_) {
        bool should_pause = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ == ControllerState::COMPLETED || 
                state_ == ControllerState::ABORTED || 
                state_ == ControllerState::FAILED) {
                break;
            }
            should_pause = paused_;
        }

        if (should_pause) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        evaluate_state();
        
        if (execution_mode_) {
            execution_mode_->execute_step(*this);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
}

void ExecutiveController::evaluate_state() {
    std::vector<StepResult> completed_results;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        collectPrefetchResults_unlocked();

        // Check for completed futures
        auto fut_it = active_step_futures_.begin();
        auto id_it = active_step_ids_.begin();

        while (fut_it != active_step_futures_.end()) {
            if (fut_it->valid() && fut_it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    completed_results.push_back(fut_it->get());
                } catch (const std::exception& e) {
                    StepResult err;
                    err.step_id = *id_it;
                    err.success = false;
                    err.error_message = std::string("Future exception: ") + e.what();
                    completed_results.push_back(err);
                }
                
                fut_it = active_step_futures_.erase(fut_it);
                id_it = active_step_ids_.erase(id_it);
            } else {
                ++fut_it;
                ++id_it;
            }
        }

        if (active_step_futures_.empty() && (state_ == ControllerState::EXECUTING_STEP || state_ == ControllerState::OBSERVING_RESULT)) {
            transition_to_unlocked(ControllerState::IDLE);
        }
    }

    for (const auto& res : completed_results) {
        handle_step_completion(res);
    }
}

void ExecutiveController::decide_transition() {
    std::vector<PlanStep> steps_to_start;
    std::vector<PlanStep> steps_to_prefetch;
    std::vector<std::pair<std::string, StepResult>> prefetched_completions;
    std::string plan_id_for_dispatch;
    StepExecutionContext execution_context;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (state_ == ControllerState::REVISING_PLAN ||
            state_ == ControllerState::PLANNING ||
            state_ == ControllerState::COMPLETED ||
            state_ == ControllerState::FAILED ||
            state_ == ControllerState::ABORTED) {
            return;
        }

        // 1. Check for Plan Completion
        bool any_unfinished = false;
        bool all_successful = true;
        for (const auto& s : current_plan_.steps) {
            if (s.status == StepStatus::PENDING || s.status == StepStatus::RUNNING) {
                any_unfinished = true;
                break;
            }
            if (s.status != StepStatus::SUCCESS) all_successful = false;
        }

        if (!any_unfinished && active_step_futures_.empty()) {
            if (current_plan_.reflection_pending) {
                lock.unlock();
                continue_pending_reflection();
                return;
            }

            float score = calculate_trajectory_score(all_successful);
            const bool timeout_failure = trajectoryHasTimeoutFailure(current_trajectory_);
            reflection_skip_reason_.clear();

            const bool reflectionEnabled = max_reflections_ > 0;
            const std::string skipReason = reflectionSkipReason(
                reflectionEnabled,
                reflection_count_,
                max_reflections_,
                score,
                Reflection::kScoreThreshold,
                timeout_failure);

            if (score < Reflection::kScoreThreshold && reflectionEnabled && skipReason.empty()) {
                std::cout << "[DEBUG] Low success score (" << score << "), triggering reflection cycle "
                          << (reflection_count_ + 1) << "\n";
                // Penalty polarity for the attempt being stored. The durable row
                // stays ACTIVE with reflection_pending until the replacement exists.
                current_plan_.status = PlanStatus::FAILED;
                reflection_handoff_ = true;
                store_plan_history(score);
                reflection_handoff_ = false;
                if (!current_plan_.graph_reinforced) {
                    current_plan_.status = PlanStatus::ACTIVE;
                    return;
                }
                auto reflection_meta = log_reflection_replan(score, reflection_count_);
                lock.unlock();
                emit_event(EventType::REFLECTION_REPLAN, "", reflection_meta);
                continue_pending_reflection();
                return;
            }

            if (!skipReason.empty() && score < Reflection::kScoreThreshold) {
                reflection_skip_reason_ = skipReason;
            }

            const bool goalSucceeded = all_successful && !current_plan_.steps.empty();
            current_plan_.status = goalSucceeded ? PlanStatus::COMPLETED : PlanStatus::FAILED;
            current_plan_.updated_at_ms = nowMs();
            final_trajectory_score_ = score;
            auto history_meta = store_plan_history(score);
            if (memory_ && history_meta.is_null()) {
                if (!current_plan_.graph_reinforced) {
                    current_plan_.status = PlanStatus::ACTIVE;
                }
                return;
            }

            final_trajectory_score_ = score;
            transition_to_unlocked(goalSucceeded ? ControllerState::COMPLETED : ControllerState::FAILED);
            emit_goal_cognitive_metrics_unlocked(goalSucceeded ? "completed" : "failed", score);
            publish_episode_completed_unlocked(goalSucceeded, score);
            if (memory_) memory_->deleteActivePlan(current_plan_.plan_id);

            lock.unlock();
            if (!history_meta.is_null()) {
                emit_event(EventType::PLAN_HISTORY_STORED, "", history_meta);
            }
            emit_event(goalSucceeded ? EventType::PLAN_COMPLETED : EventType::PLAN_FAILED);
            return;
        }

        // 2. Identify ready steps
        for (auto& step : current_plan_.steps) {
            if (step.status != StepStatus::PENDING) continue;

            // Check if this step is already being executed
            auto active_it = std::find(active_step_ids_.begin(), active_step_ids_.end(), step.step_id);
            if (active_it != active_step_ids_.end()) continue;

            bool deps_met = true;
            for (const auto& dep_id : step.depends_on) {
                bool found_dep = false;
                for (const auto& other : current_plan_.steps) {
                    if (other.step_id == dep_id) {
                        found_dep = true;
                        if (other.status != StepStatus::SUCCESS && other.status != StepStatus::FAILED) {
                            deps_met = false;
                        }
                        break;
                    }
                }
                if (found_dep && !deps_met) break;
            }

            if (deps_met) {
                if (prefetch_cache_.count(step.step_id) > 0) {
                    step.status = StepStatus::RUNNING;
                    step.started_at_ms = nowMs();
                    prefetched_completions.push_back({step.step_id, prefetch_cache_.at(step.step_id)});
                    prefetch_cache_.erase(step.step_id);
                    ++retrieval_prefetch_hits_;
                    continue;
                }

                if (step.type == StepType::RETRIEVAL &&
                    countActiveRetrievals_unlocked() >= maxParallelRetrieval_unlocked()) {
                    continue;
                }

                // Step 6.2: Global Constraint Check
                std::string action_type = "unknown";
                nlohmann::json check_payload = step.payload;
                
                if (step.type == StepType::TOOL) {
                    action_type = "tool_call";
                    check_payload["tool_name"] = step.payload.value("tool", "unknown");
                } else if (step.type == StepType::RETRIEVAL) {
                    action_type = "file_read";
                    check_payload["file_path"] = step.payload.value("query", "");
                } else if (step.type == StepType::NODE) {
                    action_type = "node_execution";
                }

                auto check_result = constraint_checker_.check_action(action_type, check_payload);
                if (!check_result.allowed) {
                    std::cout << "[DEBUG] Step " << step.step_id << " BLOCKED by constraint: " << check_result.reason << "\n";
                    step.status = StepStatus::FAILED;
                    step.result = {{"status", "error"}, {"error_message", "Action blocked by security policy: " + check_result.reason}};
                    Thoth::StepResult blocked;
                    blocked.step_id = step.step_id;
                    blocked.success = false;
                    blocked.error_message = step.result.value("error_message", "");
                    blocked.data = step.result;
                    record_trajectory_step(step, blocked);
                    persist_current_plan_unlocked();
                    continue; // Check next step
                }

                step.status = StepStatus::RUNNING;
                step.started_at_ms = nowMs();
                steps_to_start.push_back(step);
            }
        }

        if (retrievalPrefetchEnabled_unlocked()) {
            for (auto& step : current_plan_.steps) {
                if (!isRetrievalPrefetchCandidate_unlocked(step)) {
                    continue;
                }
                if (countActiveRetrievals_unlocked() >= maxParallelRetrieval_unlocked()) {
                    break;
                }

                nlohmann::json check_payload = step.payload;
                auto check_result = constraint_checker_.check_action(
                    "file_read", {{"file_path", check_payload.value("query", "")}});
                if (!check_result.allowed) {
                    continue;
                }

                steps_to_prefetch.push_back(step);
            }
        }

        if (!steps_to_start.empty()) {
            transition_to_unlocked(ControllerState::EXECUTING_STEP);
            current_plan_.updated_at_ms = nowMs();
            persist_current_plan_unlocked();
        }

        plan_id_for_dispatch = current_plan_.plan_id;
        execution_context = buildStepExecutionContext(current_plan_);
        attachEmbeddingSnapshot_unlocked(execution_context);
        execution_context.e2_strict_episode_log = e2_strict_episode_log_;
        execution_context.e2_eval_config = e2_eval_config_;
        execution_context.task_id = mtcp_task_id_;
        execution_context.session_id = session_id_;
    }

    for (const auto& [step_id, cached] : prefetched_completions) {
        emit_event(EventType::STEP_STARTED, step_id);
        handle_step_completion(cached);
    }

    for (const auto& step : steps_to_start) {
        emit_event(EventType::STEP_STARTED, step.step_id);

        if (workflow_engine_) {
            auto fut = workflow_engine_->executeStepAsync(step, plan_id_for_dispatch, execution_context);
            std::lock_guard<std::mutex> lock(mutex_);
            active_step_futures_.push_back(std::move(fut));
            active_step_ids_.push_back(step.step_id);
        }
    }

    for (const auto& step : steps_to_prefetch) {
        if (workflow_engine_) {
            auto fut = workflow_engine_->executeStepAsync(step, plan_id_for_dispatch, execution_context);
            std::lock_guard<std::mutex> lock(mutex_);
            prefetch_futures_.push_back(std::move(fut));
            prefetch_step_ids_.push_back(step.step_id);
        }
    }
}

void ExecutiveController::handle_step_completion(const Thoth::StepResult& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Find the step by ID
    auto it = std::find_if(current_plan_.steps.begin(), current_plan_.steps.end(), [&](const auto& s) {
        return s.step_id == result.step_id;
    });

    if (it == current_plan_.steps.end()) return;

    PlanStep& step = *it;
    step.result = result.data;
    step.status = result.success ? StepStatus::SUCCESS : StepStatus::FAILED;
    step.outcome.run_block_reason = result.run_block_reason;
    step.completed_at_ms = nowMs();
    current_plan_.updated_at_ms = step.completed_at_ms;

    record_step_metrics_unlocked(step, result);

    // Record Step in Trajectory (Phase 7.3)
    record_trajectory_step(step, result);

    const int episodeIndex = static_cast<int>(std::distance(current_plan_.steps.begin(), it));
    const auto storeAbortedEpisode = [&]() {
        if (!memory_) return;
        if (const auto episode = episodeFromDurableStep(current_plan_, episodeIndex)) {
            memory_->storeEpisodeStep(*episode);
        }
    };

    transition_to_unlocked(ControllerState::OBSERVING_RESULT);

    bool success = result.success;

    if (success) {
        current_plan_.updated_at_ms = nowMs();
        lock.unlock(); 
        emit_event(EventType::STEP_COMPLETED, step.step_id);
        
        lock.lock();
        update_current_embedding_unlocked();
        
        // Phase 5.6: Capture Active Set from RETRIEVAL result
        if (step.type == StepType::RETRIEVAL) {
            const RetrievalCaptureResult captured = capture_retrieval_set_unlocked(
                step.step_id, step.result, true);
            if (!captured.set.chunk_hashes.empty()) {
                if (active_sets_plan_id_.empty()) {
                    bind_active_sets_to_plan_unlocked(current_plan_.plan_id);
                }
                active_sets_.push_back(captured.set);
            }
        }

        update_trajectory_embedding_unlocked();
        commit_completed_step_unlocked(episodeIndex);
        
        transition_to_unlocked(ControllerState::IDLE);
    } else {
        current_plan_.updated_at_ms = nowMs();
        std::string err = result.error_message;
        std::string next_action = "continue";
        if (step.failure_policy.revise_plan_on_failure) {
            next_action = "revise";
        } else if (step.failure_policy.abort_on_failure) {
            next_action = "abort";
        }
        nlohmann::json fail_meta = {{"error", err}, {"next_action", next_action}};
        fail_meta["timeout_ms"] = LlmTimeoutPolicy::stepTimeoutMs(
            step.type, step.failure_policy.timeout_ms);
        if (result.data.is_object() && result.data.contains("timeout_ms")) {
            fail_meta["timeout_ms"] = result.data["timeout_ms"];
        }
        lock.unlock();
        emit_event(EventType::STEP_FAILED, step.step_id, fail_meta);

        lock.lock();
        invalidatePrefetchForStep_unlocked(step.step_id);
        if (step.failure_policy.revise_plan_on_failure) {
            transition_to_unlocked(ControllerState::REVISING_PLAN);
            current_plan_.updated_at_ms = nowMs();
            commit_completed_step_unlocked(episodeIndex);
            
            const std::string revision_task = mtcp_task_id_;
            const std::string revision_session = session_id_;
            const std::string revision_plan = current_plan_.plan_id;
            lock.unlock();
            Thoth::GenerationCallContext scopeContext;
            scopeContext.task_id = revision_task;
            scopeContext.session_id = revision_session;
            scopeContext.plan_id = revision_plan;
            scopeContext.call_type = "revision";
            Thoth::GenerationCallScope scope(scopeContext);
            Plan revised = planner_->revise_plan(current_plan_, result.data);
            
            lock.lock();
            revisions_count_++;
            current_plan_ = revised;
            current_plan_.updated_at_ms = nowMs();
            persist_current_plan_unlocked();
            lock.unlock();
            emit_event(EventType::PLAN_REVISED);
            
            lock.lock();
            transition_to_unlocked(ControllerState::IDLE);
            current_plan_.updated_at_ms = nowMs();
            persist_current_plan_unlocked();
        } else if (step.failure_policy.abort_on_failure) {
            // This path does not store history or reinforce. Drop this goal's sets
            // so they cannot join the next goal. Status matches ControllerState::FAILED.
            current_plan_.status = PlanStatus::FAILED;
            active_sets_.clear();
            active_sets_plan_id_.clear();
            retrieval_sets_memory_authoritative_ = true;
            transition_to_unlocked(ControllerState::FAILED);
            current_plan_.updated_at_ms = nowMs();
            storeAbortedEpisode();
            emit_goal_cognitive_metrics_unlocked("failed", 0.0f);
            if (memory_) memory_->deleteActivePlan(current_plan_.plan_id);
            lock.unlock();
            emit_event(EventType::PLAN_FAILED);
        } else {
            transition_to_unlocked(ControllerState::IDLE);
            commit_completed_step_unlocked(episodeIndex);
        }
    }
}

nlohmann::json ExecutiveController::dispatch_step(PlanStep& step) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!workflow_engine_) {
        return {{"status", "error"}, {"error_message", "WorkflowEngine not available"}};
    }
    const StepExecutionContext context = [&]() {
        StepExecutionContext ctx = buildStepExecutionContext(current_plan_);
        attachEmbeddingSnapshot_unlocked(ctx);
        ctx.e2_strict_episode_log = e2_strict_episode_log_;
        ctx.e2_eval_config = e2_eval_config_;
        ctx.task_id = mtcp_task_id_;
        ctx.session_id = session_id_;
        return ctx;
    }();
    auto result = workflow_engine_->executeStep(step, current_plan_.plan_id, context);
    return result.data;
}

void ExecutiveController::transition_to(ControllerState new_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    transition_to_unlocked(new_state);
}

void ExecutiveController::transition_to_unlocked(ControllerState new_state) {
    state_ = new_state;
}

void ExecutiveController::emit_event(EventType type, const std::string& step_id, const nlohmann::json& metadata) {
    ControllerEvent event;
    EventCallback cb;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        event.type = type;
        event.session_id = session_id_;
        event.plan_id = current_plan_.plan_id;
        event.step_id = step_id;
        event.timestamp_ms = nowMs();
        event.controller_state_name = state_to_name(state_);
        
        nlohmann::json enriched_meta = metadata;
        
        // Standard structural fields
        enriched_meta["current_index"] = current_plan_.current_index;
        if (!step_id.empty()) {
            enriched_meta["step_id"] = step_id;
        }

        // Schema Standardization (Cognate V2)
        // Ensure these fields exist if not provided to avoid UI key-missing errors
        if (!enriched_meta.contains("reasoning_stage")) enriched_meta["reasoning_stage"] = "standard";
        if (!enriched_meta.contains("confidence_score")) enriched_meta["confidence_score"] = 1.0f;
        if (!enriched_meta.contains("success")) enriched_meta["success"] = true;
        if (!enriched_meta.contains("iteration_count")) enriched_meta["iteration_count"] = reflection_count_;
        
        if (type == EventType::PLAN_CREATED || type == EventType::PLAN_REVISED) {
            enriched_meta["plan"] = current_plan_.to_json();
        }
        
        if (!step_id.empty()) {
            for (const auto& s : current_plan_.steps) {
                if (s.step_id == step_id) {
                    if (type == EventType::STEP_COMPLETED) {
                        enriched_meta["result"] = s.result;
                        enriched_meta["success"] = (s.status == StepStatus::SUCCESS);
                    }
                    if (type == EventType::STEP_STARTED) {
                        enriched_meta["step_type"] = static_cast<int>(s.type);
                        if (!enriched_meta.contains("timeout_ms")) {
                            enriched_meta["timeout_ms"] = LlmTimeoutPolicy::stepTimeoutMs(
                                s.type, s.failure_policy.timeout_ms);
                        }
                    }
                    if (type == EventType::STEP_FAILED) {
                        enriched_meta["success"] = false;
                        if (!enriched_meta.contains("timeout_ms")) {
                            enriched_meta["timeout_ms"] = LlmTimeoutPolicy::stepTimeoutMs(
                                s.type, s.failure_policy.timeout_ms);
                        }
                    }
                    if (!enriched_meta.contains("description") && !s.description.empty()) {
                        enriched_meta["description"] = s.description;
                    }
                    break;
                }
            }
        }

        // Special handling for terminal events
        if (type == EventType::PLAN_FAILED || type == EventType::PLAN_ABORTED) {
            enriched_meta["success"] = false;
        }
        
        event.metadata = enriched_meta;
        cb = event_callback_;
    }
    
    if (cb) {
        cb(event);
    }
    
    log_to_trace(event);
}

void ExecutiveController::bind_active_sets_to_plan_unlocked(const std::string& plan_id) {
    active_sets_plan_id_ = plan_id;
    retrieval_sets_memory_authoritative_ = true;
}

void ExecutiveController::clear_owned_active_sets_unlocked() {
    active_sets_.clear();
    if (!current_plan_.plan_id.empty()) {
        active_sets_plan_id_ = current_plan_.plan_id;
    }
    retrieval_sets_memory_authoritative_ = true;
}

void ExecutiveController::sync_retrieval_sets_into_plan_unlocked() {
    if (!retrieval_sets_memory_authoritative_) {
        return;
    }
    if (active_sets_plan_id_.empty() || active_sets_plan_id_ != current_plan_.plan_id) {
        return;
    }
    current_plan_.retrieval_sets_present = true;
    current_plan_.retrieval_sets.clear();
    for (const auto& set : active_sets_) {
        RetrievalSetRecord record;
        record.step_id = set.step_id;
        record.chunk_hashes = set.chunk_hashes;
        current_plan_.retrieval_sets.push_back(std::move(record));
    }
}

bool ExecutiveController::plan_has_unfinished_steps_unlocked() const {
    for (const auto& step : current_plan_.steps) {
        if (step.status == StepStatus::PENDING || step.status == StepStatus::RUNNING) {
            return true;
        }
    }
    return false;
}

void ExecutiveController::log_retrieval_set_diagnostic(const char* event, const std::string& detail) const {
    StructuredLogger::instance().log(
        LogLevel::Warn,
        "controller",
        event,
        detail,
        {{"plan_id", current_plan_.plan_id}});
}

ExecutiveController::RetrievalCaptureResult ExecutiveController::capture_retrieval_set_unlocked(
    const std::string& step_id,
    const nlohmann::json& result,
    bool register_nodes) {
    RetrievalCaptureResult captured;
    captured.set.step_id = step_id;
    if (!result.is_object() || !result.contains("diagnostics")) {
        captured.unusable = true;
        return captured;
    }
    try {
        const auto& diag = result["diagnostics"];
        if (!diag.is_object() || !diag.contains("breakdowns") || !diag["breakdowns"].is_array()) {
            captured.unusable = true;
            return captured;
        }
        const auto& breakdowns = diag["breakdowns"];
        float bestScore = 0.0f;
        if (!breakdowns.empty()) {
            bestScore = breakdowns[0].value("final_score", 0.0f);
        }
        for (const auto& breakdown : breakdowns) {
            if (!breakdown.is_object()) {
                continue;
            }
            const float score = breakdown.value("final_score", 0.0f);
            if (score < 0.8f * bestScore || captured.set.chunk_hashes.size() >= 5) {
                continue;
            }
            const std::string content = breakdown.value("code_text", "");
            if (content.empty()) {
                continue;
            }
            const std::string hash = Memory::calculateContentHash(content);
            captured.set.chunk_hashes.push_back(hash);
            if (register_nodes && memory_) {
                Memory::Node node;
                node.id = hash;
                node.file_path = breakdown.value("file_name", "");
                node.symbol = breakdown.value("symbol", "");
                node.type = "code_chunk";
                memory_->addNode(node);
            }
        }
    } catch (...) {
        captured.unusable = true;
        captured.set.chunk_hashes.clear();
    }
    return captured;
}

void ExecutiveController::restore_legacy_retrieval_sets_unlocked() {
    struct Candidate {
        int64_t completed_at_ms = 0;
        size_t index = 0;
        ActiveStepSet set;
    };
    std::vector<Candidate> candidates;
    for (size_t index = 0; index < current_plan_.steps.size(); ++index) {
        const PlanStep& step = current_plan_.steps[index];
        if (step.type != StepType::RETRIEVAL || step.status != StepStatus::SUCCESS) {
            continue;
        }
        const RetrievalCaptureResult captured = capture_retrieval_set_unlocked(
            step.step_id, step.result, false);
        if (captured.unusable) {
            log_retrieval_set_diagnostic(
                "retrieval_sets_legacy_incomplete",
                "Successful retrieval step has no usable diagnostics. Its set was not reconstructed.");
            continue;
        }
        if (captured.set.chunk_hashes.empty()) {
            continue;
        }
        Candidate candidate;
        candidate.completed_at_ms = step.completed_at_ms;
        candidate.index = index;
        candidate.set = captured.set;
        candidates.push_back(std::move(candidate));
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.completed_at_ms != b.completed_at_ms) {
            return a.completed_at_ms < b.completed_at_ms;
        }
        return a.index < b.index;
    });
    for (size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].completed_at_ms == candidates[i - 1].completed_at_ms) {
            log_retrieval_set_diagnostic(
                "retrieval_sets_legacy_order_uncertain",
                "Completed retrieval steps share completed_at_ms. Order follows step index.");
            break;
        }
    }
    for (const auto& candidate : candidates) {
        active_sets_.push_back(candidate.set);
    }
}

void ExecutiveController::adopt_resumed_retrieval_sets_unlocked() {
    reflection_count_ = current_plan_.reflection_count;
    if (!current_plan_.plan_id.empty() && current_plan_.plan_id == active_sets_plan_id_) {
        return;
    }

    active_sets_.clear();
    active_sets_plan_id_ = current_plan_.plan_id;

    if (current_plan_.reflection_pending ||
        planStatusIsTerminal(current_plan_.status) ||
        current_plan_.graph_reinforced) {
        retrieval_sets_memory_authoritative_ = false;
        return;
    }
    // A finished ACTIVE plan with a persisted set list and no consumed marker
    // is a rolled-back reinforcement. Restore it so the attempt is applied once.
    // A legacy finished plan has no list and no marker, so its diagnostics must
    // not be reconstructed: those edges may already have autocommitted.
    if (!plan_has_unfinished_steps_unlocked() && !current_plan_.retrieval_sets_present) {
        retrieval_sets_memory_authoritative_ = false;
        log_retrieval_set_diagnostic(
            "retrieval_sets_restore_deferred",
            "Finished legacy plan has no retrieval_sets field. Sets were not reconstructed.");
        return;
    }

    retrieval_sets_memory_authoritative_ = true;
    if (current_plan_.retrieval_sets_present) {
        for (const auto& record : current_plan_.retrieval_sets) {
            if (record.chunk_hashes.empty()) {
                continue;
            }
            ActiveStepSet set;
            set.step_id = record.step_id;
            set.chunk_hashes = record.chunk_hashes;
            active_sets_.push_back(std::move(set));
        }
        return;
    }
    restore_legacy_retrieval_sets_unlocked();
}

namespace {

MemoryRepository::ActivePlanRecord activePlanRecordFrom(const Plan& plan,
                                                        const std::string& session_id,
                                                        const std::string& controller_state) {
    MemoryRepository::ActivePlanRecord rec;
    rec.plan_id = plan.plan_id;
    rec.session_id = session_id;
    rec.goal = plan.goal;
    rec.steps_json = plan.to_json().dump();
    rec.current_index = static_cast<int>(plan.current_index);
    rec.controller_state = controller_state;
    rec.created_at_ms = plan.created_at_ms;
    rec.updated_at_ms = plan.updated_at_ms;
    return rec;
}

} // namespace

bool ExecutiveController::reinforce_plan_graph() {
    if (!memory_ || current_plan_.plan_id.empty()) {
        active_sets_.clear();
        active_sets_plan_id_.clear();
        retrieval_sets_memory_authoritative_ = true;
        return false;
    }
    if (current_plan_.graph_reinforced) {
        active_sets_.clear();
        return true;
    }

    std::vector<Memory::Edge> reinforced;
    if (graph_refiner_ && active_sets_.size() >= 2) {
        std::vector<Memory::Edge> trajectory_edges;
        for (size_t i = 0; i < active_sets_.size() - 1; ++i) {
            const auto& fromSet = active_sets_[i];
            const auto& toSet = active_sets_[i + 1];
            for (const auto& fromHash : fromSet.chunk_hashes) {
                for (const auto& toHash : toSet.chunk_hashes) {
                    if (fromHash == toHash) continue;
                    Memory::Edge edge;
                    edge.from_id = fromHash;
                    edge.to_id = toHash;
                    auto existing = memory_->getEdgesFrom(fromHash);
                    bool found = false;
                    for (const auto& candidate : existing) {
                        if (candidate.to_id == toHash) {
                            edge = candidate;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        edge.weight = 0.1f;
                        edge.success_count = 0;
                        edge.failure_count = 0;
                    }
                    edge.last_used_ms = nowMs();
                    trajectory_edges.push_back(edge);
                }
            }
        }
        if (!trajectory_edges.empty()) {
            const float success_score =
                (current_plan_.status == PlanStatus::COMPLETED) ? 1.0f : 0.0f;
            reinforced = graph_refiner_->computeReinforcedEdges(trajectory_edges, success_score);
        }
    }

    if (reflection_handoff_) {
        capture_working_trajectory_unlocked(current_plan_);
    }
    sync_goal_metrics_checkpoint_unlocked();
    Plan durable = current_plan_;
    durable.graph_reinforced = true;
    durable.retrieval_sets.clear();
    durable.retrieval_sets_present = true;
    durable.history_aging_done = false;
    durable.updated_at_ms = nowMs();
    if (reflection_handoff_) {
        durable.status = PlanStatus::ACTIVE;
        durable.reflection_pending = true;
        durable.reflection_count = reflection_count_ + 1;
    } else {
        durable.reflection_pending = false;
        durable.reflection_count = reflection_count_;
    }

    if (!memory_->commitReinforcedPlan(
            reinforced, activePlanRecordFrom(durable, session_id_, state_to_name(state_)))) {
        return false;
    }

    current_plan_ = durable;
    if (reflection_handoff_) {
        reflection_count_ = durable.reflection_count;
    }
    active_sets_.clear();
    retrieval_sets_memory_authoritative_ = true;
    return true;
}

bool ExecutiveController::commit_history_aging_unlocked() {
    if (!memory_ || current_plan_.plan_id.empty()) return false;
    if (current_plan_.history_aging_done) return true;

    sync_goal_metrics_checkpoint_unlocked();
    Plan durable = current_plan_;
    if (durable.reflection_pending) {
        capture_working_trajectory_unlocked(durable);
    }
    durable.history_aging_done = true;
    durable.graph_reinforced = true;
    durable.retrieval_sets.clear();
    durable.retrieval_sets_present = true;
    durable.updated_at_ms = nowMs();
    const std::string* metrics_payload = nullptr;
    std::string metrics_json;
    if (planStatusIsTerminal(current_plan_.status) && !current_plan_.reflection_pending) {
        std::string outcome = "failed";
        if (current_plan_.status == PlanStatus::COMPLETED) outcome = "completed";
        else if (current_plan_.status == PlanStatus::ABORTED) outcome = "aborted";
        metrics_json = goal_cognitive_metrics_payload_unlocked(outcome, final_trajectory_score_).dump();
        metrics_payload = &metrics_json;
    }
    if (!memory_->commitHistoryStoreAging(
            activePlanRecordFrom(durable, session_id_, state_to_name(state_)),
            metrics_payload)) {
        return false;
    }
    current_plan_.history_aging_done = true;
    current_plan_.updated_at_ms = durable.updated_at_ms;
    if (durable.working_trajectory_present) {
        current_plan_.working_trajectory_present = true;
        current_plan_.working_trajectory = durable.working_trajectory;
        current_plan_.working_revisions_count = durable.working_revisions_count;
    }
    return true;
}

void ExecutiveController::continue_pending_reflection() {
    std::string reflection_goal;
    std::string plannerSessionId;
    std::string reflectionTask;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!current_plan_.reflection_pending || !current_plan_.graph_reinforced) {
            return;
        }
        if (!current_plan_.history_aging_done && !commit_history_aging_unlocked()) {
            return;
        }
        if (strategy_engine_) {
            strategy_engine_->processTrajectories();
        }
        float score = 0.0f;
        if (current_plan_.recorded_trajectory_score_present) {
            score = current_plan_.recorded_trajectory_score;
        } else {
            bool all_successful = !current_plan_.steps.empty();
            for (const auto& step : current_plan_.steps) {
                if (step.status != StepStatus::SUCCESS) {
                    all_successful = false;
                    break;
                }
            }
            score = calculate_trajectory_score(all_successful);
        }
        reflection_goal = Thoth::cleanGoalForStorage(current_plan_.goal) +
            " (Reflection: previous attempt had low success score " + std::to_string(score) + ")";
        plannerSessionId = session_id_;
        reflectionTask = mtcp_task_id_;
    }

    if (!planner_) return;
    const auto planStart = nowMs();
    Plan new_plan;
    {
        PlannerLogSessionGuard sessionGuard(plannerSessionId);
        Thoth::GenerationCallContext scopeContext;
        scopeContext.task_id = reflectionTask;
        scopeContext.session_id = plannerSessionId;
        scopeContext.call_type = "plan";
        scopeContext.reflection = true;
        Thoth::GenerationCallScope scope(scopeContext);
        new_plan = planner_->create_plan(reflection_goal);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        planning_time_ms_ += nowMs() - planStart;
        sync_planning_tokens_unlocked();
        if (!current_plan_.reflection_pending) {
            return;
        }
        const Plan previous = current_plan_;
        const Trajectory previousTrajectory = current_trajectory_;
        const int previousRevisions = revisions_count_;
        const std::string retiredPlanId = previous.plan_id;
        current_plan_ = new_plan;
        current_plan_.created_at_ms = nowMs();
        current_plan_.updated_at_ms = current_plan_.created_at_ms;
        active_sets_.clear();
        bind_active_sets_to_plan_unlocked(current_plan_.plan_id);
        current_trajectory_ = Trajectory();
        current_trajectory_.trajectory_id = "traj-" + current_plan_.plan_id;
        current_trajectory_.goal = reflection_goal;
        current_trajectory_.plan_initial = current_plan_;
        current_trajectory_.created_at = current_plan_.created_at_ms;
        current_trajectory_.embedding = goal_embedding_;
        revisions_count_ = 0;
        if (!persist_current_plan_unlocked()) {
            current_plan_ = previous;
            current_trajectory_ = previousTrajectory;
            revisions_count_ = previousRevisions;
            active_sets_.clear();
            active_sets_plan_id_ = previous.plan_id;
            retrieval_sets_memory_authoritative_ = false;
            return;
        }
        if (memory_ && !retiredPlanId.empty() && retiredPlanId != current_plan_.plan_id) {
            memory_->deleteActivePlan(retiredPlanId);
        }
        transition_to_unlocked(ControllerState::IDLE);
        if (!stop_requested_ && !loop_thread_) {
            running_ = true;
            loop_thread_ = std::make_unique<std::thread>([this]() { run_loop(); });
        }
    }
    emit_event(EventType::PLAN_CREATED);
}

void ExecutiveController::record_trajectory_step(const PlanStep& step, const Thoth::StepResult& result) {
    RecordedStep recorded;
    recorded.step_id = step.step_id;
    recorded.description = step.description;
    recorded.type = step.type;
    recorded.tool = step.tool;
    recorded.result = result.data;
    recorded.error = result.error_message;
    recorded.timestamp = nowMs();
    current_trajectory_.steps.push_back(recorded);
}

void ExecutiveController::capture_working_trajectory_unlocked(Plan& plan) const {
    if (current_trajectory_.trajectory_id.empty()) return;
    plan.working_trajectory_present = true;
    plan.working_trajectory = current_trajectory_.to_json();
    plan.working_revisions_count = revisions_count_;
}

float ExecutiveController::calculate_trajectory_score(bool plan_completed_successfully) {
    if (current_plan_.steps.empty()) return 0.0f;
    float score = plan_completed_successfully ? 1.0f : 0.0f;

    int failed_steps = 0;
    for (const auto& s : current_trajectory_.steps) {
        if (!s.error.empty()) failed_steps++;
    }
    score -= (failed_steps * 0.1f);
    
    // Bonus for successfully completing even with revisions
    if (plan_completed_successfully && revisions_count_ > 0) {
        score += 0.1f; 
    }
    
    return std::max(0.0f, std::min(1.0f, score));
}

bool ExecutiveController::persist_current_plan_unlocked() {
    if (!memory_ || current_plan_.plan_id.empty()) return false;

    sync_goal_metrics_checkpoint_unlocked();
    sync_retrieval_sets_into_plan_unlocked();

    Plan stored = current_plan_;
    if (!current_trajectory_.trajectory_id.empty()) {
        stored.working_trajectory_present = true;
        stored.working_trajectory = current_trajectory_.to_json();
        stored.working_revisions_count = revisions_count_;
    }
    if (!memory_->storeActivePlan(
            activePlanRecordFrom(stored, session_id_, state_to_name(state_)))) {
        return false;
    }

    // A traj-<plan_id> row is written only after the finalized Cognate score.
    // Later active-plan checkpoints must not replace that row with score 0.
    // A computed score of 0 is valid history and is preserved the same way.
    if (memory_->loadTrajectory("traj-" + current_plan_.plan_id)) {
        return true;
    }

    MemoryRepository::CognatePlanRecord cognateRec;
    cognatePlanRecordFromPlan(current_plan_, cognateRec);
    cognateRec.status = static_cast<int>(state_);
    memory_->saveCognatePlan(cognateRec);
    return true;
}

namespace {

std::unordered_map<std::string, int> cm1IndexedCountsByFile(IndexManager* index) {
    std::unordered_map<std::string, int> counts;
    if (!index) return counts;
    for (const auto& chunk : index->getChunks()) {
        if (chunk.fileName.empty()) continue;
        counts[chunk.fileName] += 1;
    }
    return counts;
}

std::vector<std::pair<std::string, std::string>> cm1SelectedChunksFromResult(
    const nlohmann::json& step_result) {
    std::vector<std::pair<std::string, std::string>> selected;
    if (!step_result.is_object()) return selected;
    const nlohmann::json* data = nullptr;
    if (step_result.contains("data") && step_result["data"].is_object()) {
        if (step_result["data"].contains("chunks")) {
            data = &step_result["data"];
        } else if (step_result["data"].contains("data") && step_result["data"]["data"].is_object()
                   && step_result["data"]["data"].contains("chunks")) {
            data = &step_result["data"]["data"];
        }
    }
    if (!data || !data->contains("chunks") || !(*data)["chunks"].is_array()) return selected;
    for (const auto& chunk : (*data)["chunks"]) {
        if (!chunk.is_object()) continue;
        selected.emplace_back(chunk.value("file", ""), chunk.value("content", ""));
    }
    return selected;
}

std::optional<GragDiagnostics> cm1DiagnosticsFromResult(const nlohmann::json& step_result) {
    if (!step_result.is_object() || !step_result.contains("diagnostics")
        || !step_result["diagnostics"].is_object()) {
        return std::nullopt;
    }
    const auto& d = step_result["diagnostics"];
    GragDiagnostics out;
    out.scoring_type = d.value("scoring_type", "");
    out.routing_mode = d.value("routing_mode", "");
    out.alpha = d.value("alpha", 0.0f);
    out.direction_magnitude = d.value("direction_magnitude", 0.0f);
    out.chunks_retrieved = d.value("chunks_retrieved", 0);
    out.chunks_reranked = d.value("chunks_reranked", 0);
    out.graph_total_nodes = d.value("graph_total_nodes", 0);
    out.graph_total_edges = d.value("graph_total_edges", 0);
    out.graph_avg_weight = d.value("graph_avg_weight", 0.0f);
    out.graph_activations = d.value("graph_activations", 0);
    out.graph_max_contribution = d.value("graph_max_contribution", 0.0f);
    if (d.contains("final_scores") && d["final_scores"].is_array()) {
        for (const auto& s : d["final_scores"]) {
            if (s.is_number()) out.final_scores.push_back(s.get<float>());
        }
    }
    if (d.contains("breakdowns") && d["breakdowns"].is_array()) {
        for (const auto& b : d["breakdowns"]) {
            if (!b.is_object()) continue;
            ScoreBreakdown sb;
            sb.file_name = b.value("file_name", "");
            sb.symbol = b.value("symbol", "");
            sb.code_text = b.value("code_text", "");
            sb.source_node_id = b.value("source_node_id", "");
            if (b.contains("query_sim") && b["query_sim"].is_number()) {
                sb.query_sim = b["query_sim"].get<float>();
            } else {
                sb.query_sim = std::numeric_limits<float>::quiet_NaN();
            }
            sb.goal_sim = b.value("goal_sim", 0.0f);
            sb.trajectory_sim = b.value("trajectory_sim", 0.0f);
            sb.keyword_score = b.value("keyword_score", 0.0f);
            sb.graph_score = b.value("graph_score", 0.0f);
            sb.final_score = b.value("final_score", 0.0f);
            out.breakdowns.push_back(std::move(sb));
        }
    }
    return out;
}

} // namespace

bool ExecutiveController::commit_completed_step_unlocked(int episode_index) {
    if (!memory_ || current_plan_.plan_id.empty()) return false;
    const auto episode = episodeFromDurableStep(current_plan_, episode_index);
    if (!episode) return false;

    sync_goal_metrics_checkpoint_unlocked();
    sync_retrieval_sets_into_plan_unlocked();

    Plan stored = current_plan_;
    if (!current_trajectory_.trajectory_id.empty()) {
        stored.working_trajectory_present = true;
        stored.working_trajectory = current_trajectory_.to_json();
        stored.working_revisions_count = revisions_count_;
    }

    std::optional<MemoryRepository::CognatePlanRecord> cognate;
    if (!memory_->loadTrajectory("traj-" + current_plan_.plan_id)) {
        MemoryRepository::CognatePlanRecord cognateRec;
        cognatePlanRecordFromPlan(current_plan_, cognateRec);
        cognateRec.status = static_cast<int>(state_);
        cognate = cognateRec;
    }

    std::optional<Thoth::Cm1SnapshotRecord> cm1_snapshot;
    std::optional<Thoth::Cm1AnswerVersionRecord> cm1_answer;
    if (Thoth::cm1TelemetryEnabled()
        && episode_index >= 0
        && episode_index < static_cast<int>(current_plan_.steps.size())) {
        const PlanStep& step = current_plan_.steps[static_cast<std::size_t>(episode_index)];
        try {
            if (step.type == StepType::RETRIEVAL) {
                Thoth::Cm1SnapshotBuildInput input;
                input.plan_id = current_plan_.plan_id;
                input.step_id = step.step_id;
                input.snapshot_type = "retrieval_observed";
                input.lifecycle_stage = "after_retrieval";
                input.observed_at_ms = nowMs();
                const auto diagnostics = cm1DiagnosticsFromResult(step.result);
                if (diagnostics) {
                    input.diagnostics = &(*diagnostics);
                }
                IndexManager* index =
                    rag_ ? rag_->getIndexManager() : nullptr;
                auto counts = cm1IndexedCountsByFile(index);
                if (index) {
                    input.indexed_counts_available = true;
                    input.indexed_counts_by_file = std::move(counts);
                }
                input.selected_chunks = cm1SelectedChunksFromResult(step.result);
                cm1_snapshot = Thoth::buildCm1Snapshot(input);
            } else if (step.type == StepType::LLM && step.status == StepStatus::SUCCESS) {
                Thoth::Cm1SnapshotBuildInput input;
                input.plan_id = current_plan_.plan_id;
                input.step_id = step.step_id;
                input.snapshot_type = "synthesis_observed";
                input.lifecycle_stage = "after_synthesis";
                input.observed_at_ms = nowMs();
                std::optional<GragDiagnostics> held_diagnostics;
                for (const auto& prior : current_plan_.steps) {
                    if (prior.type != StepType::RETRIEVAL) continue;
                    auto selected = cm1SelectedChunksFromResult(prior.result);
                    input.selected_chunks.insert(
                        input.selected_chunks.end(), selected.begin(), selected.end());
                }
                for (auto it = current_plan_.steps.rbegin(); it != current_plan_.steps.rend();
                     ++it) {
                    if (it->type != StepType::RETRIEVAL) continue;
                    held_diagnostics = cm1DiagnosticsFromResult(it->result);
                    if (held_diagnostics) {
                        input.diagnostics = &(*held_diagnostics);
                        break;
                    }
                }
                IndexManager* index = rag_ ? rag_->getIndexManager() : nullptr;
                if (index) {
                    input.indexed_counts_available = true;
                    input.indexed_counts_by_file = cm1IndexedCountsByFile(index);
                }
                if (step.result.is_object()) {
                    if (step.result.contains("synthesis_context_truncated")) {
                        input.synthesis_context_truncated_available = true;
                        input.synthesis_context_truncated =
                            step.result.value("synthesis_context_truncated", false);
                    }
                    if (step.result.value("cm1_included_in_synthesis_available", false)
                        && step.result.contains("cm1_included_content_sha256")
                        && step.result["cm1_included_content_sha256"].is_array()) {
                        input.included_in_synthesis_available = true;
                        for (const auto& h : step.result["cm1_included_content_sha256"]) {
                            if (h.is_string()) input.included_content_sha256.push_back(h.get<std::string>());
                        }
                    }
                }
                cm1_snapshot = Thoth::buildCm1Snapshot(input);

                std::string generation_id;
                std::string answer_text;
                if (step.result.is_object()) {
                    generation_id = step.result.value("generation_id", "");
                    if (step.result.contains("data") && step.result["data"].is_object()) {
                        answer_text = step.result["data"].value("response", "");
                    }
                }
                if (!generation_id.empty() && !answer_text.empty()) {
                    cm1_answer = Thoth::buildCm1AnswerVersion(
                        current_plan_.plan_id,
                        step.step_id,
                        generation_id,
                        answer_text,
                        nowMs());
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << "[ExecutiveController] CM-1 telemetry build failed: " << ex.what()
                      << " — continuing without telemetry\n";
            cm1_snapshot.reset();
            cm1_answer.reset();
        } catch (...) {
            std::cerr << "[ExecutiveController] CM-1 telemetry build failed — "
                         "continuing without telemetry\n";
            cm1_snapshot.reset();
            cm1_answer.reset();
        }
    }

    std::optional<Thoth::Cm1BreadthEstimate> cm1_breadth;
    if (cm1_snapshot && Thoth::cm1BreadthEnabled()) {
        try {
            const auto prior = memory_->listCm1SnapshotsForPlan(current_plan_.plan_id);
            cm1_breadth = Thoth::buildCm1BreadthEstimate(*cm1_snapshot, prior);
        } catch (const std::exception& ex) {
            std::cerr << "[ExecutiveController] CM-1 breadth estimate failed: " << ex.what()
                      << " — snapshot kept, estimate omitted\n";
        } catch (...) {
            std::cerr << "[ExecutiveController] CM-1 breadth estimate failed — "
                         "snapshot kept, estimate omitted\n";
        }
    }

    return memory_->commitStepCheckpoint(
        activePlanRecordFrom(stored, session_id_, state_to_name(state_)),
        cognate,
        episode,
        cm1_snapshot,
        cm1_answer,
        cm1_breadth);
}

void ExecutiveController::backfill_durable_episode_steps_unlocked(const std::vector<int>& durable_indices) {
    if (!memory_) return;
    for (const int index : durable_indices) {
        if (const auto episode = episodeFromDurableStep(current_plan_, index)) {
            memory_->storeEpisodeStep(*episode);
        }
    }
}

std::string ExecutiveController::state_to_name(ControllerState state) const {
    switch (state) {
        case ControllerState::IDLE: return "IDLE";
        case ControllerState::PLANNING: return "PLANNING";
        case ControllerState::EXECUTING_STEP: return "EXECUTING_STEP";
        case ControllerState::OBSERVING_RESULT: return "OBSERVING_RESULT";
        case ControllerState::REVISING_PLAN: return "REVISING_PLAN";
        case ControllerState::SCIENTIFIC_MODE: return "SCIENTIFIC_MODE";
        case ControllerState::COMPLETED: return "COMPLETED";
        case ControllerState::ABORTED: return "ABORTED";
        case ControllerState::FAILED: return "FAILED";
    }
    return "UNKNOWN";
}

void ExecutiveController::log_to_trace(const ControllerEvent& event) {
    std::string type_str;
    switch (event.type) {
        case EventType::PLAN_CREATED: type_str = "PLAN_CREATED"; break;
        case EventType::STEP_STARTED: type_str = "STEP_STARTED"; break;
        case EventType::STEP_COMPLETED: type_str = "STEP_COMPLETED"; break;
        case EventType::STEP_FAILED: type_str = "STEP_FAILED"; break;
        case EventType::STEP_RETRYING: type_str = "STEP_RETRYING"; break;
        case EventType::PLAN_REVISED: type_str = "PLAN_REVISED"; break;
        case EventType::PLAN_COMPLETED: type_str = "PLAN_COMPLETED"; break;
        case EventType::PLAN_ABORTED: type_str = "PLAN_ABORTED"; break;
        case EventType::PLAN_FAILED: type_str = "PLAN_FAILED"; break;
        case EventType::STATE_CHANGED: type_str = "STATE_CHANGED"; break;
        case EventType::MODE_SWITCHED: type_str = "MODE_SWITCHED"; break;
        case EventType::EMBEDDING_FAILED: type_str = "EMBEDDING_FAILED"; break;
        case EventType::RETRIEVAL_DIAGNOSTICS: type_str = "RETRIEVAL_DIAGNOSTICS"; break;
        case EventType::INDEXING_STARTED: type_str = "INDEXING_STARTED"; break;
        case EventType::INDEXING_COMPLETED: type_str = "INDEXING_COMPLETED"; break;
        case EventType::PLAN_REUSE_INJECTION: type_str = "PLAN_REUSE_INJECTION"; break;
        case EventType::REFLECTION_REPLAN: type_str = "REFLECTION_REPLAN"; break;
        case EventType::PLAN_HISTORY_STORED: type_str = "PLAN_HISTORY_STORED"; break;
    }

    nlohmann::json entry;
    entry["timestamp_ms"] = event.timestamp_ms;
    entry["event_type"] = type_str;
    entry["plan_id"] = event.plan_id;
    entry["step_id"] = event.step_id;
    entry["controller_state"] = event.controller_state_name;
    entry["metadata"] = event.metadata;

    FileHandler fh;
    std::string path = fh.getAgentWorkspacePath("decision_trace.jsonl");
    
    static std::mutex trace_file_mutex;
    std::lock_guard<std::mutex> lock(trace_file_mutex);
    
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << entry.dump() << "\n";
    }
}

void ExecutiveController::update_goal_embedding(const std::string& goal) {
    std::lock_guard<std::mutex> lock(mutex_);
    update_goal_embedding_unlocked(goal);
}

void ExecutiveController::update_goal_embedding_unlocked(const std::string& goal) {
    if (rag_ && rag_->engine) {
        nlohmann::json structured_goal;
        structured_goal["schema_version"] = 2;
        structured_goal["type"] = "goal";
        structured_goal["content"] = goal;
        
        std::string goal_str = structured_goal.dump();
        goal_embedding_ = rag_->engine->embed(goal_str);
        
        if (goal_embedding_.empty()) {
            emit_event(EventType::EMBEDDING_FAILED, "", {{"type", "goal"}, {"input", goal_str}});
        }
        
        rag_->setGoalEmbedding(goal_embedding_);
    }
}

void ExecutiveController::update_current_embedding() {
    std::lock_guard<std::mutex> lock(mutex_);
    update_current_embedding_unlocked();
}

void ExecutiveController::update_current_embedding_unlocked() {
    if (!rag_ || !rag_->engine) return;

    nlohmann::json state;
    state["schema_version"] = 2;
    state["goal"] = Thoth::cleanGoalForStorage(current_plan_.goal);
    state["completed_steps_summary"] = "none";
    state["remaining_steps_summary"] = "none";
    state["constraints"] = "none";
    state["known_blockers"] = "none";

    const std::string state_str = state.dump();
    current_embedding_ = rag_->engine->embed(state_str);

    if (current_embedding_.empty()) {
        emit_event(EventType::EMBEDDING_FAILED, "", {{"type", "state"}, {"input", state_str}});
    }

    rag_->setCurrentEmbedding(current_embedding_);
}

void ExecutiveController::refresh_goal_state_embeddings_unlocked() {
    if (!rag_ || !rag_->engine || current_plan_.goal.empty()) {
        return;
    }

    const std::string clean_goal = Thoth::cleanGoalForStorage(current_plan_.goal);

    nlohmann::json structured_goal;
    structured_goal["schema_version"] = 2;
    structured_goal["type"] = "goal";
    structured_goal["content"] = clean_goal;

    nlohmann::json state;
    state["schema_version"] = 2;
    state["goal"] = clean_goal;
    state["completed_steps_summary"] = "none";
    state["remaining_steps_summary"] = "none";
    state["constraints"] = "none";
    state["known_blockers"] = "none";

    const std::string goal_str = structured_goal.dump();
    const std::string state_str = state.dump();
    const auto batch = rag_->engine->embedBatch({goal_str, state_str});
    if (batch.size() < 2) {
        return;
    }

    goal_embedding_ = batch[0];
    current_embedding_ = batch[1];

    if (goal_embedding_.empty()) {
        emit_event(EventType::EMBEDDING_FAILED, "", {{"type", "goal"}, {"input", goal_str}});
    }
    if (current_embedding_.empty()) {
        emit_event(EventType::EMBEDDING_FAILED, "", {{"type", "state"}, {"input", state_str}});
    }

    rag_->setGoalEmbedding(goal_embedding_);
    rag_->setCurrentEmbedding(current_embedding_);
    current_trajectory_.embedding = goal_embedding_;
}

void ExecutiveController::update_trajectory_embedding() {
    std::lock_guard<std::mutex> lock(mutex_);
    update_trajectory_embedding_unlocked();
}

void ExecutiveController::update_trajectory_embedding_unlocked() {
    if (trajectory_builder_ && rag_) {
        // T embedding is computed from recent episode steps
        trajectory_embedding_ = trajectory_builder_->buildTrajectory(current_plan_.plan_id);
        
        // Requirement: If fewer than 3 episode steps exist, return a zero vector and set wt=0.0
        // buildTrajectory already returns zero vector if < 3 steps.
        // We need to temporarily override wt in the RAG config if it's zero.
        
        bool is_zero = true;
        for (float v : trajectory_embedding_) {
            if (std::abs(v) > TrajectoryReuse::kZeroVectorEpsilon) {
                is_zero = false;
                break;
            }
        }

        if (is_zero) {
            rag_->getRetrievalConfig().wt = 0.0f;
        } else {
            // Restore from global config if it was zeroed
            if (rag_->config) rag_->getRetrievalConfig().wt = rag_->config->wt;
            else rag_->getRetrievalConfig().wt = -0.05f;
        }

        rag_->setTrajectoryEmbedding(trajectory_embedding_);
    }
}

void ExecutiveController::clear_embeddings() {
    std::lock_guard<std::mutex> lock(mutex_);
    clear_embeddings_unlocked();
}

void ExecutiveController::clear_embeddings_unlocked() {
    goal_embedding_.clear();
    current_embedding_.clear();
    trajectory_embedding_.clear();
    if (rag_) {
        rag_->setGoalEmbedding({});
        rag_->setCurrentEmbedding({});
        rag_->setTrajectoryEmbedding({});
    }
}

namespace {

bool graphCampaignHaltEnabled(const char* flag) {
    const char* value = std::getenv(flag);
    if (!value || std::string(value) != "1") {
        return false;
    }
    const char* dir = std::getenv("THOTH_GRAPH_HALT_DIR");
    if (!dir || std::string(dir).rfind("/tmp/thoth-e2e-graph-", 0) != 0) {
        return false;
    }
    if (std::string(dir).find("..") != std::string::npos) {
        return false;
    }
    return true;
}

// Test-only park. Both the flag and the disposable halt directory are required.
// The parent supervisor observes SIGSTOP, reads the committed database, then
// sends SIGKILL. Production runs leave the flag unset and never enter.
void parkForGraphCampaignHalt(const char* boundary) {
    const std::filesystem::path dir(std::getenv("THOTH_GRAPH_HALT_DIR"));
    const std::filesystem::path marker = dir / "graph_halt.json";
    const std::filesystem::path temporary = dir / "graph_halt.json.tmp";
    nlohmann::json body = {
        {"boundary", boundary},
        {"pid", static_cast<int>(::getpid())}
    };
    {
        std::ofstream out(temporary, std::ios::trunc);
        if (!out) {
            return;
        }
        out << body.dump(2);
        out.flush();
    }
    std::error_code renameError;
    std::filesystem::rename(temporary, marker, renameError);
    if (renameError) {
        return;
    }
    for (;;) {
        ::raise(SIGSTOP);
    }
}

} // namespace

nlohmann::json ExecutiveController::store_plan_history(float success_score) {
    if (!memory_ || current_plan_.plan_id.empty()) {
        active_sets_.clear();
        active_sets_plan_id_.clear();
        retrieval_sets_memory_authoritative_ = true;
        return nlohmann::json();
    }

    Memory::PastPlanRecord record;
    record.plan_id = current_plan_.plan_id;
    record.goal = Thoth::cleanGoalForStorage(current_plan_.goal);
    record.outline = current_plan_.to_json().dump();
    record.success_score = success_score;
    record.duration_ms = nowMs() - current_plan_.created_at_ms;
    record.failure_count = 0;
    for (const auto& step : current_plan_.steps) {
        record.failure_count += step.retry_count;
    }
    record.goal_embedding = goal_embedding_;
    record.plan_origin = Thoth::normalizePlanOrigin(current_plan_.plan_origin);

    if (!current_plan_.graph_reinforced) {
        // storePastPlan, saveCognatePlan, then saveTrajectory. Each returns
        // before the next begins, so a committed traj-<plan_id> row means the
        // past-plan and Cognate rows for this plan were already committed.
        // Leave all three untouched. A replace would reset usage_count and tier.
        const std::string stableTrajectoryId = "traj-" + current_plan_.plan_id;
        const auto existingTrajectory = memory_->loadTrajectory(stableTrajectoryId);
        if (!existingTrajectory) {
            if (current_trajectory_.trajectory_id.empty()) {
                current_trajectory_.trajectory_id = stableTrajectoryId;
            }
            if (graphCampaignHaltEnabled("THOTH_HALT_BEFORE_HISTORY_WRITE")) {
                parkForGraphCampaignHalt("before_history_write");
            }
            if (!memory_->storePastPlan(record)) return nlohmann::json();
            if (graphCampaignHaltEnabled("THOTH_HALT_AFTER_PAST_PLAN")) {
                parkForGraphCampaignHalt("after_past_plan");
            }

            MemoryRepository::CognatePlanRecord cognateRec;
            cognatePlanRecordFromPlan(current_plan_, cognateRec);
            cognateRec.status = static_cast<int>(state_);
            cognateRec.success_score = success_score;
            if (!memory_->saveCognatePlan(cognateRec)) return nlohmann::json();

            current_trajectory_.final_status = current_plan_.status;
            current_trajectory_.success_score = success_score;
            current_trajectory_.results = current_plan_.steps.empty()
                ? nlohmann::json::object()
                : current_plan_.steps.back().result;

            Memory::CognateTrajectoryRecord trajRec;
            trajRec.trajectory_id = current_trajectory_.trajectory_id;
            trajRec.goal = current_trajectory_.goal;
            trajRec.trajectory_json = current_trajectory_.to_json().dump();
            trajRec.success_score = success_score;
            trajRec.plan_origin = record.plan_origin;
            trajRec.embedding = current_trajectory_.embedding;
            trajRec.created_at = current_trajectory_.created_at;
            if (!memory_->saveTrajectory(trajRec)) return nlohmann::json();

            if (graphCampaignHaltEnabled("THOTH_HALT_AFTER_TRAJECTORY_COMMIT")) {
                parkForGraphCampaignHalt("after_trajectory_commit");
            }
        }

        if (reflection_handoff_) {
            current_plan_.recorded_trajectory_score = success_score;
            current_plan_.recorded_trajectory_score_present = true;
            if (current_trajectory_.trajectory_id.empty()) {
                current_trajectory_.trajectory_id = "traj-" + current_plan_.plan_id;
            }
            capture_working_trajectory_unlocked(current_plan_);
        }
        if (!reinforce_plan_graph()) return nlohmann::json();
    }

    if (graphCampaignHaltEnabled("THOTH_HALT_AFTER_GRAPH_COMMIT")) {
        parkForGraphCampaignHalt("after_graph_commit");
    }

    if (!current_plan_.history_aging_done && !commit_history_aging_unlocked()) {
        return nlohmann::json();
    }

    if (graphCampaignHaltEnabled("THOTH_HALT_AFTER_AGING_COMMIT")) {
        parkForGraphCampaignHalt("after_aging_commit");
    }
    if (strategy_engine_) {
        strategy_engine_->processTrajectories();
    }

    nlohmann::json history_meta = log_plan_history_persisted(success_score, record);
    
    StructuredLogger::instance().log(LogLevel::Info, "controller", "plan_history_stored", "Plan history persisted", 
        {{"plan_id", record.plan_id}, {"success_score", success_score}});

    Thoth::GragMetrics metrics;
    metrics.goal_id = current_plan_.plan_id;
    metrics.start_time_ms = current_plan_.created_at_ms;
    metrics.end_time_ms = nowMs();
    metrics.revision_count = revisions_count_;
    metrics.total_steps = (int)current_plan_.steps.size();
    metrics.successful_steps = 0;
    metrics.failed_steps = 0;
    for (const auto& step : current_plan_.steps) {
        if (step.status == StepStatus::SUCCESS) metrics.successful_steps++;
        else if (step.status == StepStatus::FAILED) metrics.failed_steps++;
    }
    metrics.plan_reused = plan_reused_;
    metrics.final_success_score = success_score;
    metrics.status = state_to_name(state_);

    Thoth::GragMetricsLogger::instance().logMetrics(metrics);
    return history_meta;
}

void ExecutiveController::reset_goal_metrics_unlocked() {
    goal_started_at_ms_ = nowMs();
    planning_time_ms_ = 0;
    retrieval_time_ms_ = 0;
    llm_synthesis_time_ms_ = 0;
    retrieved_chunk_count_ = 0;
    synthesis_prompt_chars_ = 0;
    synthesis_context_truncated_ = false;
    planning_tokens_ = 0;
    if (llm_interface_) {
        llm_interface_->resetSessionTokenUsage();
    }
    last_grag_alpha_ = 0.0f;
    last_grag_routing_mode_.clear();
    final_trajectory_score_ = 0.0f;
    c64_window_id_.clear();
    c64_protocol_version_.clear();
    c64_metric_schema_version_.clear();
    c64_environment_schema_version_.clear();
    c64_cohort_fingerprint_.clear();
    const C64WindowAssignment window = resolveC64WindowAssignment(goal_started_at_ms_);
    if (window.assigned) {
        c64_window_id_ = window.window_id;
        c64_protocol_version_ = window.protocol_version;
        c64_metric_schema_version_ = window.metric_schema_version;
        c64_environment_schema_version_ = window.environment_schema_version;
        c64_cohort_fingerprint_ = window.c64_cohort_fingerprint;
    }
    reflection_skip_reason_.clear();
    metrics_checkpoint_gap_ = false;
    metrics_legacy_resume_ = false;
    cognitive_metrics_emitted_ = false;
    planning_tokens_at_resume_ = 0;
    token_base_prompt_ = 0;
    token_base_completion_ = 0;
    token_base_total_ = 0;
    prefetch_cache_.clear();
    prefetch_futures_.clear();
    prefetch_step_ids_.clear();
    retrieval_prefetch_hits_ = 0;
}

void ExecutiveController::record_step_metrics_unlocked(const PlanStep& step, const StepResult& result) {
    if (step.type == StepType::RETRIEVAL) {
        retrieval_time_ms_ += result.latency_ms;
        if (result.data.contains("grag_alpha") && result.data["grag_alpha"].is_number()) {
            last_grag_alpha_ = result.data["grag_alpha"].get<float>();
        }
        if (result.data.contains("grag_routing_mode") && result.data["grag_routing_mode"].is_string()) {
            last_grag_routing_mode_ = result.data["grag_routing_mode"].get<std::string>();
        }
        if (result.data.contains("retrieved_chunk_count") && result.data["retrieved_chunk_count"].is_number()) {
            retrieved_chunk_count_ += result.data["retrieved_chunk_count"].get<int>();
        } else if (result.data.contains("data") && result.data["data"].is_object() &&
                   result.data["data"].contains("chunks") && result.data["data"]["chunks"].is_array()) {
            retrieved_chunk_count_ +=
                static_cast<int>(result.data["data"]["chunks"].size());
        }
    } else if (step.type == StepType::LLM) {
        llm_synthesis_time_ms_ += result.latency_ms;
        if (result.data.contains("synthesis_prompt_chars") && result.data["synthesis_prompt_chars"].is_number()) {
            synthesis_prompt_chars_ = result.data["synthesis_prompt_chars"].get<int>();
        }
        if (result.data.contains("synthesis_context_truncated") &&
            result.data["synthesis_context_truncated"].is_boolean()) {
            synthesis_context_truncated_ = result.data["synthesis_context_truncated"].get<bool>();
        }
    }
}

void ExecutiveController::sync_planning_tokens_unlocked() {
    if (!llm_interface_) {
        return;
    }
    planning_tokens_ = planning_tokens_at_resume_ + llm_interface_->sessionTokenUsage().total_tokens;
}

void ExecutiveController::sync_goal_metrics_checkpoint_unlocked() {
    if (metrics_legacy_resume_) {
        return;
    }
    LlmTokenUsage session;
    if (llm_interface_) {
        session = llm_interface_->sessionTokenUsage();
    }
    GoalMetricsCheckpoint& metrics = current_plan_.goal_metrics;
    metrics.present = true;
    metrics.window_assigned = !c64_window_id_.empty();
    metrics.window_id = c64_window_id_;
    metrics.protocol_version = c64_protocol_version_;
    metrics.metric_schema_version = c64_metric_schema_version_;
    metrics.environment_schema_version = c64_environment_schema_version_;
    metrics.cohort_fingerprint = c64_cohort_fingerprint_;
    metrics.run_id = benchmark_attribution_.run_id;
    metrics.env_hash = benchmark_attribution_.env_hash;
    metrics.goal_started_at_ms = goal_started_at_ms_;
    metrics.planning_time_ms = planning_time_ms_;
    metrics.retrieval_time_ms = retrieval_time_ms_;
    metrics.llm_synthesis_time_ms = llm_synthesis_time_ms_;
    metrics.retrieved_chunk_count = retrieved_chunk_count_;
    metrics.grag_alpha = last_grag_alpha_;
    metrics.grag_routing_mode = last_grag_routing_mode_;
    metrics.synthesis_prompt_chars = synthesis_prompt_chars_;
    metrics.synthesis_context_truncated = synthesis_context_truncated_;
    metrics.planning_tokens = planning_tokens_;
    metrics.prompt_tokens = token_base_prompt_ + session.prompt_tokens;
    metrics.completion_tokens = token_base_completion_ + session.completion_tokens;
    metrics.total_tokens = token_base_total_ + session.total_tokens;
}

void ExecutiveController::restore_goal_metrics_unlocked() {
    metrics_checkpoint_gap_ = true;
    cognitive_metrics_emitted_ = false;
    const GoalMetricsCheckpoint& metrics = current_plan_.goal_metrics;
    if (!metrics.present) {
        metrics_legacy_resume_ = true;
        c64_window_id_.clear();
        c64_protocol_version_.clear();
        c64_metric_schema_version_.clear();
        c64_environment_schema_version_.clear();
        c64_cohort_fingerprint_.clear();
        goal_started_at_ms_ = 0;
        planning_time_ms_ = 0;
        retrieval_time_ms_ = 0;
        llm_synthesis_time_ms_ = 0;
        retrieved_chunk_count_ = 0;
        synthesis_prompt_chars_ = 0;
        synthesis_context_truncated_ = false;
        planning_tokens_ = 0;
        planning_tokens_at_resume_ = 0;
        token_base_prompt_ = 0;
        token_base_completion_ = 0;
        token_base_total_ = 0;
        benchmark_attribution_ = {};
        return;
    }
    metrics_legacy_resume_ = false;
    if (metrics.window_assigned) {
        c64_window_id_ = metrics.window_id;
        c64_protocol_version_ = metrics.protocol_version;
        c64_metric_schema_version_ = metrics.metric_schema_version;
        c64_environment_schema_version_ = metrics.environment_schema_version;
        c64_cohort_fingerprint_ = metrics.cohort_fingerprint;
    } else {
        c64_window_id_.clear();
        c64_protocol_version_.clear();
        c64_metric_schema_version_.clear();
        c64_environment_schema_version_.clear();
        c64_cohort_fingerprint_.clear();
    }
    goal_started_at_ms_ = metrics.goal_started_at_ms;
    planning_time_ms_ = metrics.planning_time_ms;
    retrieval_time_ms_ = metrics.retrieval_time_ms;
    llm_synthesis_time_ms_ = metrics.llm_synthesis_time_ms;
    retrieved_chunk_count_ = metrics.retrieved_chunk_count;
    last_grag_alpha_ = metrics.grag_alpha;
    last_grag_routing_mode_ = metrics.grag_routing_mode;
    synthesis_prompt_chars_ = metrics.synthesis_prompt_chars;
    synthesis_context_truncated_ = metrics.synthesis_context_truncated;
    planning_tokens_ = metrics.planning_tokens;
    planning_tokens_at_resume_ = metrics.planning_tokens;
    token_base_prompt_ = metrics.prompt_tokens;
    token_base_completion_ = metrics.completion_tokens;
    token_base_total_ = metrics.total_tokens;
    benchmark_attribution_.run_id = metrics.run_id;
    benchmark_attribution_.env_hash = metrics.env_hash;
}

nlohmann::json ExecutiveController::goal_cognitive_metrics_payload_unlocked(
    const std::string& outcome,
    float trajectory_score) {
    sync_goal_metrics_checkpoint_unlocked();

    const std::int64_t finished = nowMs();
    const GoalMetricsCheckpoint& metrics = current_plan_.goal_metrics;
    GoalCognitiveMetricsRecord record;
    record.plan_id = current_plan_.plan_id;
    record.session_id = session_id_;
    record.goal = cleanGoalForStorage(current_plan_.goal);
    record.outcome = outcome;
    record.goal_finished_at_ms = finished;
    record.step_count = static_cast<int>(current_plan_.steps.size());
    record.trajectory_score = trajectory_score;
    record.final_success_score = trajectory_score;
    record.reflection_count = reflection_count_;
    record.revisions_count = revisions_count_;
    record.max_reflections = max_reflections_;
    record.plan_reused = plan_reused_;
    record.reflection_skip_reason = reflection_skip_reason_;
    record.metrics_partial = metrics_checkpoint_gap_ || metrics_legacy_resume_;
    if (metrics_legacy_resume_) {
        record.metrics_partial_reason = "not_checkpointed";
        record.emit_exact_counters = false;
        record.emit_available_counters = false;
        record.emit_goal_started = false;
        record.emit_wall_clock = false;
    } else if (metrics_checkpoint_gap_) {
        record.metrics_partial_reason = "checkpoint_gap";
        record.emit_exact_counters = false;
        record.emit_available_counters = true;
        record.emit_goal_started = true;
        record.emit_wall_clock = false;
        record.goal_started_at_ms = goal_started_at_ms_;
        record.planning_time_ms = metrics.planning_time_ms;
        record.retrieval_time_ms = metrics.retrieval_time_ms;
        record.llm_synthesis_time_ms = metrics.llm_synthesis_time_ms;
        record.retrieved_chunk_count = metrics.retrieved_chunk_count;
        record.total_tokens = metrics.total_tokens;
        record.prompt_tokens = metrics.prompt_tokens;
        record.completion_tokens = metrics.completion_tokens;
        record.planning_tokens = metrics.planning_tokens;
        record.synthesis_tokens = std::max<std::int64_t>(0, metrics.total_tokens - metrics.planning_tokens);
        record.synthesis_prompt_chars = metrics.synthesis_prompt_chars;
        record.c64_window_id = c64_window_id_;
        record.c64_protocol_version = c64_protocol_version_;
        record.c64_metric_schema_version = c64_metric_schema_version_;
        record.c64_environment_schema_version = c64_environment_schema_version_;
        record.c64_cohort_fingerprint = c64_cohort_fingerprint_;
    } else {
        record.emit_exact_counters = true;
        record.emit_available_counters = false;
        record.emit_goal_started = true;
        record.emit_wall_clock = true;
        record.goal_started_at_ms =
            goal_started_at_ms_ > 0 ? goal_started_at_ms_ : current_plan_.created_at_ms;
        record.total_wall_clock_ms =
            record.goal_started_at_ms > 0 ? finished - record.goal_started_at_ms : 0;
        record.planning_time_ms = metrics.planning_time_ms;
        record.retrieval_time_ms = metrics.retrieval_time_ms;
        record.llm_synthesis_time_ms = metrics.llm_synthesis_time_ms;
        record.retrieved_chunk_count = metrics.retrieved_chunk_count;
        record.grag_alpha = metrics.grag_alpha;
        record.grag_routing_mode = metrics.grag_routing_mode;
        record.total_tokens = metrics.total_tokens;
        record.prompt_tokens = metrics.prompt_tokens;
        record.completion_tokens = metrics.completion_tokens;
        record.planning_tokens = metrics.planning_tokens;
        record.synthesis_tokens = std::max<std::int64_t>(0, metrics.total_tokens - metrics.planning_tokens);
        record.synthesis_prompt_chars = metrics.synthesis_prompt_chars;
        record.synthesis_context_truncated = metrics.synthesis_context_truncated;
        record.c64_window_id = c64_window_id_;
        record.c64_protocol_version = c64_protocol_version_;
        record.c64_metric_schema_version = c64_metric_schema_version_;
        record.c64_environment_schema_version = c64_environment_schema_version_;
        record.c64_cohort_fingerprint = c64_cohort_fingerprint_;
    }
    record.plan_origin = Thoth::normalizePlanOrigin(current_plan_.plan_origin);
    record.learning_eligible = Thoth::learningEligibleOrigin(record.plan_origin);
    for (const auto& step : current_plan_.steps) {
        record.step_ids.push_back(step.step_id);
        if (record.retrieval_query.empty() && step.type == StepType::RETRIEVAL
            && step.payload.is_object() && step.payload.contains("query")
            && step.payload["query"].is_string()) {
            record.retrieval_query = step.payload["query"].get<std::string>();
        }
    }
    if (!benchmark_attribution_.empty()) {
        record.run_id = benchmark_attribution_.run_id;
        record.env_hash = benchmark_attribution_.env_hash;
    }
    if (!metrics_legacy_resume_) {
        record.c64_window_id = c64_window_id_;
        record.c64_protocol_version = c64_protocol_version_;
        record.c64_metric_schema_version = c64_metric_schema_version_;
        record.c64_environment_schema_version = c64_environment_schema_version_;
        record.c64_cohort_fingerprint = c64_cohort_fingerprint_;
    }

    nlohmann::json payload = CognitiveMetricsLogger::toJson(record);
    payload["metrics_id"] = current_plan_.plan_id;
    return payload;
}

bool ExecutiveController::flush_cognitive_metrics_outbox_unlocked() {
    if (!memory_) return false;
    const char* inject = std::getenv("THOTH_INJECT_METRICS_FLUSH");
    const std::string mode = inject ? inject : "";
    const auto pending = memory_->pendingCognitiveMetrics();
    for (const auto& row : pending) {
        if (mode == "before_append") return false;
        nlohmann::json payload;
        try {
            payload = nlohmann::json::parse(row.payload_json);
        } catch (...) {
            std::cerr << "[ExecutiveController] cognitive metrics outbox payload is not JSON; "
                         "left unflushed\n";
            return false;
        }
        if (!payload.is_object() || !payload.contains("metrics_id") || !payload["metrics_id"].is_string()
            || payload["metrics_id"].get<std::string>() != row.metrics_id) {
            std::cerr << "[ExecutiveController] cognitive metrics outbox metrics_id mismatch; "
                         "left unflushed\n";
            return false;
        }
        if (!CognitiveMetricsLogger::instance().appendIfNotDelivered(row.metrics_id, payload)) return false;
        if (mode == "after_append") return false;
        if (mode == "kill_after_append") {
            ::raise(SIGKILL);
        }
        if (!memory_->markCognitiveMetricsFlushed(row.metrics_id)) return false;
    }
    return true;
}

void ExecutiveController::emit_goal_cognitive_metrics_unlocked(const std::string& outcome,
                                                               float trajectory_score) {
    if (current_plan_.plan_id.empty() || cognitive_metrics_emitted_) {
        return;
    }
    const nlohmann::json payload = goal_cognitive_metrics_payload_unlocked(outcome, trajectory_score);
    if (memory_) {
        if (!memory_->commitCognitiveMetrics(current_plan_.plan_id, payload.dump())) return;
        if (!flush_cognitive_metrics_outbox_unlocked()) return;
    } else if (!CognitiveMetricsLogger::instance().appendStoredRecord(payload)) {
        return;
    }
    cognitive_metrics_emitted_ = true;
    StructuredLogger::instance().log(LogLevel::Info,
                                     "controller",
                                     "GOAL_COGNITIVE_METRICS",
                                     "Per-goal cognitive metrics recorded",
                                     payload);
}

namespace {

int countStepsFromOutline(const std::string& outline) {
    try {
        auto j = nlohmann::json::parse(outline);
        if (j.contains("steps") && j["steps"].is_array()) {
            return static_cast<int>(j["steps"].size());
        }
    } catch (...) {
    }
    return 0;
}

std::string truncateOutline(const std::string& outline) {
    if (outline.size() <= Thoth::PlanReuse::kOutlineMaxChars) return outline;
    return outline.substr(0, Thoth::PlanReuse::kOutlineMaxChars) + "...";
}

nlohmann::json pastPlansToJson(const std::vector<Memory::PastPlanRecord>& plans) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& p : plans) {
        arr.push_back({
            {"plan_id", p.plan_id},
            {"goal", p.goal},
            {"success_score", p.success_score},
            {"step_count", countStepsFromOutline(p.outline)},
            {"outline_preview", truncateOutline(p.outline)},
            {"plan_origin", Thoth::normalizePlanOrigin(p.plan_origin)},
            {"learning_eligible", Thoth::learningEligibleOrigin(p.plan_origin)}
        });
    }
    return arr;
}

} // namespace

std::string ExecutiveController::build_plan_reuse_context(const std::vector<Memory::PastPlanRecord>& plans) const {
    std::ostringstream oss;
    bool any = false;
    for (const auto& p : plans) {
        if (!Thoth::learningEligibleOrigin(p.plan_origin)) {
            continue;
        }
        if (!any) {
            oss << "\n\n[RELEVANT PAST APPROACHES — similar goals, prior success >= "
                << Thoth::PlanReuse::kMinSuccessScore << "]\n";
            any = true;
        }
        oss << "- Plan ID: " << p.plan_id << "\n";
        oss << "  Goal: " << Thoth::cleanGoalForStorage(p.goal) << "\n";
        oss << "  Success: " << std::fixed << std::setprecision(2) << p.success_score;
        oss << " | Steps: " << countStepsFromOutline(p.outline) << "\n";
        oss << "  Outline: " << Thoth::sanitizePlanOutline(p.outline) << "\n";
    }
    return oss.str();
}

nlohmann::json ExecutiveController::log_plan_reuse_injection(
    const std::vector<Memory::PastPlanRecord>& plans,
    const std::string& source) {
    nlohmann::json meta = {
        {"source", source},
        {"plan_count", plans.size()},
        {"min_success_score", Thoth::PlanReuse::kMinSuccessScore},
        {"success_boost_threshold", Thoth::PlanReuse::kSuccessBoostThreshold},
        {"success_boost", Thoth::PlanReuse::kSuccessBoost},
        {"retrieve_limit", Thoth::PlanReuse::kDefaultRetrieveLimit},
        {"plans", pastPlansToJson(plans)}
    };

    StructuredLogger::instance().log(
        LogLevel::Info,
        "executive",
        "PLAN_REUSE_INJECTION",
        "Injected similar past plans into planner context",
        meta,
        "",
        session_id_);

    return meta;
}

nlohmann::json ExecutiveController::log_reflection_replan(float score, int reflection_cycle) {
    nlohmann::json meta = {
        {"trajectory_score", score},
        {"reflection_threshold", Thoth::Reflection::kScoreThreshold},
        {"reflection_cycle", reflection_cycle},
        {"max_reflections", max_reflections_},
        {"plan_id", current_plan_.plan_id}
    };

    StructuredLogger::instance().log(
        LogLevel::Info,
        "executive",
        "REFLECTION_REPLAN",
        "Low trajectory score triggered reflection replan",
        meta,
        "",
        session_id_);

    return meta;
}

nlohmann::json ExecutiveController::log_plan_history_persisted(float success_score, const Memory::PastPlanRecord& past_record) {
    nlohmann::json meta = {
        {"plan_id", past_record.plan_id},
        {"success_score", success_score},
        {"failure_count", past_record.failure_count},
        {"duration_ms", past_record.duration_ms},
        {"storage_targets", nlohmann::json::array({"past_plans", "cognate_plans", "trajectories"})},
        {"past_plans_table", "past_plans"},
        {"cognate_plans_table", "cognate_plans"},
        {"embedding_version", 2},
        {"min_success_for_reuse", Thoth::PlanReuse::kMinSuccessScore},
        {"plan_origin", Thoth::normalizePlanOrigin(past_record.plan_origin)},
        {"learning_eligible", Thoth::learningEligibleOrigin(past_record.plan_origin)},
        {"trajectory_score", success_score}
    };

    StructuredLogger::instance().log(
        LogLevel::Info,
        "executive",
        "PLAN_HISTORY_STORED",
        "Persisted completed plan to past_plans and cognate_plans",
        meta,
        "",
        session_id_);

    return meta;
}

} // namespace Thoth
