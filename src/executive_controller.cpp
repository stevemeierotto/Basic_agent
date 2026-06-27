/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Implementation (Parallel Engine v1.0)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/executive_controller.h"
#include "../include/llm_interface.h"
#include "../include/logger.h"
#include "../include/grag_scorer.h"
#include "../include/memory.h"
#include "../include/plan_reuse_config.h"
#include "../include/grag_metrics.h"
#include "../include/step_metrics_repository.h"
#include "../include/file_handler.h"
#include "../include/goal_text_utils.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace Thoth {

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
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

ExecutiveController::ExecutiveController(
    std::shared_ptr<IPlanner> planner,
    std::shared_ptr<ToolRegistry> tool_registry,
    std::shared_ptr<RAGPipeline> rag,
    std::shared_ptr<Memory> memory
) : planner_(planner), tool_registry_(tool_registry), rag_(rag), memory_(memory) {
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
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
        thread_to_join = std::move(loop_thread_);
    }
    if (thread_to_join && thread_to_join->joinable()) {
        thread_to_join->join();
    }
}

void ExecutiveController::set_llm_interface(LLMInterface* llm) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (workflow_engine_) {
        workflow_engine_->setLLMInterface(llm);
    }
}

std::string ExecutiveController::execute_goal(const std::string& goal) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (loop_thread_) {
            stop_requested_ = true;
            mutex_.unlock(); 
            if (loop_thread_->joinable()) loop_thread_->join();
            mutex_.lock();
            stop_requested_ = false;
        }

        revisions_count_ = 0;
        reflection_count_ = 0;
        plan_reused_ = false;
        transition_to_unlocked(ControllerState::PLANNING);
        
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

        current_plan_ = planner_->create_plan(enhanced_goal);
        current_plan_.created_at_ms = nowMs();
        current_plan_.updated_at_ms = current_plan_.created_at_ms;
        
        // Initialize Trajectory (Phase 7.3)
        current_trajectory_ = Trajectory();
        current_trajectory_.trajectory_id = "traj-" + current_plan_.plan_id;
        current_trajectory_.goal = goal;
        current_trajectory_.plan_initial = current_plan_;
        current_trajectory_.created_at = current_plan_.created_at_ms;
        current_trajectory_.embedding = goal_embedding_;

        update_current_embedding_unlocked();
        persist_current_plan_unlocked();

        // Start the execution loop
        state_ = ControllerState::IDLE;
        running_ = true;
        loop_thread_ = std::make_unique<std::thread>([this]() {
            run_loop();
        });
    }

    if (!plan_reuse_meta.is_null()) {
        emit_event(EventType::PLAN_REUSE_INJECTION, "", plan_reuse_meta);
    }
    
    emit_event(EventType::PLAN_CREATED);
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
        stop_requested_ = true;
        current_plan_.updated_at_ms = nowMs();
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
        std::lock_guard<std::mutex> lock(mutex_);
        if (loop_thread_) {
            stop_requested_ = true;
            mutex_.unlock();
            if (loop_thread_->joinable()) loop_thread_->join();
            mutex_.lock();
            stop_requested_ = false;
        }

        current_plan_ = plan;
        current_plan_.updated_at_ms = nowMs();
        
        // Restore session ID if it's not set but exists in the plan's context
        // (Note: active_plans table now has session_id, but the Plan struct doesn't yet have it as a direct member)
        // For now, we rely on the controller already having session_id set by the AgentInterface.
        
        update_goal_embedding_unlocked(current_plan_.goal);
        update_current_embedding_unlocked();
        persist_current_plan_unlocked();
        
        state_ = ControllerState::IDLE;
        running_ = true;
        loop_thread_ = std::make_unique<std::thread>([this]() {
            run_loop();
        });
    }
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
        update_current_embedding_unlocked();

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
            float score = calculate_trajectory_score(all_successful);
            
            // Step 6.3: Reflection Loop
            if (score < Reflection::kScoreThreshold && reflection_count_ < MAX_REFLECTIONS) {
                std::cout << "[DEBUG] Low success score (" << score << "), triggering reflection cycle " << (reflection_count_ + 1) << "\n";
                reflection_count_++;

                auto reflection_meta = log_reflection_replan(score, reflection_count_);
                
                store_plan_history(score);
                transition_to_unlocked(ControllerState::PLANNING);
                
                // Re-run planning with context
                std::string reflection_goal = current_plan_.goal + " (Reflection: previous attempt had low success score " + std::to_string(score) + ")";
                nlohmann::json plan_reuse_meta;

                if (memory_ && rag_) {
                    auto past_plans = memory_->retrieveSimilarPlans(goal_embedding_, Thoth::PlanReuse::kDefaultRetrieveLimit);
                    if (!past_plans.empty()) {
                        reflection_goal += build_plan_reuse_context(past_plans);
                        plan_reused_ = true;
                        plan_reuse_meta = log_plan_reuse_injection(past_plans, "reflection_replan");
                    }
                }
                
                // We must unlock to call planner
                lock.unlock();
                emit_event(EventType::REFLECTION_REPLAN, "", reflection_meta);
                if (!plan_reuse_meta.is_null()) {
                    emit_event(EventType::PLAN_REUSE_INJECTION, "", plan_reuse_meta);
                }
                auto new_plan = planner_->create_plan(reflection_goal);
                lock.lock();
                
                current_plan_ = new_plan;
                current_plan_.created_at_ms = nowMs();
                current_plan_.updated_at_ms = current_plan_.created_at_ms;
                
                // Reset trajectory for new attempt
                current_trajectory_ = Trajectory();
                current_trajectory_.trajectory_id = "traj-" + current_plan_.plan_id;
                current_trajectory_.goal = reflection_goal;
                current_trajectory_.plan_initial = current_plan_;
                current_trajectory_.created_at = current_plan_.created_at_ms;
                current_trajectory_.embedding = goal_embedding_;

                persist_current_plan_unlocked();
                transition_to_unlocked(ControllerState::IDLE);
                
                lock.unlock();
                emit_event(EventType::PLAN_CREATED);
                return;
            }

            transition_to_unlocked((all_successful && !current_plan_.steps.empty()) ? ControllerState::COMPLETED : ControllerState::FAILED);
            current_plan_.updated_at_ms = nowMs();
            
            auto history_meta = store_plan_history(score);
            if (memory_) memory_->deleteActivePlan(current_plan_.plan_id);
            
            lock.unlock();
            if (!history_meta.is_null()) {
                emit_event(EventType::PLAN_HISTORY_STORED, "", history_meta);
            }
            emit_event((all_successful && !current_plan_.steps.empty()) ? EventType::PLAN_COMPLETED : EventType::PLAN_FAILED);
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
                    persist_current_plan_unlocked();
                    continue; // Check next step
                }

                step.status = StepStatus::RUNNING;
                step.started_at_ms = nowMs();
                steps_to_start.push_back(step);
            }
        }

        if (!steps_to_start.empty()) {
            transition_to_unlocked(ControllerState::EXECUTING_STEP);
            current_plan_.updated_at_ms = nowMs();
            persist_current_plan_unlocked();
        }

        plan_id_for_dispatch = current_plan_.plan_id;
        execution_context = buildStepExecutionContext(current_plan_);
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
    step.completed_at_ms = nowMs();
    current_plan_.updated_at_ms = step.completed_at_ms;

    // Record Step in Trajectory (Phase 7.3)
    record_trajectory_step(step, result);

    // Phase 5.5: Store Episode Step for semantic trajectory retrieval
    if (memory_) {
        Memory::EpisodeStepRecord ep;
        ep.episode_id = "traj-" + current_plan_.plan_id;
        ep.goal_id = current_plan_.plan_id;
        ep.step_index = static_cast<int>(std::distance(current_plan_.steps.begin(), it));
        ep.state_summary = step.description;
        ep.action_taken = step.tool.dump();
        ep.result_status = result.success ? "SUCCESS" : "FAILED";
        ep.timestamp_ms = nowMs();
        // We could embed the state here too, but for now we embed the full summary in TrajectoryBuilder
        memory_->storeEpisodeStep(ep);
    }

    transition_to_unlocked(ControllerState::OBSERVING_RESULT);

    bool success = result.success;

    if (success) {
        current_plan_.updated_at_ms = nowMs();
        lock.unlock(); 
        emit_event(EventType::STEP_COMPLETED, step.step_id);
        
        lock.lock();
        update_current_embedding_unlocked();
        
        // Phase 5.6: Capture Active Set from RETRIEVAL result
        if (step.type == StepType::RETRIEVAL && step.result.contains("diagnostics")) {
            try {
                auto diag = step.result["diagnostics"];
                if (diag.contains("breakdowns") && diag["breakdowns"].is_array()) {
                    auto breakdowns = diag["breakdowns"];
                    
                    float bestScore = 0.0f;
                    if (!breakdowns.empty()) {
                        bestScore = breakdowns[0].value("final_score", 0.0f);
                    }

                    ActiveStepSet currentSet;
                    currentSet.step_id = step.step_id;

                    for (const auto& b : breakdowns) {
                        float score = b.value("final_score", 0.0f);
                        // Take chunks where final_score >= 0.8 * best_score, cap at 5
                        if (score >= 0.8f * bestScore && currentSet.chunk_hashes.size() < 5) {
                            std::string content = b.value("code_text", ""); // Need to ensure code_text is in breakdown
                            if (content.empty()) continue;

                            std::string hash = Memory::calculateContentHash(content);
                            currentSet.chunk_hashes.push_back(hash);

                            // Register node if it doesn't exist
                            Memory::Node node;
                            node.id = hash;
                            node.file_path = b.value("file_name", "");
                            node.symbol = b.value("symbol", "");
                            node.type = "code_chunk";
                            memory_->addNode(node);
                        }
                    }
                    if (!currentSet.chunk_hashes.empty()) {
                        active_sets_.push_back(currentSet);
                    }
                }
            } catch (...) {}
        }

        update_trajectory_embedding_unlocked();
        persist_current_plan_unlocked();
        
        transition_to_unlocked(ControllerState::IDLE);
    } else {
        current_plan_.updated_at_ms = nowMs();
        std::string err = result.error_message;
        lock.unlock();
        emit_event(EventType::STEP_FAILED, step.step_id, {{"error", err}});

        lock.lock();
        if (step.failure_policy.revise_plan_on_failure) {
            transition_to_unlocked(ControllerState::REVISING_PLAN);
            current_plan_.updated_at_ms = nowMs();
            persist_current_plan_unlocked();
            
            lock.unlock();
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
            transition_to_unlocked(ControllerState::FAILED);
            current_plan_.updated_at_ms = nowMs();
            if (memory_) memory_->deleteActivePlan(current_plan_.plan_id);
            lock.unlock();
            emit_event(EventType::PLAN_FAILED);
        } else {
            transition_to_unlocked(ControllerState::IDLE);
            persist_current_plan_unlocked();
        }
    }
}

nlohmann::json ExecutiveController::dispatch_step(PlanStep& step) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!workflow_engine_) {
        return {{"status", "error"}, {"error_message", "WorkflowEngine not available"}};
    }
    const StepExecutionContext context = buildStepExecutionContext(current_plan_);
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
                    }
                    if (type == EventType::STEP_FAILED) {
                        enriched_meta["success"] = false;
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

void ExecutiveController::reinforce_plan_graph() {
    if (!memory_ || !graph_refiner_ || current_plan_.plan_id.empty() || active_sets_.size() < 2) return;

    // Phase 5.6: Causal Edge Generation
    // Identify all edges in this trajectory for the refiner
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
                
                // Fetch existing if present to preserve counts
                auto existing = memory_->getEdgesFrom(fromHash);
                bool found = false;
                for (const auto& e : existing) {
                    if (e.to_id == toHash) {
                        edge = e;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    edge.weight = 0.1f; // Initial weight
                    edge.success_count = 0;
                    edge.failure_count = 0;
                }
                edge.last_used_ms = nowMs();
                trajectory_edges.push_back(edge);
            }
        }
    }

    if (!trajectory_edges.empty()) {
        float success_score = (current_plan_.status == PlanStatus::COMPLETED) ? 1.0f : 0.0f;
        graph_refiner_->refineFromTrajectory(trajectory_edges, success_score);
    }
    
    active_sets_.clear(); // Reset for next plan
}

void ExecutiveController::record_trajectory_step(const PlanStep& step, const Thoth::StepResult& result) {
    RecordedStep recorded;
    recorded.step_id = step.step_id;
    recorded.description = step.description;
    recorded.tool = step.tool;
    recorded.result = result.data;
    recorded.error = result.error_message;
    recorded.timestamp = nowMs();
    current_trajectory_.steps.push_back(recorded);
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

void ExecutiveController::persist_current_plan_unlocked() {
    if (!memory_ || current_plan_.plan_id.empty()) return;

    MemoryRepository::ActivePlanRecord rec;
    rec.plan_id = current_plan_.plan_id;
    rec.session_id = session_id_;
    rec.goal = current_plan_.goal;
    rec.steps_json = current_plan_.to_json().dump();
    rec.current_index = static_cast<int>(current_plan_.current_index);
    rec.controller_state = state_to_name(state_);
    rec.created_at_ms = current_plan_.created_at_ms;
    rec.updated_at_ms = current_plan_.updated_at_ms;
    memory_->storeActivePlan(rec);

    MemoryRepository::CognatePlanRecord cognateRec;
    cognatePlanRecordFromPlan(current_plan_, cognateRec);
    cognateRec.status = static_cast<int>(state_);
    memory_->saveCognatePlan(cognateRec);
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
    state["goal"] = current_plan_.goal;
    
    state["completed_steps_summary"] = "none";
    state["remaining_steps_summary"] = "none";
    state["constraints"] = "none";
    state["known_blockers"] = "none";

    std::string state_str = state.dump();
    current_embedding_ = rag_->engine->embed(state_str);
    
    if (current_embedding_.empty()) {
        emit_event(EventType::EMBEDDING_FAILED, "", {{"type", "state"}, {"input", state_str}});
    }

    rag_->setCurrentEmbedding(current_embedding_);
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
            else rag_->getRetrievalConfig().wt = 0.2f;
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

nlohmann::json ExecutiveController::store_plan_history(float success_score) {
    if (!memory_ || current_plan_.plan_id.empty()) return nlohmann::json();

    Memory::PastPlanRecord record;
    record.plan_id = current_plan_.plan_id;
    record.goal = current_plan_.goal;
    record.outline = current_plan_.to_json().dump();
    record.success_score = success_score;
    record.duration_ms = nowMs() - current_plan_.created_at_ms;
    record.failure_count = 0;
    for (const auto& step : current_plan_.steps) {
        record.failure_count += step.retry_count;
    }
    record.goal_embedding = goal_embedding_;

    memory_->storePastPlan(record);

    nlohmann::json history_meta = log_plan_history_persisted(success_score, record);

    // Phase 5.6: Reinforce Graph Memory
    reinforce_plan_graph();
    
    MemoryRepository::CognatePlanRecord cognateRec;
    cognatePlanRecordFromPlan(current_plan_, cognateRec);
    cognateRec.status = static_cast<int>(state_);
    cognateRec.success_score = success_score;
    memory_->saveCognatePlan(cognateRec);

    current_trajectory_.final_status = current_plan_.status;
    current_trajectory_.success_score = success_score;
    current_trajectory_.results = current_plan_.steps.empty() ? nlohmann::json::object() : current_plan_.steps.back().result;
    
    Memory::CognateTrajectoryRecord trajRec;
    trajRec.trajectory_id = current_trajectory_.trajectory_id;
    trajRec.goal = current_trajectory_.goal;
    trajRec.trajectory_json = current_trajectory_.to_json().dump();
    trajRec.success_score = success_score;
    trajRec.embedding = current_trajectory_.embedding;
    trajRec.created_at = current_trajectory_.created_at;
    memory_->saveTrajectory(trajRec);
    
    memory_->processMemoryAging();
    
    if (strategy_engine_) {
        strategy_engine_->processTrajectories();
    }
    
    if (success_score > 0.0f) {
        reinforce_plan_graph();
    }
    
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
            {"outline_preview", truncateOutline(p.outline)}
        });
    }
    return arr;
}

} // namespace

std::string ExecutiveController::build_plan_reuse_context(const std::vector<Memory::PastPlanRecord>& plans) const {
    std::ostringstream oss;
    oss << "\n\n[RELEVANT PAST APPROACHES — similar goals, prior success >= "
        << Thoth::PlanReuse::kMinSuccessScore << "]\n";
    for (const auto& p : plans) {
        oss << "- Plan ID: " << p.plan_id << "\n";
        oss << "  Goal: " << p.goal << "\n";
        oss << "  Success: " << std::fixed << std::setprecision(2) << p.success_score;
        oss << " | Steps: " << countStepsFromOutline(p.outline) << "\n";
        oss << "  Outline: " << truncateOutline(p.outline) << "\n";
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
        {"max_reflections", MAX_REFLECTIONS},
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
        {"min_success_for_reuse", Thoth::PlanReuse::kMinSuccessScore}
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
