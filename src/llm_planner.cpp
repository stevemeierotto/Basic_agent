/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/llm_planner.h"
#include "../include/logger.h"
#include "../include/plan_parser.h"
#include "../include/plan_reuse_config.h"
#include "../include/goal_text_utils.h"
#include <chrono>
#include <sstream>
#include <iomanip>

LLMPlanner::LLMPlanner(std::shared_ptr<Memory> memory, 
                       std::shared_ptr<RAGPipeline> rag, 
                       std::shared_ptr<PromptFactory> prompt_factory,
                       LLMInterface* llm) 
    : memory_(memory), rag_(rag), prompt_factory_(prompt_factory), llm_(llm) {}

static int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string LLMPlanner::generate_uuid() {
    static int counter = 0;
    std::ostringstream oss;
    oss << "plan-" << nowMs() << "-" << ++counter;
    return oss.str();
}

Plan LLMPlanner::create_plan(const std::string& goal) {
    Plan plan;
    plan.plan_id = generate_uuid();
    plan.goal = goal;
    plan.created_at_ms = nowMs();
    plan.updated_at_ms = plan.created_at_ms;
    plan.status = PlanStatus::ACTIVE;

    if (!prompt_factory_ || !llm_) {
        plan.status = PlanStatus::FAILED;
        return plan;
    }

    auto [prompt_goal, reuse_block] = Thoth::splitPlanReuseInjection(goal);

    // Phase 3.2: Gather context for the prompt (Cognate V2)
    std::string strategy_context;
    std::string past_experience;
    std::size_t trajectory_count = 0;
    std::size_t strategy_count = 0;
    
    if (rag_ && rag_->engine && memory_) {
        auto goal_embedding = rag_->engine->embed(prompt_goal);
        
        // 1. Past Trajectories (Experience-Guided Planning)
        auto trajectories = memory_->retrieveSimilarTrajectories(goal_embedding, 3);
        if (!trajectories.empty()) {
            trajectory_count = trajectories.size();
            std::ostringstream traj_oss;
            traj_oss << "[PAST EXPERIENCE - RELEVANT TRAJECTORIES]\n";
            for (const auto& t : trajectories) {
                traj_oss << "- Goal: " << t.goal << "\n  Trajectory: " << t.trajectory_json << "\n\n";
            }
            past_experience = traj_oss.str();
            
            StructuredLogger::instance().log(LogLevel::Info, "planner", "TRAJECTORY_INJECTION", 
                "Injected " + std::to_string(trajectories.size()) + " trajectories as prior experience", 
                {{"goal", goal}, {"trajectory_count", trajectories.size()}});
        }

        // 2. Emerged Strategies (The Learned proof)
        // Thesis Differentiator: Prioritize strategies promoted by 80%/3-run threshold
        auto strats = memory_->getAllStrategies();
        if (!strats.empty()) {
            strategy_count = strats.size();
            std::ostringstream strat_oss;
            strat_oss << "[LEARNED STRATEGIES - HIGH SUCCESS PATTERNS]\n";
            for (const auto& s : strats) {
                strat_oss << "- Strategy ID: " << s.strategy_id << "\n";
                strat_oss << "  Description: " << s.description << "\n";
                strat_oss << "  Pattern: " << s.step_pattern_json << "\n";
                strat_oss << "  Historical Success Rate: " << (s.success_rate * 100.0f) << "%\n\n";
            }
            strategy_context = strat_oss.str();

            StructuredLogger::instance().log(LogLevel::Info, "planner", "STRATEGY_INJECTION", 
                "Injected " + std::to_string(strats.size()) + " learned strategies into prompt", 
                {{"goal", goal}, {"strategy_count", strats.size()}});
        }
    }

    if (!reuse_block.empty()) {
        if (!past_experience.empty()) {
            past_experience += "\n\n";
        }
        past_experience += reuse_block;
    }

    const bool plan_reuse_in_goal = !reuse_block.empty();

    StructuredLogger::instance().log(
        LogLevel::Info,
        "planner",
        "PLANNER_CONTEXT_ASSEMBLY",
        "Assembled planner prompt context from memory subsystems",
        {
            {"goal", prompt_goal},
            {"trajectory_injection", trajectory_count > 0},
            {"trajectory_count", trajectory_count},
            {"trajectory_min_episode_steps", Thoth::TrajectoryReuse::kMinEpisodeStepsForEmbedding},
            {"strategy_injection", strategy_count > 0},
            {"strategy_count", strategy_count},
            {"plan_reuse_in_goal", plan_reuse_in_goal},
            {"plan_reuse_marker", "[RELEVANT PAST APPROACHES"},
            {"past_plans_table", plan_reuse_in_goal ? "past_plans (injected by ExecutiveController)" : "not injected"},
            {"cognate_plans_table", "cognate_plans (active plan snapshots via save_plan)"}
        });

    std::string prompt = prompt_factory_->buildPlanPrompt(prompt_goal, strategy_context, past_experience);
    std::string llm_response = llm_->query(prompt);

    auto parsed_plan = Thoth::PlanParser::parse(llm_response, plan.plan_id);
    
    // Step 9.4 Retry logic
    if (!parsed_plan.has_value()) {
        StructuredLogger::instance().log(LogLevel::Warn, "planner", "plan_parse_failed", "First plan parsing attempt failed, retrying...", {{"llm_response", llm_response}});
        
        std::string retry_prompt = prompt + "\n\nERROR: Your previous response was not a valid JSON plan. Please correct it and follow the schema exactly.\nPrevious Response:\n" + llm_response;
        llm_response = llm_->query(retry_prompt);
        parsed_plan = Thoth::PlanParser::parse(llm_response, plan.plan_id);
    }

    if (parsed_plan.has_value()) {
        plan = parsed_plan.value();
        plan.goal = prompt_goal;
        plan.created_at_ms = nowMs();
        plan.updated_at_ms = plan.created_at_ms;
        plan.status = PlanStatus::ACTIVE;

        StructuredLogger::instance().log(
            LogLevel::Info,
            "planner",
            "plan_generated",
            "LLMPlanner created a dynamic multi-step plan",
            {{"plan_id", plan.plan_id}, {"step_count", plan.steps.size()}});
    } else {
        plan.status = PlanStatus::FAILED;
        StructuredLogger::instance().log(LogLevel::Error, "planner", "plan_failed", "Plan generation failed after retry", {{"goal", prompt_goal}});
    }

    save_plan(plan);
    return plan;
}

Plan LLMPlanner::revise_plan(const Plan& existing_plan,
                             const nlohmann::json& step_result) {
    if (!prompt_factory_ || !llm_) {
        Plan revised = existing_plan;
        revised.updated_at_ms = nowMs();
        return revised;
    }

    auto [prompt_goal, reuse_block] = Thoth::splitPlanReuseInjection(existing_plan.goal);
    std::string revision_goal = prompt_goal;
    if (!reuse_block.empty()) {
        revision_goal += "\n\n";
        revision_goal += reuse_block;
    }

    std::string prompt = prompt_factory_->buildRevisionPrompt(revision_goal, 
                                                               existing_plan.to_json().dump(), 
                                                               step_result.dump());
    
    std::string llm_response = llm_->query(prompt);

    auto parsed_plan = Thoth::PlanParser::parse(llm_response, existing_plan.plan_id);
    
    // Retry logic
    if (!parsed_plan.has_value()) {
        StructuredLogger::instance().log(LogLevel::Warn, "planner", "revision_parse_failed", "First plan revision attempt failed, retrying...", {{"llm_response", llm_response}});
        
        std::string retry_prompt = prompt + "\n\nERROR: Your previous response was not a valid JSON plan. Please correct it and follow the schema exactly.\nPrevious Response:\n" + llm_response;
        llm_response = llm_->query(retry_prompt);
        parsed_plan = Thoth::PlanParser::parse(llm_response, existing_plan.plan_id);
    }

    if (parsed_plan.has_value()) {
        Plan revised = parsed_plan.value();
        revised.goal = prompt_goal;
        revised.created_at_ms = existing_plan.created_at_ms; // Maintain creation time
        revised.updated_at_ms = nowMs();
        revised.status = PlanStatus::ACTIVE;
        
        // Reset current_index if the new plan starts from scratch or a new state
        // For now, we assume the LLM generates the REMAINING steps or a FULL new plan.
        // We set index to 0 for the new plan structure.
        revised.current_index = 0;

        StructuredLogger::instance().log(
            LogLevel::Info,
            "planner",
            "plan_revised",
            "LLMPlanner revised the existing plan",
            {{"plan_id", revised.plan_id}, {"step_count", revised.steps.size()}});

        save_plan(revised);
        return revised;
    } else {
        StructuredLogger::instance().log(LogLevel::Error, "planner", "revision_failed", "Plan revision failed after retry", {{"plan_id", existing_plan.plan_id}});
        Plan revised = existing_plan;
        revised.updated_at_ms = nowMs();
        save_plan(revised);
        return revised;
    }
}

void LLMPlanner::save_plan(const Plan& plan) {
    if (!memory_) return;

    Memory::CognatePlanRecord rec;
    rec.plan_id = plan.plan_id;
    rec.goal = plan.goal;
    rec.plan_json = plan.to_json().dump();
    rec.status = static_cast<int>(plan.status);
    rec.success_score = 0.0f; // Default for new/active plans
    rec.created_at = plan.created_at_ms;
    rec.updated_at = plan.updated_at_ms;

    // Generate goal embedding (Phase 3.1)
    if (rag_ && rag_->engine) {
        rec.embedding = rag_->engine->embed(plan.goal);
    }

    memory_->saveCognatePlan(rec);

    StructuredLogger::instance().log(
        LogLevel::Info,
        "planner",
        "COGNATE_PLAN_PERSISTED",
        "Saved plan snapshot to cognate_plans table",
        {
            {"plan_id", rec.plan_id},
            {"table", "cognate_plans"},
            {"step_count", plan.steps.size()},
            {"has_embedding", !rec.embedding.empty()}
        });
}

