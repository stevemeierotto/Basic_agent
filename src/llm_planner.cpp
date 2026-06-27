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
#include "../include/plan_validator.h"
#include "../include/plan_reuse_config.h"
#include "../include/goal_text_utils.h"
#include "../include/planner_injection_config.h"
#include "../include/grag_scorer.h"
#include "../include/embedding_engine.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <optional>

namespace {

std::optional<Memory::CognateStrategyRecord> selectRelevantStrategy(
    Memory& memory,
    EmbeddingEngine& engine,
    const std::vector<float>& goal_embedding) {
    auto strategies = memory.getAllStrategies();
    if (strategies.empty() || goal_embedding.empty()) {
        return std::nullopt;
    }

    float bestScore = -1.0f;
    std::optional<Memory::CognateStrategyRecord> best;
    for (const auto& strategy : strategies) {
        const std::string embedText = strategy.description + " " + strategy.step_pattern_json;
        const auto strategyEmbedding = engine.embed(embedText);
        if (strategyEmbedding.empty()) {
            continue;
        }
        const float score = GragScorer::cosine_similarity(goal_embedding, strategyEmbedding);
        if (score > bestScore) {
            bestScore = score;
            best = strategy;
        }
    }

    if (!best.has_value() || bestScore < Thoth::PlannerInjection::kMinStrategySimilarity) {
        return std::nullopt;
    }
    return best;
}

std::string formatStrategyContext(const Memory::CognateStrategyRecord& strategy, float similarity) {
    std::ostringstream oss;
    oss << "[RELEVANT STRATEGY — similarity " << std::fixed << std::setprecision(2) << similarity << "]\n";
    oss << "- Strategy ID: " << strategy.strategy_id << "\n";
    oss << "  Description: " << strategy.description << "\n";
    oss << "  Pattern: " << strategy.step_pattern_json << "\n";
    oss << "  Historical Success Rate: " << (strategy.success_rate * 100.0f) << "%\n";
    return oss.str();
}

std::optional<Plan> parsePlanWithValidation(const std::string& llm_response,
                                            const std::string& plan_id,
                                            const std::string& prompt_goal,
                                            bool allow_tool_steps,
                                            bool& depends_on_repaired,
                                            bool& fallback_used,
                                            std::string& validation_reason) {
    depends_on_repaired = false;
    fallback_used = false;

    auto parsed = Thoth::PlanParser::parse(llm_response, plan_id);
    if (!parsed.has_value()) {
        validation_reason = "JSON parse failed";
        return std::nullopt;
    }

    Plan candidate = parsed.value();
    candidate.goal = prompt_goal;

    auto validation = Thoth::PlanValidator::validateAndRepair(candidate, allow_tool_steps);
    depends_on_repaired = validation.depends_on_repaired;
    if (validation.valid) {
        validation_reason = validation.depends_on_repaired ? "depends_on wired" : "ok";
        return candidate;
    }

    validation_reason = validation.reason;
    return std::nullopt;
}

Plan finalizePlanOrFallback(const std::string& plan_id,
                            const std::string& prompt_goal,
                            std::optional<Plan> validated,
                            bool& fallback_used) {
    if (validated.has_value()) {
        Plan plan = validated.value();
        plan.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
        plan.updated_at_ms = plan.created_at_ms;
        plan.status = PlanStatus::ACTIVE;
        return plan;
    }

    fallback_used = true;
    Plan plan = Thoth::PlanValidator::createFallbackPlan(plan_id, prompt_goal);
    plan.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    plan.updated_at_ms = plan.created_at_ms;
    return plan;
}

} // namespace

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

    std::string strategy_context;
    std::string past_experience;
    std::size_t trajectory_count = 0;
    float strategy_similarity = 0.0f;
    bool strategy_injected = false;

    if (rag_ && rag_->engine && memory_) {
        const auto goal_embedding = rag_->engine->embed(prompt_goal);

        const auto strategy = selectRelevantStrategy(*memory_, *rag_->engine, goal_embedding);
        if (strategy.has_value()) {
            const std::string embedText = strategy->description + " " + strategy->step_pattern_json;
            strategy_similarity = GragScorer::cosine_similarity(
                goal_embedding, rag_->engine->embed(embedText));
            strategy_context = formatStrategyContext(*strategy, strategy_similarity);
            strategy_injected = true;

            StructuredLogger::instance().log(LogLevel::Info, "planner", "STRATEGY_INJECTION",
                "Injected top-1 scored strategy into planner prompt",
                {{"goal", prompt_goal},
                 {"strategy_id", strategy->strategy_id},
                 {"similarity", strategy_similarity},
                 {"min_similarity", Thoth::PlannerInjection::kMinStrategySimilarity}});
        }

        auto trajectories = memory_->retrieveSimilarTrajectories(
            goal_embedding, Thoth::PlannerInjection::kMaxTrajectoryInject);
        if (!trajectories.empty()) {
            trajectory_count = trajectories.size();
            std::ostringstream traj_oss;
            traj_oss << "[PAST EXPERIENCE — RELEVANT TRAJECTORY]\n";
            const auto& t = trajectories.front();
            traj_oss << "- Goal: " << Thoth::cleanGoalForStorage(t.goal) << "\n";
            traj_oss << "  Success: " << t.success_score << "\n";
            past_experience = traj_oss.str();

            StructuredLogger::instance().log(LogLevel::Info, "planner", "TRAJECTORY_INJECTION",
                "Injected scored trajectory into planner prompt",
                {{"goal", prompt_goal}, {"trajectory_count", trajectory_count}});
        }
    }

    if (!reuse_block.empty()) {
        const std::string cappedReuse =
            Thoth::capInjectionText(reuse_block, Thoth::PlannerInjection::kMaxPlanReuseChars);
        if (!past_experience.empty()) {
            past_experience += "\n\n";
        }
        past_experience += cappedReuse;
    }

    const bool plan_reuse_in_goal = !reuse_block.empty();

    Thoth::PlannerPromptMetrics prompt_metrics;
    std::string prompt = prompt_factory_->buildPlanPrompt(
        prompt_goal, strategy_context, past_experience, &prompt_metrics);
    if (plan_reuse_in_goal) {
        prompt_metrics.plan_reuse_bytes = reuse_block.size();
    }

    StructuredLogger::instance().log(
        LogLevel::Info,
        "planner",
        "PLANNER_CONTEXT_ASSEMBLY",
        "Assembled planner prompt with protected core sections",
        {
            {"goal", prompt_goal},
            {"rules_bytes", prompt_metrics.rules_bytes},
            {"schema_bytes", prompt_metrics.schema_bytes},
            {"goal_bytes", prompt_metrics.goal_bytes},
            {"strategy_bytes", prompt_metrics.strategy_bytes},
            {"trajectory_bytes", prompt_metrics.trajectory_bytes},
            {"plan_reuse_bytes", prompt_metrics.plan_reuse_bytes},
            {"total_bytes", prompt_metrics.total_bytes},
            {"experience_dropped", prompt_metrics.experience_dropped},
            {"strategy_injection", strategy_injected},
            {"strategy_similarity", strategy_similarity},
            {"trajectory_injection", trajectory_count > 0},
            {"trajectory_count", trajectory_count},
            {"plan_reuse_in_goal", plan_reuse_in_goal},
        });

    std::string llm_response = llm_->query(prompt);

    bool depends_on_repaired = false;
    bool fallback_used = false;
    std::string validation_reason;
    auto validated = parsePlanWithValidation(
        llm_response, plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);

    if (!validated.has_value()) {
        StructuredLogger::instance().log(LogLevel::Warn, "planner", "plan_validation_failed",
            "First plan attempt failed (" + validation_reason + "), retrying...",
            {{"llm_response", llm_response}, {"reason", validation_reason}});

        std::string retry_prompt = prompt + "\n\nERROR: " + validation_reason +
            ". Respond with JSON only. Step 1 MUST be RETRIEVAL, step 2 MUST be LLM with depends_on.\nPrevious Response:\n" +
            llm_response;
        llm_response = llm_->query(retry_prompt);
        validated = parsePlanWithValidation(
            llm_response, plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);
    }

    plan = finalizePlanOrFallback(plan.plan_id, prompt_goal, validated, fallback_used);

    StructuredLogger::instance().log(
        LogLevel::Info,
        "planner",
        fallback_used ? "plan_fallback_used" : "plan_generated",
        fallback_used ? "Used programmatic RETRIEVAL→LLM fallback plan"
                      : "LLMPlanner created a validated plan",
        {{"plan_id", plan.plan_id},
         {"step_count", plan.steps.size()},
         {"depends_on_repaired", depends_on_repaired},
         {"fallback_used", fallback_used},
         {"validation_reason", validation_reason}});

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

    const auto [prompt_goal, _] = Thoth::splitPlanReuseInjection(existing_plan.goal);

    Thoth::PlannerPromptMetrics prompt_metrics;
    std::string prompt = prompt_factory_->buildRevisionPrompt(
        prompt_goal,
        existing_plan.to_json().dump(),
        step_result.dump(),
        &prompt_metrics);

    StructuredLogger::instance().log(
        LogLevel::Info,
        "planner",
        "PLANNER_REVISION_CONTEXT",
        "Assembled revision prompt (failure context only, no plan-reuse injection)",
        {
            {"plan_id", existing_plan.plan_id},
            {"rules_bytes", prompt_metrics.rules_bytes},
            {"schema_bytes", prompt_metrics.schema_bytes},
            {"goal_bytes", prompt_metrics.goal_bytes},
            {"total_bytes", prompt_metrics.total_bytes},
        });

    std::string llm_response = llm_->query(prompt);

    bool depends_on_repaired = false;
    bool fallback_used = false;
    std::string validation_reason;
    auto validated = parsePlanWithValidation(
        llm_response, existing_plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);

    if (!validated.has_value()) {
        StructuredLogger::instance().log(LogLevel::Warn, "planner", "revision_validation_failed",
            "First revision attempt failed (" + validation_reason + "), retrying...",
            {{"llm_response", llm_response}, {"reason", validation_reason}});

        std::string retry_prompt = prompt + "\n\nERROR: " + validation_reason +
            ". Respond with JSON only. Step 1 MUST be RETRIEVAL, step 2 MUST be LLM with depends_on.\nPrevious Response:\n" +
            llm_response;
        llm_response = llm_->query(retry_prompt);
        validated = parsePlanWithValidation(
            llm_response, existing_plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);
    }

    if (validated.has_value()) {
        Plan revised = validated.value();
        revised.goal = prompt_goal;
        revised.created_at_ms = existing_plan.created_at_ms;
        revised.updated_at_ms = nowMs();
        revised.status = PlanStatus::ACTIVE;
        revised.current_index = 0;

        StructuredLogger::instance().log(
            LogLevel::Info,
            "planner",
            "plan_revised",
            "LLMPlanner revised the existing plan",
            {{"plan_id", revised.plan_id},
             {"step_count", revised.steps.size()},
             {"depends_on_repaired", depends_on_repaired},
             {"fallback_used", false}});

        save_plan(revised);
        return revised;
    }

    StructuredLogger::instance().log(
        LogLevel::Warn,
        "planner",
        "revision_fallback_kept",
        "Revision validation failed; keeping existing plan",
        {{"plan_id", existing_plan.plan_id}, {"reason", validation_reason}});

    Plan revised = existing_plan;
    revised.updated_at_ms = nowMs();
    save_plan(revised);
    return revised;
}

void LLMPlanner::save_plan(const Plan& plan) {
    if (!memory_) return;

    Memory::CognatePlanRecord rec;
    rec.plan_id = plan.plan_id;
    rec.goal = plan.goal;
    rec.plan_json = plan.to_json().dump();
    rec.status = static_cast<int>(plan.status);
    rec.success_score = 0.0f;
    rec.created_at = plan.created_at_ms;
    rec.updated_at = plan.updated_at_ms;

    if (rag_ && rag_->engine) {
        rec.embedding = rag_->engine->embed(Thoth::cleanGoalForStorage(plan.goal));
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
