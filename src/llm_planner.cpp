/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/llm_planner.h"
#include "../include/alp_sha256.h"
#include "../include/generation_budget.h"
#include "../include/generation_call_log.h"
#include "../include/logger.h"
#include "../include/plan_parser.h"
#include "../include/plan_validator.h"
#include "../include/plan_reuse_config.h"
#include "../include/goal_text_utils.h"
#include "../include/planner_injection_config.h"
#include "../include/grag_scorer.h"
#include "../include/embedding_engine.h"
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <optional>
#include <vector>

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

} // namespace

namespace Thoth {

GeneratedPlanAssessment assessGeneratedPlan(const std::string& llm_response,
                                            const std::string& plan_id,
                                            const std::string& prompt_goal,
                                            bool allow_tool_steps) {
    GeneratedPlanAssessment assessment;
    std::string parser_reason;
    auto parsed = PlanParser::parse(llm_response, plan_id, &parser_reason);
    if (!parsed.has_value()) {
        assessment.reason = parser_reason;
        return assessment;
    }

    Plan candidate = parsed.value();
    candidate.goal = prompt_goal;
    auto validation = PlanValidator::validateAndRepair(candidate, allow_tool_steps);
    assessment.depends_on_repaired = validation.depends_on_repaired;
    if (validation.valid) {
        assessment.plan = candidate;
        assessment.reason = validation.depends_on_repaired ? "depends_on wired" : "ok";
        return assessment;
    }
    assessment.reason = validation.reason;
    return assessment;
}

const char* plannerFailureClassLabel(PlannerFailureClass failure) {
    switch (failure) {
        case PlannerFailureClass::Length: return "length";
        case PlannerFailureClass::Parse: return "parse";
        case PlannerFailureClass::Schema: return "schema";
        case PlannerFailureClass::Provider: return "provider";
        case PlannerFailureClass::None: return "none";
    }
    return "none";
}

bool isSchemaAssessmentReason(const std::string& reason) {
    return reason.rfind("Step missing or invalid", 0) == 0
        || reason.rfind("Invalid 'step_type'", 0) == 0
        || reason.rfind("Plan step is not", 0) == 0
        || reason == "Plan contains no steps."
        || reason.rfind("Plan has no steps", 0) == 0
        || reason.rfind("TOOL steps", 0) == 0
        || reason.rfind("Corpus Q&A", 0) == 0
        || reason.rfind("First executable step", 0) == 0
        || reason.rfind("Plan must contain", 0) == 0
        || reason.rfind("RETRIEVAL step requires", 0) == 0;
}

std::string capDiagnostic(const std::string& text) {
    std::string capped = text.substr(0, std::min<std::size_t>(text.size(), 240));
    for (char& ch : capped) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return capped;
}

PlannerFailureClass classifyPlannerFailure(bool provider_ok,
                                           const std::string& finish_reason,
                                           const std::string& assessment_reason,
                                           bool plan_accepted) {
    if (!provider_ok) {
        return PlannerFailureClass::Provider;
    }
    if (plan_accepted) {
        return PlannerFailureClass::None;
    }
    if (finish_reason == "length") {
        return PlannerFailureClass::Length;
    }
    if (isSchemaAssessmentReason(assessment_reason)) {
        return PlannerFailureClass::Schema;
    }
    return PlannerFailureClass::Parse;
}

std::string buildPlannerRetryPrompt(const std::string& prompt,
                                    const std::string& validation_reason,
                                    const std::string& previous_response,
                                    PlannerFailureClass failure) {
    std::string retry = prompt + "\n\nERROR: " + validation_reason + ". ";
    switch (failure) {
        case PlannerFailureClass::Length:
            retry += "Respond with JSON only. Step 1 MUST be RETRIEVAL, step 2 MUST be LLM with depends_on.\n";
            retry += "Previous response omitted because it hit the completion token limit.";
            return retry;
        case PlannerFailureClass::Parse:
            retry += "Return one fresh, complete JSON plan. Do not continue or repair the previous text.\n";
            return retry;
        case PlannerFailureClass::Schema:
            retry += "A required field is missing or invalid. Return a new JSON plan with a nonempty "
                     "description on every step, a nonempty retrieval query, and depends_on on the LLM step. "
                     "Do not repeat the rejected plan.\n";
            return retry;
        case PlannerFailureClass::Provider:
            retry += "The model provider failed. This is not a schema error. Return one complete JSON plan.\n";
            if (!previous_response.empty()) {
                retry += "Provider detail: " + capDiagnostic(previous_response) + "\n";
            }
            return retry;
        case PlannerFailureClass::None:
            break;
    }
    retry += "Respond with JSON only. Step 1 MUST be RETRIEVAL, step 2 MUST be LLM with depends_on.\n";
    return retry;
}

}  // namespace Thoth

namespace {

std::optional<Plan> parsePlanWithValidation(const std::string& llm_response,
                                            const std::string& plan_id,
                                            const std::string& prompt_goal,
                                            bool allow_tool_steps,
                                            bool& depends_on_repaired,
                                            bool& fallback_used,
                                            std::string& validation_reason) {
    depends_on_repaired = false;
    fallback_used = false;
    const Thoth::GeneratedPlanAssessment assessment =
        Thoth::assessGeneratedPlan(llm_response, plan_id, prompt_goal, allow_tool_steps);
    depends_on_repaired = assessment.depends_on_repaired;
    validation_reason = assessment.reason;
    return assessment.plan;
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
        plan.plan_origin = "model";
        return plan;
    }

    fallback_used = true;
    Plan plan = Thoth::PlanValidator::createFallbackPlan(plan_id, prompt_goal);
    plan.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    plan.updated_at_ms = plan.created_at_ms;
    plan.plan_origin = "fallback";
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

    Thoth::GenerationCallContext call = Thoth::GenerationCallScope::current();
    call.call_type = "plan";
    call.attempt = 0;
    call.plan_id = plan.plan_id;
    const Thoth::GenerationOutcome first = llm_->generateCall(prompt, -1, {}, call);

    bool depends_on_repaired = false;
    bool fallback_used = false;
    std::string validation_reason;
    std::optional<Plan> validated;
    Thoth::PlannerFailureClass first_failure = Thoth::PlannerFailureClass::None;
    if (!first.ok) {
        first_failure = Thoth::PlannerFailureClass::Provider;
        validation_reason = first.error;
    } else {
        validated = parsePlanWithValidation(
            first.text, plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);
        first_failure = Thoth::classifyPlannerFailure(
            true, first.finish_reason, validation_reason, validated.has_value());
    }

    if (!validated.has_value()) {
        logGeneration(first, false, false, false, depends_on_repaired, first_failure, "retry", Plan{});
        StructuredLogger::instance().log(LogLevel::Warn, "planner", "plan_validation_failed",
            "First plan attempt failed (" + validation_reason + "), retrying...",
            {{"reason", validation_reason},
             {"failure_class", Thoth::plannerFailureClassLabel(first_failure)}});

        const std::string diagnostic = first.ok ? first.text : first.error;
        std::string retry_prompt = Thoth::buildPlannerRetryPrompt(
            prompt, validation_reason, diagnostic, first_failure);
        Thoth::GenerationCallContext retry = call;
        retry.call_type = "plan_retry";
        retry.attempt = 1;
        const Thoth::GenerationOutcome second = llm_->generateCall(retry_prompt, -1, {}, retry);
        Thoth::PlannerFailureClass second_failure = Thoth::PlannerFailureClass::None;
        if (!second.ok) {
            second_failure = Thoth::PlannerFailureClass::Provider;
            validation_reason = second.error;
            validated.reset();
        } else {
            validated = parsePlanWithValidation(
                second.text, plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);
            second_failure = Thoth::classifyPlannerFailure(
                true, second.finish_reason, validation_reason, validated.has_value());
        }
        plan = finalizePlanOrFallback(plan.plan_id, prompt_goal, validated, fallback_used);
        logGeneration(second, validated.has_value(), fallback_used, false, depends_on_repaired,
                      validated.has_value() ? Thoth::PlannerFailureClass::None : second_failure,
                      validated.has_value() ? "accepted" : "fallback", plan);
    } else {
        plan = finalizePlanOrFallback(plan.plan_id, prompt_goal, validated, fallback_used);
        logGeneration(first, true, false, false, depends_on_repaired, Thoth::PlannerFailureClass::None,
                      "accepted", plan);
    }

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
         {"plan_origin", plan.plan_origin},
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
        existing_plan.toLlmFacingJson().dump(),
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

    Thoth::GenerationCallContext call = Thoth::GenerationCallScope::current();
    call.call_type = "revision";
    call.attempt = 0;
    if (call.plan_id.empty()) {
        call.plan_id = existing_plan.plan_id;
    }
    const std::string wrapper_hash = Thoth::sha256Hex(prompt);
    const Thoth::GenerationOutcome first_raw = llm_->generateCall(prompt, -1, {}, call);
    Thoth::GenerationOutcome first = first_raw;
    first.wrapper_sha256 = wrapper_hash;

    bool depends_on_repaired = false;
    bool fallback_used = false;
    std::string validation_reason;
    std::optional<Plan> validated;
    std::optional<Thoth::GenerationOutcome> retryOutcome;
    Thoth::PlannerFailureClass first_failure = Thoth::PlannerFailureClass::None;
    if (!first.ok) {
        first_failure = Thoth::PlannerFailureClass::Provider;
        validation_reason = first.error;
    } else {
        validated = parsePlanWithValidation(
            first.text, existing_plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used, validation_reason);
        first_failure = Thoth::classifyPlannerFailure(
            true, first.finish_reason, validation_reason, validated.has_value());
    }

    if (!validated.has_value()) {
        logGeneration(first, false, false, false, depends_on_repaired, first_failure, "retry", Plan{});
        StructuredLogger::instance().log(LogLevel::Warn, "planner", "revision_validation_failed",
            "First revision attempt failed (" + validation_reason + "), retrying...",
            {{"reason", validation_reason},
             {"failure_class", Thoth::plannerFailureClassLabel(first_failure)}});

        const std::string diagnostic = first.ok ? first.text : first.error;
        std::string retry_prompt = Thoth::buildPlannerRetryPrompt(
            prompt, validation_reason, diagnostic, first_failure);
        Thoth::GenerationCallContext retry = call;
        retry.call_type = "revision_retry";
        retry.attempt = 1;
        const Thoth::GenerationOutcome second_raw = llm_->generateCall(retry_prompt, -1, {}, retry);
        Thoth::GenerationOutcome second = second_raw;
        second.wrapper_sha256 = wrapper_hash;
        Thoth::PlannerFailureClass second_failure = Thoth::PlannerFailureClass::None;
        if (!second.ok) {
            second_failure = Thoth::PlannerFailureClass::Provider;
            validation_reason = second.error;
            validated.reset();
        } else {
            validated = parsePlanWithValidation(
                second.text, existing_plan.plan_id, prompt_goal, false, depends_on_repaired, fallback_used,
                validation_reason);
            second_failure = Thoth::classifyPlannerFailure(
                true, second.finish_reason, validation_reason, validated.has_value());
        }
        retryOutcome = second;
        if (!validated.has_value()) {
            logGeneration(second, false, false, true, depends_on_repaired, second_failure, "kept_existing",
                          existing_plan);
        }
    }

    if (validated.has_value()) {
        Plan revised = validated.value();
        revised.goal = prompt_goal;
        revised.created_at_ms = existing_plan.created_at_ms;
        revised.updated_at_ms = nowMs();
        revised.status = PlanStatus::ACTIVE;
        revised.current_index = 0;
        revised.plan_origin = "model";
        const Thoth::GenerationOutcome& acceptedOutcome = retryOutcome.has_value() ? *retryOutcome : first;

        logGeneration(acceptedOutcome, true, false, false, depends_on_repaired,
                      Thoth::PlannerFailureClass::None, "accepted", revised);

        StructuredLogger::instance().log(
            LogLevel::Info,
            "planner",
            "plan_revised",
            "LLMPlanner revised the existing plan",
            {{"plan_id", revised.plan_id},
             {"step_count", revised.steps.size()},
             {"depends_on_repaired", depends_on_repaired},
             {"fallback_used", false},
             {"plan_origin", revised.plan_origin}});

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

std::string LLMPlanner::buildRevisionPromptText(const Plan& existing_plan,
                                               const nlohmann::json& step_result) const {
    if (!prompt_factory_) {
        return {};
    }
    const auto [prompt_goal, unused] = Thoth::splitPlanReuseInjection(existing_plan.goal);
    (void)unused;
    return prompt_factory_->buildRevisionPrompt(
        prompt_goal, existing_plan.toLlmFacingJson().dump(), step_result.dump(), nullptr);
}

std::string LLMPlanner::revisionWrapperSha256(const Plan& existing_plan,
                                              const nlohmann::json& step_result) const {
    return Thoth::sha256Hex(buildRevisionPromptText(existing_plan, step_result));
}

void LLMPlanner::logGeneration(const Thoth::GenerationOutcome& outcome,
                               bool validation_ok,
                               bool fallback_used,
                               bool kept_existing_plan,
                               bool depends_on_repaired,
                               Thoth::PlannerFailureClass failure,
                               const std::string& attempt_disposition,
                               const Plan& recorded_plan) const {
    Thoth::GenerationRecordFields fields;
    fields.has_validation_ok = true;
    fields.validation_ok = validation_ok;
    fields.fallback_used = fallback_used;
    fields.kept_existing_plan = kept_existing_plan;
    fields.has_depends_on_repaired = true;
    fields.depends_on_repaired = depends_on_repaired;
    fields.failure_class = Thoth::plannerFailureClassLabel(failure);
    fields.attempt_disposition = attempt_disposition;
    fields.completion_sha256 = Thoth::sha256Hex(outcome.text);
    if (attempt_disposition != "retry") {
        fields.plan_origin = recorded_plan.plan_origin;
        for (const auto& step : recorded_plan.steps) {
            fields.step_ids.push_back(step.step_id);
            if (fields.retrieval_query.empty() && step.type == StepType::RETRIEVAL
                && step.payload.is_object() && step.payload.contains("query")
                && step.payload["query"].is_string()) {
                fields.retrieval_query = step.payload["query"].get<std::string>();
            }
        }
    }
    fields.associated_generation_id = outcome.generation_id;
    fields.context_overflow = reportedContextOverflow(outcome);
    Thoth::GenerationCallLog::append(outcome, fields);
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

namespace {

std::string gnuBuildIdHex(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "unavailable";
    }
    Elf64_Ehdr header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64
        || header.e_ident[EI_DATA] != ELFDATA2LSB) {
        return "unavailable";
    }
    for (Elf64_Half i = 0; i < header.e_phnum; ++i) {
        in.seekg(static_cast<std::streamoff>(header.e_phoff + static_cast<Elf64_Off>(i) * header.e_phentsize));
        Elf64_Phdr program{};
        in.read(reinterpret_cast<char*>(&program), sizeof(program));
        if (!in || program.p_type != PT_NOTE) {
            continue;
        }
        std::vector<char> notes(static_cast<std::size_t>(program.p_filesz));
        in.seekg(static_cast<std::streamoff>(program.p_offset));
        in.read(notes.data(), static_cast<std::streamsize>(notes.size()));
        if (!in) {
            return "unavailable";
        }
        std::size_t offset = 0;
        while (offset + 12 <= notes.size()) {
            std::uint32_t namesz = 0;
            std::uint32_t descsz = 0;
            std::uint32_t type = 0;
            std::memcpy(&namesz, notes.data() + offset, 4);
            std::memcpy(&descsz, notes.data() + offset + 4, 4);
            std::memcpy(&type, notes.data() + offset + 8, 4);
            const std::size_t nameOffset = offset + 12;
            const std::size_t descOffset = nameOffset + ((static_cast<std::size_t>(namesz) + 3u) & ~3u);
            const std::size_t next = descOffset + ((static_cast<std::size_t>(descsz) + 3u) & ~3u);
            if (next > notes.size()) {
                break;
            }
            if (type == NT_GNU_BUILD_ID && namesz >= 3
                && std::string(notes.data() + nameOffset, 3) == "GNU") {
                std::ostringstream hex;
                hex << std::hex << std::setfill('0');
                for (std::uint32_t byte = 0; byte < descsz; ++byte) {
                    hex << std::setw(2)
                        << static_cast<int>(static_cast<unsigned char>(notes[descOffset + byte]));
                }
                return hex.str();
            }
            offset = next;
        }
    }
    return "unavailable";
}

} // namespace

Thoth::LibraryProvenance Thoth::loadedLibraryProvenance() {
    LibraryProvenance provenance;
    provenance.sha256 = "unavailable";
    provenance.build_id = "unavailable";
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&loadedLibraryProvenance), &info) == 0 || info.dli_fname == nullptr) {
        return provenance;
    }
    const std::string loaded(info.dli_fname);
    if (loaded.find("libbasic_agent.so") == std::string::npos) {
        return provenance;
    }
    provenance.path = loaded;
    std::ifstream libraryFile(loaded, std::ios::binary);
    if (libraryFile) {
        std::ostringstream bytes;
        bytes << libraryFile.rdbuf();
        const std::string hash = sha256Hex(bytes.str());
        if (!hash.empty()) {
            provenance.sha256 = hash;
        }
    }
    provenance.build_id = gnuBuildIdHex(loaded);
    return provenance;
}
