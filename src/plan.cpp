/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/plan.h"
#include "../include/goal_text_utils.h"

namespace {

Thoth::E2RunBlockReason runBlockReasonFromJsonString(const std::string& value) {
    if (value == "RUNTIME_HEURISTIC_GUARD") {
        return Thoth::E2RunBlockReason::RUNTIME_HEURISTIC_GUARD;
    }
    if (value == "WIRING_GATE") {
        return Thoth::E2RunBlockReason::WIRING_GATE;
    }
    if (value == "STRICT_BOUNDARY_VIOLATION") {
        return Thoth::E2RunBlockReason::STRICT_BOUNDARY_VIOLATION;
    }
    if (value == "PROVENANCE_VIOLATION") {
        return Thoth::E2RunBlockReason::PROVENANCE_VIOLATION;
    }
    return Thoth::E2RunBlockReason::NONE;
}

void loadPlanStepOutcomeFromJson(const nlohmann::json& j, PlanStepOutcome* outcome) {
    if (!outcome) {
        return;
    }
    if (j.contains("outcome") && j["outcome"].is_object()) {
        outcome->run_block_reason =
            runBlockReasonFromJsonString(j["outcome"].value("run_block_reason", "NONE"));
        return;
    }
    if (j.contains("run_block_reason") && j["run_block_reason"].is_string()) {
        outcome->run_block_reason =
            runBlockReasonFromJsonString(j["run_block_reason"].get<std::string>());
    }
}

} // namespace

nlohmann::json PlanStep::to_json() const {
    nlohmann::json j;
    j["step_id"] = step_id;
    j["description"] = description;
    j["type"] = static_cast<int>(type);
    j["tool"] = tool;
    j["payload"] = payload;
    j["status"] = static_cast<int>(status);
    j["retry_count"] = retry_count;
    j["failure_policy"] = {
        {"max_retries", failure_policy.max_retries},
        {"abort_on_failure", failure_policy.abort_on_failure},
        {"revise_plan_on_failure", failure_policy.revise_plan_on_failure}
    };
    j["result"] = result;
    j["outcome"] = {
        {"run_block_reason", Thoth::e2RunBlockReasonToString(outcome.run_block_reason)}};
    j["reasoning"] = reasoning;
    j["started_at_ms"] = started_at_ms;
    j["completed_at_ms"] = completed_at_ms;
    j["depends_on"] = depends_on;
    return j;
}

PlanStep PlanStep::from_json(const nlohmann::json& j) {
    PlanStep step;
    step.step_id = j.value("step_id", "");
    step.description = j.value("description", "");
    step.type = static_cast<StepType>(j.value("type", 0));
    step.tool = j.value("tool", nlohmann::json::object());
    step.payload = j.value("payload", nlohmann::json::object());
    step.status = static_cast<StepStatus>(j.value("status", 0));
    step.retry_count = j.value("retry_count", 0);
    if (j.contains("failure_policy")) {
        step.failure_policy.max_retries = j["failure_policy"].value("max_retries", 1);
        step.failure_policy.abort_on_failure = j["failure_policy"].value("abort_on_failure", false);
        step.failure_policy.revise_plan_on_failure = j["failure_policy"].value("revise_plan_on_failure", false);
    }
    step.result = j.value("result", nlohmann::json::object());
    loadPlanStepOutcomeFromJson(j, &step.outcome);
    step.reasoning = j.value("reasoning", "");
    step.started_at_ms = j.value("started_at_ms", static_cast<std::int64_t>(0));
    step.completed_at_ms = j.value("completed_at_ms", static_cast<std::int64_t>(0));
    step.depends_on = j.value("depends_on", std::vector<std::string>());
    return step;
}

nlohmann::json Plan::toLlmFacingJson() const {
    nlohmann::json steps = nlohmann::json::array();
    for (const auto& step : this->steps) {
        nlohmann::json item = nlohmann::json::object();
        if (!step.step_id.empty()) {
            item["step_id"] = step.step_id;
        }
        item["step_type"] = Thoth::stepTypeLabel(step.type);
        item["description"] = step.description;
        if (!step.depends_on.empty()) {
            item["depends_on"] = step.depends_on;
        }
        if (!step.payload.is_null()) {
            item["payload"] = step.payload;
        }
        steps.push_back(std::move(item));
    }
    return nlohmann::json{{"plan", std::move(steps)}};
}

nlohmann::json Plan::to_json() const {
    nlohmann::json j;
    j["plan_id"] = plan_id;
    j["goal"] = goal;
    if (!plan_origin.empty()) {
        j["plan_origin"] = plan_origin;
    }
    j["current_index"] = current_index;
    j["status"] = static_cast<int>(status);
    j["created_at_ms"] = created_at_ms;
    j["updated_at_ms"] = updated_at_ms;
    j["steps"] = nlohmann::json::array();
    for (const auto& step : steps) {
        j["steps"].push_back(step.to_json());
    }
    j["graph_reinforced"] = graph_reinforced;
    j["reflection_pending"] = reflection_pending;
    j["reflection_count"] = reflection_count;
    j["history_aging_done"] = history_aging_done;
    if (retrieval_sets_present) {
        nlohmann::json sets = nlohmann::json::array();
        for (const auto& set : retrieval_sets) {
            sets.push_back({
                {"step_id", set.step_id},
                {"chunk_hashes", set.chunk_hashes}
            });
        }
        j["retrieval_sets"] = std::move(sets);
    }
    if (working_trajectory_present) {
        j["working_trajectory"] = working_trajectory;
        j["working_revisions_count"] = working_revisions_count;
    }
    return j;
}

Plan Plan::from_json(const nlohmann::json& j) {
    Plan plan;
    plan.plan_id = j.value("plan_id", "");
    plan.goal = j.value("goal", "");
    if (j.contains("plan_origin") && j["plan_origin"].is_string()) {
        plan.plan_origin = j["plan_origin"].get<std::string>();
    }
    plan.current_index = j.value("current_index", 0);
    plan.status = static_cast<PlanStatus>(j.value("status", 0));
    plan.created_at_ms = j.value("created_at_ms", static_cast<std::int64_t>(0));
    plan.updated_at_ms = j.value("updated_at_ms", static_cast<std::int64_t>(0));
    if (j.contains("steps") && j["steps"].is_array()) {
        for (const auto& step_j : j["steps"]) {
            plan.steps.push_back(PlanStep::from_json(step_j));
        }
    }
    if (j.contains("graph_reinforced") && j["graph_reinforced"].is_boolean()) {
        plan.graph_reinforced = j["graph_reinforced"].get<bool>();
    }
    if (j.contains("reflection_pending") && j["reflection_pending"].is_boolean()) {
        plan.reflection_pending = j["reflection_pending"].get<bool>();
    }
    if (j.contains("reflection_count") && j["reflection_count"].is_number_integer()) {
        plan.reflection_count = j["reflection_count"].get<int>();
    }
    if (j.contains("history_aging_done") && j["history_aging_done"].is_boolean()) {
        plan.history_aging_done = j["history_aging_done"].get<bool>();
    }
    if (j.contains("retrieval_sets") && j["retrieval_sets"].is_array()) {
        plan.retrieval_sets_present = true;
        for (const auto& set_j : j["retrieval_sets"]) {
            if (!set_j.is_object()) {
                continue;
            }
            RetrievalSetRecord set;
            set.step_id = set_j.value("step_id", "");
            if (set_j.contains("chunk_hashes") && set_j["chunk_hashes"].is_array()) {
                for (const auto& hash_j : set_j["chunk_hashes"]) {
                    if (hash_j.is_string()) {
                        set.chunk_hashes.push_back(hash_j.get<std::string>());
                    }
                }
            }
            plan.retrieval_sets.push_back(std::move(set));
        }
    }
    if (j.contains("working_trajectory") && j["working_trajectory"].is_object()) {
        plan.working_trajectory_present = true;
        plan.working_trajectory = j["working_trajectory"];
        plan.working_revisions_count = j.value("working_revisions_count", 0);
    }
    return plan;
}
