/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 7.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/trajectory.h"

namespace Thoth {

nlohmann::json RecordedStep::to_json() const {
    nlohmann::json j;
    j["step_id"] = step_id;
    j["description"] = description;
    if (type.has_value()) {
        j["type"] = static_cast<int>(*type);
    }
    j["tool"] = tool;
    j["result"] = result;
    j["error"] = error;
    j["revision_note"] = revision_note;
    j["timestamp"] = timestamp;
    return j;
}

RecordedStep RecordedStep::from_json(const nlohmann::json& j) {
    RecordedStep step;
    step.step_id = j.value("step_id", "");
    step.description = j.value("description", "");
    if (j.contains("type") && j["type"].is_number_integer()) {
        step.type = static_cast<StepType>(j["type"].get<int>());
    }
    step.tool = j.value("tool", nlohmann::json::object());
    step.result = j.value("result", nlohmann::json::object());
    step.error = j.value("error", "");
    step.revision_note = j.value("revision_note", "");
    step.timestamp = j.value("timestamp", 0);
    return step;
}

nlohmann::json Trajectory::to_json() const {
    nlohmann::json j;
    j["trajectory_id"] = trajectory_id;
    j["goal"] = goal;
    j["plan_initial"] = plan_initial.to_json();
    
    nlohmann::json steps_j = nlohmann::json::array();
    for (const auto& s : steps) {
        steps_j.push_back(s.to_json());
    }
    j["steps"] = steps_j;
    
    j["results"] = results;
    
    nlohmann::json revisions_j = nlohmann::json::array();
    for (const auto& r : revisions) {
        revisions_j.push_back(r.to_json());
    }
    j["revisions"] = revisions_j;
    
    j["final_status"] = static_cast<int>(final_status);
    j["success_score"] = success_score;
    j["created_at"] = created_at;
    j["embedding"] = embedding;
    
    return j;
}

Trajectory Trajectory::from_json(const nlohmann::json& j) {
    Trajectory t;
    t.trajectory_id = j.value("trajectory_id", "");
    t.goal = j.value("goal", "");
    
    if (j.contains("plan_initial")) {
        t.plan_initial = Plan::from_json(j["plan_initial"]);
    }
    
    if (j.contains("steps") && j["steps"].is_array()) {
        for (const auto& sj : j["steps"]) {
            t.steps.push_back(RecordedStep::from_json(sj));
        }
    }
    
    t.results = j.value("results", nlohmann::json::object());
    
    if (j.contains("revisions") && j["revisions"].is_array()) {
        for (const auto& rj : j["revisions"]) {
            t.revisions.push_back(Plan::from_json(rj));
        }
    }
    
    t.final_status = static_cast<PlanStatus>(j.value("final_status", 0));
    t.success_score = j.value("success_score", 0.0f);
    t.created_at = j.value("created_at", 0);
    t.embedding = j.value("embedding", std::vector<float>());
    
    return t;
}

} // namespace Thoth
