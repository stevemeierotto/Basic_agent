/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "json.hpp"
#include "episodic_learning_eval.h"

// StepType — what kind of work this step does
enum class StepType {
    TOOL,       // Execute a registered tool via ToolRegistry
    RETRIEVAL,  // RAG or GRAG retrieval
    LLM,        // LLM reasoning call via IPlanner
    NODE        // Future: execute a NODE graph node by node_id
};

// StepStatus — lifecycle state of a single step
enum class StepStatus {
    PENDING,
    RUNNING,
    SUCCESS,
    FAILED,
    SKIPPED
};

// PlanStatus — lifecycle state of the whole plan
enum class PlanStatus {
    ACTIVE,
    COMPLETED,
    FAILED,
    ABORTED
};

// StepFailurePolicy — per-step behavior on failure
struct StepFailurePolicy {
    int max_retries = 1;
    bool abort_on_failure = false;
    bool revise_plan_on_failure = false;
    int timeout_ms = 30000; // Default 30s
};

/** Execution envelope — mirrors non-payload StepResult fields (B3.1). */
struct PlanStepOutcome {
    Thoth::E2RunBlockReason run_block_reason = Thoth::E2RunBlockReason::NONE;
};

// PlanStep — a single unit of work inside a Plan
struct PlanStep {
    std::string step_id;             // UUID string
    std::string description;         // Human-readable label
    StepType type;
    nlohmann::json tool;             // Optional tool schema/info
    nlohmann::json payload;          // Tool args, retrieval config, node_id, etc.
    StepStatus status = StepStatus::PENDING;
    int retry_count = 0;
    StepFailurePolicy failure_policy;
    nlohmann::json result;           // Structured output after execution
    PlanStepOutcome outcome;         // Execution envelope — not domain payload
    std::string reasoning;           // Why this step exists (for trace logging)
    int64_t started_at_ms = 0;
    int64_t completed_at_ms = 0;
    std::vector<std::string> depends_on;

    nlohmann::json to_json() const;
    static PlanStep from_json(const nlohmann::json& j);
};

// One captured GRAG retrieval set, in completion order.
// Stage 1 persists this list. It does not mean the edges were reinforced.
struct RetrievalSetRecord {
    std::string step_id;
    std::vector<std::string> chunk_hashes;
};

// Plan — the full goal execution plan
struct Plan {
    std::string plan_id;             // UUID string
    std::string goal;
    /** Empty on legacy rows. "model" or "fallback" after the planner selects a plan. */
    std::string plan_origin;
    std::vector<PlanStep> steps;
    size_t current_index = 0;
    PlanStatus status = PlanStatus::ACTIVE;
    int64_t created_at_ms = 0;
    int64_t updated_at_ms = 0;
    // False on legacy rows that omit the key. True only after the graph
    // transaction commits the edges and this marker together.
    bool graph_reinforced = false;
    // Set with the graph marker when this attempt must continue into a
    // replacement plan. The stored status stays ACTIVE until that plan exists.
    bool reflection_pending = false;
    int reflection_count = 0;
    // True only after history-store aging commits with this marker.
    bool history_aging_done = false;
    // False when `retrieval_sets` was absent. An empty present list is a real value.
    bool retrieval_sets_present = false;
    std::vector<RetrievalSetRecord> retrieval_sets;
    // Absent on older active plans. Present only on the stored checkpoint copy.
    // Not a finalized trajectory-table row.
    bool working_trajectory_present = false;
    nlohmann::json working_trajectory = nlohmann::json::object();
    int working_revisions_count = 0;

    nlohmann::json to_json() const;
    /** Planner-boundary view: root `plan` array and string `step_type`. Not the stored plan. */
    nlohmann::json toLlmFacingJson() const;
    static Plan from_json(const nlohmann::json& j);
};
