/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C2 EvaluationSubscriber
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "evaluation_subscriber.h"

#include "diagnostic_service.h"
#include "episodic_evaluation_service.h"
#include "pipeline_telemetry_service.h"
#include "plan.h"

#include <chrono>
#include <optional>

namespace Thoth {

namespace {

// Clock invariant (E2-C4): durations use steady_clock only; absolute correlation uses
// system_clock only. Never derive durations from system_clock.
using SteadyClock = std::chrono::steady_clock;

std::int64_t durationMsSince(const SteadyClock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(SteadyClock::now() - start)
        .count();
}

std::int64_t wallClockNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

const EpisodicLearningSummary* g_last_summary_for_tests = nullptr;
EpisodicLearningSummary g_last_summary_storage;
const EvaluationDiagnosticsSummary* g_last_run_diagnostics_for_tests = nullptr;
EvaluationDiagnosticsSummary g_last_run_diagnostics_storage;
const E2PipelineStageTimings* g_last_stage_timings_for_tests = nullptr;
E2PipelineStageTimings g_last_stage_timings_storage;
const E2PipelineTelemetryRecord* g_last_telemetry_record_for_tests = nullptr;
E2PipelineTelemetryRecord g_last_telemetry_record_storage;
bool g_pipeline_telemetry_enabled = false;
std::optional<E2EvalConfig> g_eval_config_for_tests;

/** Test observation only — never read by production control paths. */
void setTelemetryTestObservation(const E2PipelineTelemetryRecord* record) {
    g_last_telemetry_record_for_tests = record;
}

E2ArmScoringStatus armScoringStatusFromBridgeString(const std::string& status) {
    if (status == "FAILED_RETRIEVAL") {
        return E2ArmScoringStatus::FAILED_RETRIEVAL;
    }
    if (status == "FAILED_PROVENANCE") {
        return E2ArmScoringStatus::FAILED_PROVENANCE;
    }
    if (status == "FAILED_STRICT_BOUNDARY") {
        return E2ArmScoringStatus::FAILED_STRICT_BOUNDARY;
    }
    return E2ArmScoringStatus::OK;
}

EpisodicLearningExpectations expectationsFromBridgeJson(const nlohmann::json& bridge) {
    EpisodicLearningExpectations exp;
    if (!bridge.is_object()) {
        return exp;
    }
    exp.expect_warm_retrieval_hit = bridge.value("expect_warm_retrieval_hit", false);
    const std::string constraint = bridge.value("lift_constraint", "GTE");
    exp.lift_constraint =
        constraint == "ABS_LT" ? EpisodicLiftConstraint::ABS_LT : EpisodicLiftConstraint::GTE;
    exp.lift_threshold = bridge.value("lift_threshold", kEpisodicLearningLiftMargin);
    exp.allow_binary_pass = bridge.value("allow_binary_pass", true);
    exp.include_in_mean_episodic_lift = bridge.value("include_in_mean_episodic_lift", false);
    exp.retrieval_match_token = bridge.value("retrieval_match_token", "");
    if (bridge.contains("forbidden_retrieval_tokens") &&
        bridge["forbidden_retrieval_tokens"].is_array()) {
        for (const auto& token : bridge["forbidden_retrieval_tokens"]) {
            if (token.is_string()) {
                exp.forbidden_retrieval_tokens.push_back(token.get<std::string>());
            }
        }
    }
    return exp;
}

void applyRetrievalBridge(EpisodicRetrievalProvenance& retrieval, const nlohmann::json& bridge) {
    if (!bridge.is_object()) {
        return;
    }
    if (bridge.contains("forbidden_tokens_found") && bridge["forbidden_tokens_found"].is_array()) {
        retrieval.forbidden_tokens_found.clear();
        for (const auto& token : bridge["forbidden_tokens_found"]) {
            if (token.is_string()) {
                retrieval.forbidden_tokens_found.push_back(token.get<std::string>());
            }
        }
    }
    if (bridge.contains("arm_scoring_status") && bridge["arm_scoring_status"].is_string()) {
        retrieval.arm_scoring_status =
            armScoringStatusFromBridgeString(bridge["arm_scoring_status"].get<std::string>());
    }
    if (bridge.contains("chunks") && bridge["chunks"].is_array()) {
        retrieval.chunks.clear();
        for (const auto& item : bridge["chunks"]) {
            if (!item.is_object()) {
                continue;
            }
            RetrievedChunkRecord chunk;
            chunk.chunk_id = item.value("chunk_id", "");
            chunk.source_id = item.value("source_id", "");
            const std::string source = item.value("source", "corpus");
            if (source == "evaluation") {
                chunk.source = RetrievedChunkSource::EVALUATION;
            } else if (source == "synthetic") {
                chunk.source = RetrievedChunkSource::SYNTHETIC;
            } else if (source == "user") {
                chunk.source = RetrievedChunkSource::USER;
            } else if (source == "system") {
                chunk.source = RetrievedChunkSource::SYSTEM;
            } else {
                chunk.source = RetrievedChunkSource::CORPUS;
            }
            const std::string status = item.value("validation_status", "valid");
            if (status == "invalid") {
                chunk.validation_status = ProvenanceValidationStatus::INVALID;
            } else if (status == "untraced") {
                chunk.validation_status = ProvenanceValidationStatus::UNTRACED;
            } else {
                chunk.validation_status = ProvenanceValidationStatus::VALID;
            }
            retrieval.chunks.push_back(std::move(chunk));
        }
    }
    if (bridge.contains("warm_retrieval_hit")) {
        retrieval.warm_retrieval_hit = bridge.value("warm_retrieval_hit", false);
    } else if (!retrieval.chunks.empty()) {
        retrieval.warm_retrieval_hit = true;
    }
}

std::string terminalStateFromEvent(const EpisodeCompleted& event) {
    if (event.terminal_state == "COMPLETED") {
        return "COMPLETED";
    }
    return "FAILED";
}

float scoreFromPlanSnapshot(const nlohmann::json& plan) {
    if (!plan.is_object()) {
        return 0.0f;
    }
    if (plan.contains("success_score") && plan["success_score"].is_number()) {
        return plan["success_score"].get<float>();
    }
    return 0.0f;
}

} // namespace

ProductionEpisodeMapping mapEpisodeToProductionObservations(const EpisodeCompleted& event) {
    ProductionEpisodeMapping mapped;
    mapped.expectations.allow_binary_pass = true;
    mapped.expectations.lift_constraint = EpisodicLiftConstraint::GTE;
    mapped.expectations.lift_threshold = kEpisodicLearningLiftMargin;
    mapped.expectations.include_in_mean_episodic_lift = true;

    if (event.plan_snapshot.is_object() &&
        event.plan_snapshot.contains("e2_expectations")) {
        mapped.expectations =
            expectationsFromBridgeJson(event.plan_snapshot["e2_expectations"]);
    }

    const float warmScore = event.final_success_score;
    const float coldScore = scoreFromPlanSnapshot(event.plan_snapshot);

    mapped.cold.arm_label = "cold";
    mapped.cold.terminal_state = "FAILED";
    if (event.plan_snapshot.is_object() &&
        event.plan_snapshot.contains("e2_cold_terminal_state") &&
        event.plan_snapshot["e2_cold_terminal_state"].is_string()) {
        mapped.cold.terminal_state = event.plan_snapshot["e2_cold_terminal_state"].get<std::string>();
    }
    mapped.cold.final_success_score = coldScore;
    mapped.cold.arm_scoring_status = E2ArmScoringStatus::OK;
    if (event.plan_snapshot.is_object() &&
        event.plan_snapshot.contains("e2_cold_arm_scoring_status") &&
        event.plan_snapshot["e2_cold_arm_scoring_status"].is_string()) {
        mapped.cold.arm_scoring_status = armScoringStatusFromBridgeString(
            event.plan_snapshot["e2_cold_arm_scoring_status"].get<std::string>());
    }

    mapped.warm.arm_label = "warm";
    mapped.warm.terminal_state = terminalStateFromEvent(event);
    mapped.warm.final_success_score = warmScore;
    mapped.warm.arm_scoring_status = E2ArmScoringStatus::OK;
    if (event.plan_snapshot.is_object() &&
        event.plan_snapshot.contains("e2_warm_arm_scoring_status") &&
        event.plan_snapshot["e2_warm_arm_scoring_status"].is_string()) {
        mapped.warm.arm_scoring_status = armScoringStatusFromBridgeString(
            event.plan_snapshot["e2_warm_arm_scoring_status"].get<std::string>());
    }

    if (event.plan_snapshot.is_object() && event.plan_snapshot.contains("steps") &&
        event.plan_snapshot["steps"].is_array()) {
        for (const auto& step : event.plan_snapshot["steps"]) {
            if (!step.is_object()) {
                continue;
            }
            const int stepType = step.value("type", -1);
            if (stepType != static_cast<int>(StepType::RETRIEVAL)) {
                continue;
            }
            RetrievedChunkRecord chunk;
            chunk.source = RetrievedChunkSource::CORPUS;
            chunk.validation_status = ProvenanceValidationStatus::VALID;
            chunk.chunk_id = step.value("step_id", "");
            chunk.source_id = chunk.chunk_id;
            mapped.warm.retrieval.chunks.push_back(chunk);
            mapped.warm.retrieval.warm_retrieval_hit = !mapped.warm.retrieval.chunks.empty();
            break;
        }
    }
    mapped.warm.retrieval.arm_scoring_status = mapped.warm.arm_scoring_status;
    if (event.plan_snapshot.is_object() &&
        event.plan_snapshot.contains("e2_warm_retrieval_bridge")) {
        applyRetrievalBridge(mapped.warm.retrieval,
                             event.plan_snapshot["e2_warm_retrieval_bridge"]);
        mapped.warm.arm_scoring_status = mapped.warm.retrieval.arm_scoring_status;
    }
    if (event.plan_snapshot.is_object() &&
        event.plan_snapshot.contains("e2_cold_retrieval_bridge")) {
        applyRetrievalBridge(mapped.cold.retrieval,
                             event.plan_snapshot["e2_cold_retrieval_bridge"]);
        mapped.cold.arm_scoring_status = mapped.cold.retrieval.arm_scoring_status;
    }

    return mapped;
}

void EvaluationSubscriber::onEpisodeCompleted(const EpisodeCompleted& event) {
    const SteadyClock::time_point pipeline_start = SteadyClock::now();
    const std::int64_t subscriber_entry_ms = wallClockNowMs();

    const SteadyClock::time_point mapping_start = SteadyClock::now();
    const ProductionEpisodeMapping mapped = mapEpisodeToProductionObservations(event);
    const std::int64_t mapping_duration_ms = durationMsSince(mapping_start);

    E2EvalConfig config = g_eval_config_for_tests.value_or(E2EvalConfig::integrationDefaults());
    if (!g_eval_config_for_tests.has_value()) {
        config.tier = E2EvalTier::INTEGRATION;
    }

    const IEpisodicEvaluationService& evalService = episodicEvaluationService();
    const SteadyClock::time_point evaluation_start = SteadyClock::now();
    EpisodicLearningCaseEvaluation evaluation =
        evalService.evaluateCase(event.plan_id, mapped.expectations, mapped.cold, mapped.warm,
                                 config);
    if (event.plan_snapshot.is_object() && event.plan_snapshot.contains("e2_run_block_reason") &&
        event.plan_snapshot["e2_run_block_reason"].is_number_integer()) {
        evaluation.run_block_reason = static_cast<E2RunBlockReason>(
            event.plan_snapshot["e2_run_block_reason"].get<int>());
    } else {
        evaluation.run_block_reason = E2RunBlockReason::NONE;
    }
    evalService.applyCaseResolution(evaluation);

    std::vector<EpisodicLearningCaseEvaluation> case_results{evaluation};
    std::vector<EpisodicLearningExpectations> expectations{mapped.expectations};
    EpisodicLearningSummary summary =
        evalService.summarize(case_results, expectations, config);
    const std::int64_t evaluation_duration_ms = durationMsSince(evaluation_start);

    g_last_summary_storage = summary;
    g_last_summary_for_tests = &g_last_summary_storage;

    EvaluationDiagnosticsContext diagContext;
    diagContext.run_id = event.run_id;
    diagContext.env_hash = event.env_hash;
    diagContext.e2_eval_config = config.toJson();
    const auto fingerprint = evalService.computeFingerprint(config);
    diagContext.fingerprint_hash = fingerprint.fingerprint_hash;

    const SteadyClock::time_point diagnostic_start = SteadyClock::now();
    const EvaluationDiagnosticsSummary runDiagnostics =
        episodicDiagnosticService().generateRunDiagnostics(summary, diagContext);
    const std::int64_t diagnostic_duration_ms = durationMsSince(diagnostic_start);

    g_last_run_diagnostics_storage = runDiagnostics;
    g_last_run_diagnostics_for_tests = &g_last_run_diagnostics_storage;
    (void)evaluationDiagnosticSummaryToJson(runDiagnostics);

    E2PipelineStageTimings stage_timings;
    stage_timings.mapping_duration_ms = mapping_duration_ms;
    stage_timings.evaluation_duration_ms = evaluation_duration_ms;
    stage_timings.diagnostic_duration_ms = diagnostic_duration_ms;
    stage_timings.pipeline_duration_ms = durationMsSince(pipeline_start);
    stage_timings.episodes_processed = 1;
    if (event.completed_at_ms > 0 && subscriber_entry_ms >= event.completed_at_ms) {
        stage_timings.publication_to_subscriber_ms = subscriber_entry_ms - event.completed_at_ms;
    }
    g_last_stage_timings_storage = stage_timings;
    g_last_stage_timings_for_tests = &g_last_stage_timings_storage;

    if (g_pipeline_telemetry_enabled) {
        setTelemetryTestObservation(nullptr);
        try {
            E2PipelineTelemetryContext telemetry_context;
            telemetry_context.run_id = event.run_id;
            telemetry_context.env_hash = event.env_hash;
            telemetry_context.plan_id = event.plan_id;
            telemetry_context.goal_id = event.goal;
            telemetry_context.episode_completed_at_ms = event.completed_at_ms;
            telemetry_context.subscriber_entry_ms = subscriber_entry_ms;

            const E2PipelineTelemetryRecord record =
                episodicPipelineTelemetryService().recordPipelineRun(stage_timings,
                                                                     telemetry_context);
            g_last_telemetry_record_storage = record;
            setTelemetryTestObservation(&g_last_telemetry_record_storage);
            (void)e2PipelineTelemetryToJson(record);
        } catch (...) {
            setTelemetryTestObservation(nullptr);
        }
    } else {
        setTelemetryTestObservation(nullptr);
    }

    EpisodicLearningRunEnvelope envelope{false, true, "INTEGRATION"};
    EpisodicLearningLogContext ctx;
    ctx.timestamp_ms = event.completed_at_ms;
    ctx.run_id = event.run_id;
    ctx.env_hash = event.env_hash;
    ctx.e2_eval_config = config.toJson();
    const nlohmann::json row =
        episodicLearningSummaryLogRow(ctx, summary, evaluation.passes ? 1 : 0, 1, envelope);
    (void)row;
}

const EpisodicLearningSummary* EvaluationSubscriber::lastSummaryForTests() {
    return g_last_summary_for_tests;
}

const EvaluationDiagnosticsSummary* EvaluationSubscriber::lastRunDiagnosticsForTests() {
    return g_last_run_diagnostics_for_tests;
}

const E2PipelineStageTimings* EvaluationSubscriber::lastStageTimingsForTests() {
    return g_last_stage_timings_for_tests;
}

const E2PipelineTelemetryRecord* EvaluationSubscriber::lastTelemetryRecordForTests() {
    return g_last_telemetry_record_for_tests;
}

void setEvaluationSubscriberPipelineTelemetryEnabled(bool enabled) {
    g_pipeline_telemetry_enabled = enabled;
}

bool evaluationSubscriberPipelineTelemetryEnabled() {
    return g_pipeline_telemetry_enabled;
}

void setEvaluationSubscriberEvalConfigForTests(std::optional<E2EvalConfig> config) {
    g_eval_config_for_tests = std::move(config);
}

void registerEvaluationSubscriber(IEpisodeEventChannel& channel) {
    channel.subscribe(std::make_shared<EvaluationSubscriber>());
}

} // namespace Thoth
