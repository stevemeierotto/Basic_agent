/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C5 path equivalence harness
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "e2_path_equivalence.h"

#include "episodic_evaluation_service.h"
#include "evaluation_subscriber.h"
#include "plan.h"

namespace Thoth {

namespace {

nlohmann::json expectationsToBridgeJson(const EpisodicLearningExpectations& exp) {
    nlohmann::json forbidden = nlohmann::json::array();
    for (const auto& token : exp.forbidden_retrieval_tokens) {
        forbidden.push_back(token);
    }
    return {{"expect_warm_retrieval_hit", exp.expect_warm_retrieval_hit},
            {"lift_constraint",
             exp.lift_constraint == EpisodicLiftConstraint::GTE ? "GTE" : "ABS_LT"},
            {"lift_threshold", exp.lift_threshold},
            {"allow_binary_pass", exp.allow_binary_pass},
            {"include_in_mean_episodic_lift", exp.include_in_mean_episodic_lift},
            {"forbidden_retrieval_tokens", std::move(forbidden)},
            {"retrieval_match_token", exp.retrieval_match_token}};
}

bool chunkRecordsEvaluationRelevantEqual(const RetrievedChunkRecord& a,
                                         const RetrievedChunkRecord& b) {
    return a.chunk_id == b.chunk_id && a.source == b.source && a.source_id == b.source_id &&
           a.validation_status == b.validation_status;
}

bool retrievalProvenanceEvaluationRelevantEqual(const EpisodicRetrievalProvenance& a,
                                                const EpisodicRetrievalProvenance& b) {
    if (a.warm_retrieval_hit != b.warm_retrieval_hit ||
        a.arm_scoring_status != b.arm_scoring_status ||
        a.forbidden_tokens_found != b.forbidden_tokens_found) {
        return false;
    }
    if (a.chunks.size() != b.chunks.size()) {
        return false;
    }
    for (size_t i = 0; i < a.chunks.size(); ++i) {
        if (!chunkRecordsEvaluationRelevantEqual(a.chunks[i], b.chunks[i])) {
            return false;
        }
    }
    return true;
}

EvaluationDiagnosticsContext makeDiagnosticsContext(const E2EvalConfig& config,
                                                    const std::string& run_id,
                                                    const std::string& env_hash) {
    EvaluationDiagnosticsContext ctx;
    ctx.run_id = run_id;
    ctx.env_hash = env_hash;
    ctx.e2_eval_config = config.toJson();
    ctx.fingerprint_hash = episodicEvaluationService().computeFingerprint(config).fingerprint_hash;
    return ctx;
}

void appendMismatch(std::vector<std::string>& mismatches, const std::string& field,
                    const std::string& benchmark_val, const std::string& production_val) {
    mismatches.push_back(field + ": benchmark=" + benchmark_val + " production=" + production_val);
}

nlohmann::json retrievalToBridgeJson(const EpisodicRetrievalProvenance& retrieval) {
    nlohmann::json forbidden = nlohmann::json::array();
    for (const auto& token : retrieval.forbidden_tokens_found) {
        forbidden.push_back(token);
    }
    nlohmann::json chunks = nlohmann::json::array();
    for (const auto& chunk : retrieval.chunks) {
        chunks.push_back({{"chunk_id", chunk.chunk_id},
                          {"source_id", chunk.source_id},
                          {"source", retrievedChunkSourceToString(chunk.source)},
                          {"validation_status",
                           provenanceValidationStatusToString(chunk.validation_status)}});
    }
    return {{"warm_retrieval_hit", retrieval.warm_retrieval_hit},
            {"arm_scoring_status", e2ArmScoringStatusToString(retrieval.arm_scoring_status)},
            {"forbidden_tokens_found", std::move(forbidden)},
            {"chunks", std::move(chunks)}};
}

} // namespace

EpisodeCompleted episodeFromBenchmarkArmsForTests(
    const EpisodicLearningCase& spec,
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm,
    const EpisodicLearningExpectations& expectations,
    const E2RunBlockReason run_block_reason) {
    EpisodeCompleted event;
    event.plan_id = spec.id;
    event.goal = spec.goal;
    event.terminal_state = warm.terminal_state == "COMPLETED" ? "COMPLETED" : "FAILED";
    event.final_success_score = warm.final_success_score;
    event.completed_at_ms = 1'700'000'000'000;
    event.run_id = "e2-c5-synthetic-run";
    event.env_hash = "e2-c5-synthetic-env";

    nlohmann::json steps = nlohmann::json::array();
    for (const auto& chunk : warm.retrieval.chunks) {
        steps.push_back({{"type", static_cast<int>(StepType::RETRIEVAL)},
                         {"step_id", chunk.chunk_id.empty() ? chunk.source_id : chunk.chunk_id}});
    }

    event.plan_snapshot = {{"success_score", cold.final_success_score},
                           {"e2_cold_terminal_state", cold.terminal_state},
                           {"e2_cold_arm_scoring_status",
                            e2ArmScoringStatusToString(cold.arm_scoring_status)},
                           {"e2_warm_arm_scoring_status",
                            e2ArmScoringStatusToString(warm.arm_scoring_status)},
                           {"e2_expectations", expectationsToBridgeJson(expectations)},
                           {"e2_run_block_reason", static_cast<int>(run_block_reason)},
                           {"e2_warm_retrieval_bridge", retrievalToBridgeJson(warm.retrieval)},
                           {"e2_cold_retrieval_bridge", retrievalToBridgeJson(cold.retrieval)},
                           {"steps", std::move(steps)}};
    return event;
}

bool armsEvaluationRelevantEqual(const EpisodicLearningArmObservation& a,
                                 const EpisodicLearningArmObservation& b,
                                 std::string* mismatch_out) {
    if (a.terminal_state != b.terminal_state) {
        if (mismatch_out) {
            *mismatch_out = "terminal_state";
        }
        return false;
    }
    if (a.final_success_score != b.final_success_score) {
        if (mismatch_out) {
            *mismatch_out = "final_success_score";
        }
        return false;
    }
    if (a.arm_scoring_status != b.arm_scoring_status) {
        if (mismatch_out) {
            *mismatch_out = "arm_scoring_status";
        }
        return false;
    }
    if (!retrievalProvenanceEvaluationRelevantEqual(a.retrieval, b.retrieval)) {
        if (mismatch_out) {
            *mismatch_out = "retrieval";
        }
        return false;
    }
    return true;
}

bool expectationsEvaluationRelevantEqual(const EpisodicLearningExpectations& a,
                                         const EpisodicLearningExpectations& b,
                                         std::string* mismatch_out) {
    if (a.expect_warm_retrieval_hit != b.expect_warm_retrieval_hit ||
        a.lift_constraint != b.lift_constraint || a.lift_threshold != b.lift_threshold ||
        a.allow_binary_pass != b.allow_binary_pass ||
        a.include_in_mean_episodic_lift != b.include_in_mean_episodic_lift ||
        a.forbidden_retrieval_tokens != b.forbidden_retrieval_tokens ||
        a.retrieval_match_token != b.retrieval_match_token) {
        if (mismatch_out) {
            *mismatch_out = "expectations";
        }
        return false;
    }
    return true;
}

bool validateMappingFidelityForCase(const EpisodicLearningCase& spec,
                                    const EpisodicLearningArmObservation& cold,
                                    const EpisodicLearningArmObservation& warm,
                                    const EpisodicLearningExpectations& expectations,
                                    std::string* report_out) {
    const EpisodeCompleted event =
        episodeFromBenchmarkArmsForTests(spec, cold, warm, expectations);
    const ProductionEpisodeMapping mapped = mapEpisodeToProductionObservations(event);

    std::string mismatch;
    if (!armsEvaluationRelevantEqual(cold, mapped.cold, &mismatch)) {
        if (report_out) {
            *report_out = spec.id + ": cold arm mismatch at " + mismatch;
        }
        return false;
    }
    if (!armsEvaluationRelevantEqual(warm, mapped.warm, &mismatch)) {
        if (report_out) {
            *report_out = spec.id + ": warm arm mismatch at " + mismatch;
        }
        return false;
    }
    if (!expectationsEvaluationRelevantEqual(expectations, mapped.expectations, &mismatch)) {
        if (report_out) {
            *report_out = spec.id + ": " + mismatch;
        }
        return false;
    }
    return true;
}

E2PathEquivalenceArtifacts runBenchmarkPathArtifacts(
    const std::string& case_id,
    const EpisodicLearningExpectations& expectations,
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm,
    const E2EvalConfig& config,
    const E2RunBlockReason run_block_reason) {
    const IEpisodicEvaluationService& svc = episodicEvaluationService();
    E2PathEquivalenceArtifacts artifacts;
    artifacts.fingerprint = svc.computeFingerprint(config);

    EpisodicLearningCaseEvaluation evaluation =
        svc.evaluateCase(case_id, expectations, cold, warm, config);
    evaluation.run_block_reason = run_block_reason;
    svc.applyCaseResolution(evaluation);
    artifacts.case_eval = evaluation;

    const std::vector<EpisodicLearningCaseEvaluation> case_results{evaluation};
    const std::vector<EpisodicLearningExpectations> expectations_vec{expectations};
    artifacts.summary = svc.summarize(case_results, expectations_vec, config);

    const EvaluationDiagnosticsContext diag_ctx =
        makeDiagnosticsContext(config, "e2-c5-benchmark", "e2-c5-benchmark-env");
    artifacts.diagnostics =
        episodicDiagnosticService().generateRunDiagnostics(artifacts.summary, diag_ctx);
    return artifacts;
}

E2PathEquivalenceArtifacts runProductionPathArtifacts(const EpisodeCompleted& event,
                                                        const E2EvalConfig& pinned_config) {
    setEvaluationSubscriberEvalConfigForTests(pinned_config);
    setEvaluationSubscriberPipelineTelemetryEnabled(false);

    EvaluationSubscriber subscriber;
    subscriber.onEpisodeCompleted(event);

    E2PathEquivalenceArtifacts artifacts;
    artifacts.fingerprint = episodicEvaluationService().computeFingerprint(pinned_config);

    const EpisodicLearningSummary* summary = EvaluationSubscriber::lastSummaryForTests();
    const EvaluationDiagnosticsSummary* diagnostics =
        EvaluationSubscriber::lastRunDiagnosticsForTests();
    if (summary && !summary->case_results.empty()) {
        artifacts.summary = *summary;
        artifacts.case_eval = summary->case_results.front();
    }
    if (diagnostics) {
        artifacts.diagnostics = *diagnostics;
    }

    setEvaluationSubscriberEvalConfigForTests(std::nullopt);
    return artifacts;
}

E2PathEquivalenceDiff diffPathEquivalence(const E2PathEquivalenceArtifacts& benchmark,
                                          const E2PathEquivalenceArtifacts& production) {
    E2PathEquivalenceDiff diff;

    const auto bench_case = pathEquivalenceCaseEvalSnapshot(benchmark.case_eval);
    const auto prod_case = pathEquivalenceCaseEvalSnapshot(production.case_eval);
    if (bench_case != prod_case) {
        diff.equivalent = false;
        diff.mismatches.push_back("case_eval: " + bench_case.dump() + " vs " + prod_case.dump());
    }

    const auto bench_summary = pathEquivalenceSummarySnapshot(benchmark.summary);
    const auto prod_summary = pathEquivalenceSummarySnapshot(production.summary);
    if (bench_summary != prod_summary) {
        diff.equivalent = false;
        diff.mismatches.push_back("summary: " + bench_summary.dump() + " vs " + prod_summary.dump());
    }

    const auto bench_diag = pathEquivalenceDiagnosticsSnapshot(benchmark.diagnostics);
    const auto prod_diag = pathEquivalenceDiagnosticsSnapshot(production.diagnostics);
    if (bench_diag != prod_diag) {
        diff.equivalent = false;
        diff.mismatches.push_back("diagnostics: " + bench_diag.dump() + " vs " + prod_diag.dump());
    }

    if (benchmark.fingerprint.fingerprint_hash != production.fingerprint.fingerprint_hash) {
        diff.equivalent = false;
        appendMismatch(diff.mismatches, "fingerprint_hash", benchmark.fingerprint.fingerprint_hash,
                       production.fingerprint.fingerprint_hash);
    }

    if (benchmark.fingerprint.toJson().value("e2_eval_config", nlohmann::json::object()) !=
        production.fingerprint.toJson().value("e2_eval_config", nlohmann::json::object())) {
        diff.equivalent = false;
        diff.mismatches.push_back("e2_eval_config fingerprint pins differ");
    }

    return diff;
}

nlohmann::json pathEquivalenceCaseEvalSnapshot(const EpisodicLearningCaseEvaluation& eval) {
    nlohmann::json row = {{"case_id", eval.case_id},
                          {"lift", eval.lift},
                          {"passes", eval.passes},
                          {"failure_reason", eval.failure_reason},
                          {"run_block_reason", e2RunBlockReasonToString(eval.run_block_reason)}};
    if (eval.evaluation_resolution.has_value()) {
        row["evaluation_resolution"] =
            e2EvaluationResolutionToString(*eval.evaluation_resolution);
    }
    row["cold_arm_scoring_status"] = e2ArmScoringStatusToString(eval.cold.arm_scoring_status);
    row["warm_arm_scoring_status"] = e2ArmScoringStatusToString(eval.warm.arm_scoring_status);
    return row;
}

nlohmann::json pathEquivalenceSummarySnapshot(const EpisodicLearningSummary& summary) {
    nlohmann::json case_resolutions = nlohmann::json::array();
    for (const auto& eval : summary.case_results) {
        case_resolutions.push_back(pathEquivalenceCaseEvalSnapshot(eval));
    }
    nlohmann::json snap = {{"case_results", case_resolutions},
                           {"mean_episodic_lift", summary.mean_episodic_lift},
                           {"scorable_cases", summary.scorable_cases},
                           {"not_scorable_cases", summary.not_scorable_cases}};
    if (summary.evaluation_resolution.has_value()) {
        snap["evaluation_resolution"] =
            e2EvaluationResolutionToString(*summary.evaluation_resolution);
    }
    return snap;
}

nlohmann::json pathEquivalenceDiagnosticsSnapshot(const EvaluationDiagnosticsSummary& diagnostics) {
    nlohmann::json cases = nlohmann::json::array();
    for (const auto& item : diagnostics.case_diagnostics) {
        nlohmann::json row = {{"case_id", item.case_id},
                              {"diagnosis_bucket", item.diagnosis_bucket},
                              {"failure_classification",
                               e2DiagnosticFailureClassificationToString(
                                   item.failure_classification)}};
        if (item.evaluation_resolution_snapshot.has_value()) {
            row["evaluation_resolution_snapshot"] =
                e2EvaluationResolutionToString(*item.evaluation_resolution_snapshot);
        }
        cases.push_back(std::move(row));
    }
    return {{"diagnosis_bucket", diagnostics.diagnosis_bucket},
            {"failure_classification",
             e2DiagnosticFailureClassificationToString(diagnostics.failure_classification)},
            {"case_diagnostics", cases}};
}

} // namespace Thoth
