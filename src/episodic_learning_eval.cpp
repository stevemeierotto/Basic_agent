/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic memory learning evaluator (table-driven)
 *
 * Protocol: docs/E2_PROTOCOL.md v1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/episodic_learning_eval.h"
#include "../include/e2_strict_enforcement.h"

#include <sstream>

namespace Thoth {

namespace {

bool chunkContentContainsToken(const std::string& content, const std::string& token) {
    return !token.empty() && content.find(token) != std::string::npos;
}

bool armFailClosed(const EpisodicLearningArmObservation& arm) {
    return arm.arm_scoring_status != E2ArmScoringStatus::OK ||
           arm.retrieval.arm_scoring_status != E2ArmScoringStatus::OK;
}

void scanBreakdowns(const nlohmann::json& metadata,
                    const EpisodicLearningExpectations& expectations,
                    EpisodicRetrievalProvenance* prov) {
    if (!prov || !metadata.contains("breakdowns") || !metadata["breakdowns"].is_array()) {
        return;
    }

    const int topK = metadata.value("chunks_retrieved", 0);
    std::size_t idx = 0;
    for (const auto& row : metadata["breakdowns"]) {
        if (topK > 0 && static_cast<int>(idx) >= topK) {
            break;
        }
        ++idx;
        if (!row.is_object()) {
            continue;
        }
        const std::string file_name = row.value("file_name", "");
        const std::string code_text = row.value("code_text", "");

        for (const auto& forbidden : expectations.forbidden_retrieval_tokens) {
            if (!forbidden.empty() &&
                (chunkContentContainsToken(code_text, forbidden) ||
                 chunkContentContainsToken(file_name, forbidden))) {
                prov->forbidden_tokens_found.push_back(forbidden);
            }
        }

        if (file_name.rfind("warm_memory:", 0) != 0 &&
            file_name.rfind("episode:", 0) != 0) {
            continue;
        }

        const bool token_match =
            expectations.retrieval_match_token.empty()
                ? true
                : chunkContentContainsToken(code_text, expectations.retrieval_match_token);

        if (!token_match) {
            continue;
        }

        prov->warm_retrieval_hit = true;
        if (prov->retrieved_chunk_id.empty()) {
            prov->retrieved_chunk_id = file_name;
            if (file_name.rfind("warm_memory:", 0) == 0) {
                prov->retrieved_memory_id = file_name.substr(std::string("warm_memory:").size());
            } else if (file_name.rfind("episode:", 0) == 0) {
                prov->retrieved_memory_id = file_name.substr(std::string("episode:").size());
            }
            prov->matched_token = expectations.retrieval_match_token.empty()
                                        ? code_text.substr(0, std::min<std::size_t>(32, code_text.size()))
                                        : expectations.retrieval_match_token;
        }
    }
}

} // namespace

std::string e2EvalTierToString(E2EvalTier tier) {
    switch (tier) {
        case E2EvalTier::STRICT:
            return "STRICT";
        case E2EvalTier::INTEGRATION:
            return "INTEGRATION";
    }
    return "STRICT";
}

std::string retrievedChunkSourceToString(RetrievedChunkSource source) {
    switch (source) {
        case RetrievedChunkSource::CORPUS:
            return "corpus";
        case RetrievedChunkSource::SYNTHETIC:
            return "synthetic";
        case RetrievedChunkSource::USER:
            return "user";
        case RetrievedChunkSource::EVALUATION:
            return "evaluation";
        case RetrievedChunkSource::SYSTEM:
            return "system";
    }
    return "system";
}

std::string provenanceValidationStatusToString(ProvenanceValidationStatus status) {
    switch (status) {
        case ProvenanceValidationStatus::VALID:
            return "valid";
        case ProvenanceValidationStatus::INVALID:
            return "invalid";
        case ProvenanceValidationStatus::UNTRACED:
            return "untraced";
    }
    return "untraced";
}

std::string e2ArmScoringStatusToString(E2ArmScoringStatus status) {
    switch (status) {
        case E2ArmScoringStatus::OK:
            return "OK";
        case E2ArmScoringStatus::FAILED_RETRIEVAL:
            return "FAILED_RETRIEVAL";
        case E2ArmScoringStatus::FAILED_PROVENANCE:
            return "FAILED_PROVENANCE";
        case E2ArmScoringStatus::FAILED_SEALED_LOG_MUTATION:
            return "FAILED_SEALED_LOG_MUTATION";
        case E2ArmScoringStatus::FAILED_STRICT_BOUNDARY:
            return "FAILED_STRICT_BOUNDARY";
    }
    return "OK";
}

bool E2VersionPin::satisfiesStrictRequirements(bool uses_embeddings) const {
    if (corpus_snapshot_id.empty() || model_version_or_weights_hash.empty()) {
        return false;
    }
    if (retrieval_engine_version.empty()) {
        return false;
    }
    if (uses_embeddings && embedding_model_version.empty()) {
        return false;
    }
    return true;
}

nlohmann::json E2VersionPin::toJson() const {
    return {{"corpus_snapshot_id", corpus_snapshot_id},
            {"model_version_or_weights_hash", model_version_or_weights_hash},
            {"embedding_model_version", embedding_model_version},
            {"retrieval_engine_version", retrieval_engine_version}};
}

E2VersionPin E2VersionPin::fromJson(const nlohmann::json& j) {
    E2VersionPin pin;
    pin.corpus_snapshot_id = j.value("corpus_snapshot_id", "");
    pin.model_version_or_weights_hash = j.value("model_version_or_weights_hash", "");
    pin.embedding_model_version = j.value("embedding_model_version", "");
    pin.retrieval_engine_version = j.value("retrieval_engine_version", "");
    return pin;
}

nlohmann::json E2EvalConfig::toJson() const {
    return {{"tier", e2EvalTierToString(tier)},
            {"official_scoring", officialScoring()},
            {"cross_session_enabled", crossSessionEnabled()},
            {"heuristics_allowed", heuristicsAllowed()},
            {"versions", versions.toJson()}};
}

E2EvalConfig E2EvalConfig::strictDefaults() {
    E2EvalConfig cfg;
    cfg.tier = E2EvalTier::STRICT;
    return cfg;
}

E2EvalConfig E2EvalConfig::integrationDefaults() {
    E2EvalConfig cfg;
    cfg.tier = E2EvalTier::INTEGRATION;
    return cfg;
}

void SealedEpisodeInjectionLog::append(EpisodeInjectionEntry entry) {
    if (sealed_) {
        throw std::logic_error("SealedEpisodeInjectionLog: append after seal");
    }
    entries_.push_back(std::move(entry));
}

void SealedEpisodeInjectionLog::seal() {
    sealed_ = true;
}

nlohmann::json SealedEpisodeInjectionLog::toJson() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& e : entries_) {
        arr.push_back({{"episode_id", e.episode_id},
                       {"source", e.source},
                       {"content_hash", e.content_hash},
                       {"injected_at_ms", e.injected_at_ms}});
    }
    return {{"sealed", sealed_}, {"entries", arr}};
}

bool strictProvenanceValid(const std::vector<RetrievedChunkRecord>& chunks) {
    for (const auto& chunk : chunks) {
        if (chunk.validation_status == ProvenanceValidationStatus::UNTRACED ||
            chunk.source_id.empty()) {
            return false;
        }
    }
    return true;
}

std::string e2OutcomeToString(E2Outcome outcome) {
    switch (outcome) {
        case E2Outcome::SUCCESS:
            return "SUCCESS";
        case E2Outcome::FAILURE:
            return "FAILURE";
    }
    return "FAILURE";
}

EpisodicRetrievalProvenance provenanceFromRetrievalStepResult(
    const nlohmann::json& step_result,
    const EpisodicLearningExpectations& expectations) {
    EpisodicRetrievalProvenance prov;
    if (!step_result.is_object() || !step_result.contains("data") ||
        !step_result["data"].is_object()) {
        return prov;
    }
    const auto& data = step_result["data"];
    if (!data.contains("chunks") || !data["chunks"].is_array()) {
        return prov;
    }

    nlohmann::json pseudo = {{"breakdowns", nlohmann::json::array()}};
    for (const auto& chunk : data["chunks"]) {
        if (!chunk.is_object()) {
            continue;
        }
        pseudo["breakdowns"].push_back(
            {{"file_name", chunk.value("file", "")},
             {"code_text", chunk.value("content", "")}});
    }
    pseudo["chunks_retrieved"] = pseudo["breakdowns"].size();
    return provenanceFromRetrievalDiagnostics(pseudo, expectations);
}

EpisodicRetrievalProvenance provenanceFromRetrievalDiagnostics(
    const nlohmann::json& metadata,
    const EpisodicLearningExpectations& expectations) {
    EpisodicRetrievalProvenance prov;
    scanBreakdowns(metadata, expectations, &prov);
    return prov;
}

bool liftMatchesExpectation(float lift,
                            EpisodicLiftConstraint constraint,
                            float threshold,
                            bool allow_binary_pass,
                            const std::string& cold_terminal_state,
                            const std::string& warm_terminal_state) {
    switch (constraint) {
        case EpisodicLiftConstraint::GTE:
            if (lift >= threshold) {
                return true;
            }
            return allow_binary_pass && cold_terminal_state != "COMPLETED" &&
                   warm_terminal_state == "COMPLETED";
        case EpisodicLiftConstraint::ABS_LT:
            return std::fabs(lift) < threshold;
    }
    return false;
}

bool retrievalHitMatchesExpectation(bool observed_hit, bool expected_hit) {
    return observed_hit == expected_hit;
}

bool forbiddenTokensAbsent(const EpisodicLearningArmObservation& cold,
                           const EpisodicLearningArmObservation& warm,
                           const std::vector<std::string>& forbidden_tokens,
                           std::string* failure_reason) {
    if (forbidden_tokens.empty()) {
        return true;
    }

    for (const auto& token : forbidden_tokens) {
        if (token.empty()) {
            continue;
        }
        for (const auto& found : cold.retrieval.forbidden_tokens_found) {
            if (found == token) {
                if (failure_reason) {
                    *failure_reason = "forbidden token '" + token + "' in cold retrieval";
                }
                return false;
            }
        }
        for (const auto& found : warm.retrieval.forbidden_tokens_found) {
            if (found == token) {
                if (failure_reason) {
                    *failure_reason = "forbidden token '" + token + "' in warm retrieval";
                }
                return false;
            }
        }
    }
    return true;
}

EpisodicLearningCaseEvaluation evaluateEpisodicLearningCase(
    const std::string& case_id,
    const EpisodicLearningExpectations& expectations,
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm,
    const E2EvalConfig& config) {
    EpisodicLearningCaseEvaluation eval;
    eval.case_id = case_id;
    eval.cold = cold;
    eval.warm = warm;
    eval.lift = warm.final_success_score - cold.final_success_score;

    if (config.tier == E2EvalTier::STRICT) {
        if (!config.versions.satisfiesStrictRequirements(true)) {
            eval.passes = false;
            eval.failure_reason = "STRICT missing required version pins";
            return eval;
        }
        if (config.versions.retrieval_engine_version != kE2StrictRetrievalEngineVersion) {
            eval.passes = false;
            eval.failure_reason = "STRICT retrieval_engine_version mismatch";
            return eval;
        }
        if (armFailClosed(cold)) {
            eval.passes = false;
            eval.failure_reason = "cold arm fail-closed: " +
                                  e2ArmScoringStatusToString(cold.arm_scoring_status);
            return eval;
        }
        if (armFailClosed(warm)) {
            eval.passes = false;
            eval.failure_reason = "warm arm fail-closed: " +
                                  e2ArmScoringStatusToString(warm.arm_scoring_status);
            return eval;
        }
        if (!strictProvenanceValid(cold.retrieval.chunks)) {
            eval.passes = false;
            eval.failure_reason = "cold arm FAILED_PROVENANCE";
            return eval;
        }
        if (!strictProvenanceValid(warm.retrieval.chunks)) {
            eval.passes = false;
            eval.failure_reason = "warm arm FAILED_PROVENANCE";
            return eval;
        }
    }

    if (!retrievalHitMatchesExpectation(warm.retrieval.warm_retrieval_hit,
                                        expectations.expect_warm_retrieval_hit)) {
        eval.passes = false;
        eval.failure_reason = "warm_retrieval_hit expected " +
                              std::string(expectations.expect_warm_retrieval_hit ? "true" : "false") +
                              " got " +
                              std::string(warm.retrieval.warm_retrieval_hit ? "true" : "false");
        return eval;
    }

    if (!forbiddenTokensAbsent(cold, warm, expectations.forbidden_retrieval_tokens,
                               &eval.failure_reason)) {
        eval.passes = false;
        return eval;
    }

    if (!liftMatchesExpectation(eval.lift,
                                expectations.lift_constraint,
                                expectations.lift_threshold,
                                expectations.allow_binary_pass,
                                cold.terminal_state,
                                warm.terminal_state)) {
        eval.passes = false;
        if (expectations.lift_constraint == EpisodicLiftConstraint::GTE) {
            eval.failure_reason = "lift " + std::to_string(eval.lift) + " < " +
                                  std::to_string(expectations.lift_threshold);
        } else {
            eval.failure_reason = "|lift| " + std::to_string(std::fabs(eval.lift)) + " >= " +
                                  std::to_string(expectations.lift_threshold);
        }
        return eval;
    }

    eval.passes = true;
    return eval;
}

EpisodicLearningSummary summarizeEpisodicLearning(
    const std::vector<EpisodicLearningCaseEvaluation>& case_results,
    const std::vector<EpisodicLearningExpectations>& case_expectations,
    const E2EvalConfig& config) {
    EpisodicLearningSummary summary;
    summary.scoring_tier = config.tier;
    summary.official_scoring = config.officialScoring();
    summary.case_results = case_results;

    if (config.tier == E2EvalTier::STRICT &&
        !config.versions.satisfiesStrictRequirements(true)) {
        summary.outcome = E2Outcome::FAILURE;
        summary.outcome_rationale = "STRICT summary rejected: missing version pins";
        return summary;
    }
    if (config.tier == E2EvalTier::STRICT &&
        config.versions.retrieval_engine_version != kE2StrictRetrievalEngineVersion) {
        summary.outcome = E2Outcome::FAILURE;
        summary.outcome_rationale = "STRICT summary rejected: retrieval_engine_version mismatch";
        return summary;
    }

    float lift_sum = 0.0f;
    int lift_count = 0;

    for (std::size_t i = 0; i < case_results.size(); ++i) {
        if (i < case_expectations.size() && case_expectations[i].include_in_mean_episodic_lift) {
            lift_sum += case_results[i].lift;
            ++lift_count;
        }
    }

    summary.mean_episodic_lift =
        lift_count > 0 ? lift_sum / static_cast<float>(lift_count) : 0.0f;

    if (config.tier == E2EvalTier::INTEGRATION) {
        summary.outcome = E2Outcome::FAILURE;
        summary.outcome_rationale = "INTEGRATION tier is non-scoring diagnostic mode";
        return summary;
    }

    bool all_pass = !case_results.empty();
    for (const auto& c : case_results) {
        if (!c.passes) {
            all_pass = false;
            break;
        }
    }

    if (all_pass && summary.mean_episodic_lift > 0.0f) {
        summary.outcome = E2Outcome::SUCCESS;
        summary.outcome_rationale = "all cases pass table expectations and mean_episodic_lift > 0";
    } else {
        summary.outcome = E2Outcome::FAILURE;
        std::ostringstream oss;
        if (!all_pass) {
            oss << "one or more cases failed expectations";
        } else {
            oss << "mean_episodic_lift <= 0";
        }
        summary.outcome_rationale = oss.str();
    }

    return summary;
}

nlohmann::json retrievedChunkToJson(const RetrievedChunkRecord& chunk) {
    return {{"chunk_id", chunk.chunk_id},
            {"source", retrievedChunkSourceToString(chunk.source)},
            {"source_id", chunk.source_id},
            {"validation_status", provenanceValidationStatusToString(chunk.validation_status)}};
}

nlohmann::json provenanceToJson(const EpisodicRetrievalProvenance& prov) {
    nlohmann::json chunks = nlohmann::json::array();
    for (const auto& c : prov.chunks) {
        chunks.push_back(retrievedChunkToJson(c));
    }
    return {{"warm_retrieval_hit", prov.warm_retrieval_hit},
            {"retrieved_memory_id", prov.retrieved_memory_id},
            {"retrieved_chunk_id", prov.retrieved_chunk_id},
            {"matched_token", prov.matched_token},
            {"forbidden_tokens_found", prov.forbidden_tokens_found},
            {"arm_scoring_status", e2ArmScoringStatusToString(prov.arm_scoring_status)},
            {"chunks", chunks}};
}

nlohmann::json armObservationToJson(const EpisodicLearningArmObservation& arm) {
    return {{"arm", arm.arm_label},
            {"terminal_state", arm.terminal_state},
            {"final_success_score", arm.final_success_score},
            {"arm_scoring_status", e2ArmScoringStatusToString(arm.arm_scoring_status)},
            {"retrieval", provenanceToJson(arm.retrieval)},
            {"wall_clock_ms", arm.wall_clock_ms},
            {"planning_time_ms", arm.planning_time_ms},
            {"total_tokens", arm.total_tokens}};
}

nlohmann::json caseEvaluationToJson(const EpisodicLearningCaseEvaluation& eval) {
    return {{"case_id", eval.case_id},
            {"lift", eval.lift},
            {"passes", eval.passes},
            {"failure_reason", eval.failure_reason},
            {"cold", armObservationToJson(eval.cold)},
            {"warm", armObservationToJson(eval.warm)}};
}

} // namespace Thoth
