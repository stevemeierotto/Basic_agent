/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 episodic memory learning evaluator (table-driven)
 *
 * Protocol: docs/E2_PROTOCOL.md v1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODIC_LEARNING_EVAL_H
#define THOTH_EPISODIC_LEARNING_EVAL_H

#include "json.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thoth {

/** Protocol v1.2 — immutable during in-flight E2 STRICT runs. */
constexpr float kEpisodicLearningLiftMargin = 0.10f;

/** Frozen scoring function identifier — see docs/E2_PROTOCOL.md § Scoring freeze. */
constexpr const char* kEpisodicLearningScoringFunction =
    "ExecutiveController::calculate_trajectory_score";

/** Official scoring (STRICT) vs diagnostic-only (INTEGRATION). */
enum class E2EvalTier {
    STRICT,
    INTEGRATION,
};

/** Chunk origin — required on every retrieved chunk in STRICT. */
enum class RetrievedChunkSource {
    CORPUS,
    SYNTHETIC,
    USER,
    EVALUATION,
    SYSTEM,
};

enum class ProvenanceValidationStatus {
    VALID,
    INVALID,
    UNTRACED,
};

/** Fail-closed arm status (STRICT). */
enum class E2ArmScoringStatus {
    OK,
    FAILED_RETRIEVAL,
    FAILED_PROVENANCE,
    FAILED_SEALED_LOG_MUTATION,
    FAILED_STRICT_BOUNDARY,
};

std::string e2EvalTierToString(E2EvalTier tier);
std::string retrievedChunkSourceToString(RetrievedChunkSource source);
std::string provenanceValidationStatusToString(ProvenanceValidationStatus status);
std::string e2ArmScoringStatusToString(E2ArmScoringStatus status);

/** Required version pins — see E2_PROTOCOL.md § Version pinning. */
struct E2VersionPin {
    /** Required. E1 corpus / index fingerprint. */
    std::string corpus_snapshot_id;
    /** Required. LLM id, mock label, or weights hash. */
    std::string model_version_or_weights_hash;
    /** Required when embeddings used. */
    std::string embedding_model_version;
    /** Required when retrieval engine is versioned separately. */
    std::string retrieval_engine_version;

    bool satisfiesStrictRequirements(bool uses_embeddings) const;
    nlohmann::json toJson() const;
    static E2VersionPin fromJson(const nlohmann::json& j);
};

/**
 * Evaluation mode configuration — canonical authority (not env-string modes).
 * STRICT: official scoring. INTEGRATION: diagnostic only.
 */
struct E2EvalConfig {
    E2EvalTier tier = E2EvalTier::STRICT;
    E2VersionPin versions;

    bool officialScoring() const { return tier == E2EvalTier::STRICT; }
    bool crossSessionEnabled() const { return tier == E2EvalTier::INTEGRATION; }
    bool heuristicsAllowed() const { return tier == E2EvalTier::INTEGRATION; }

    nlohmann::json toJson() const;

    /** Test scaffolding only — empty version pins; STRICT scoring MUST reject this config. */
    static E2EvalConfig strictDefaults();
    static E2EvalConfig integrationDefaults();
};

/** Single episode declared before arm execution — frozen at seal. */
struct EpisodeInjectionEntry {
    std::string episode_id;
    /** synthetic | user | evaluation | system */
    std::string source;
    std::string content;
    std::string content_hash;
    std::int64_t injected_at_ms = 0;
};

/**
 * Immutable episode log for one evaluation arm.
 *
 * Enforcement: deep copy at seal(); sealed_ flag; mutating APIs throw
 * std::logic_error after seal(). const& alone is NOT sufficient — callers
 * must invoke seal() before passing to retrieval.
 */
class SealedEpisodeInjectionLog {
public:
    SealedEpisodeInjectionLog() = default;

    /** Append entry — illegal after seal(). */
    void append(EpisodeInjectionEntry entry);

    /** Freeze log; subsequent append/clear throw. */
    void seal();

    bool isSealed() const { return sealed_; }

    const std::vector<EpisodeInjectionEntry>& entries() const { return entries_; }

    nlohmann::json toJson() const;

private:
    std::vector<EpisodeInjectionEntry> entries_;
    bool sealed_ = false;
};

/** Per-chunk provenance (STRICT — all fields required for scoring). */
struct RetrievedChunkRecord {
    std::string chunk_id;
    RetrievedChunkSource source = RetrievedChunkSource::CORPUS;
    std::string source_id;
    std::string content;
    ProvenanceValidationStatus validation_status = ProvenanceValidationStatus::UNTRACED;
};

enum class EpisodicLiftConstraint {
    GTE,
    ABS_LT,
};

/** Per-case expectations — evaluator has no case-ID logic. */
struct EpisodicLearningExpectations {
    bool expect_warm_retrieval_hit = false;
    EpisodicLiftConstraint lift_constraint = EpisodicLiftConstraint::GTE;
    float lift_threshold = kEpisodicLearningLiftMargin;
    bool allow_binary_pass = true;
    std::vector<std::string> forbidden_retrieval_tokens;
    bool include_in_mean_episodic_lift = false;
    std::string retrieval_match_token;
};

struct EpisodicRetrievalProvenance {
    bool warm_retrieval_hit = false;
    std::string retrieved_memory_id;
    std::string retrieved_chunk_id;
    std::string matched_token;
    std::vector<std::string> forbidden_tokens_found;
    std::vector<RetrievedChunkRecord> chunks;
    E2ArmScoringStatus arm_scoring_status = E2ArmScoringStatus::OK;
};

struct EpisodicLearningArmObservation {
    std::string arm_label;
    std::string terminal_state;
    float final_success_score = 0.0f;
    EpisodicRetrievalProvenance retrieval;
    E2ArmScoringStatus arm_scoring_status = E2ArmScoringStatus::OK;
    std::int64_t wall_clock_ms = 0;
    std::int64_t planning_time_ms = 0;
    std::int64_t total_tokens = 0;
};

struct EpisodicLearningCaseEvaluation {
    std::string case_id;
    EpisodicLearningArmObservation cold;
    EpisodicLearningArmObservation warm;
    float lift = 0.0f;
    bool passes = false;
    std::string failure_reason;
};

enum class E2Outcome {
    SUCCESS,
    FAILURE,
};

struct EpisodicLearningSummary {
    E2EvalTier scoring_tier = E2EvalTier::STRICT;
    bool official_scoring = true;
    std::vector<EpisodicLearningCaseEvaluation> case_results;
    float mean_episodic_lift = 0.0f;
    E2Outcome outcome = E2Outcome::FAILURE;
    std::string outcome_rationale;
};

std::string e2OutcomeToString(E2Outcome outcome);

/** STRICT: any untraced chunk → false (arm must fail closed). */
bool strictProvenanceValid(const std::vector<RetrievedChunkRecord>& chunks);

EpisodicRetrievalProvenance provenanceFromRetrievalStepResult(
    const nlohmann::json& step_result,
    const EpisodicLearningExpectations& expectations);

EpisodicRetrievalProvenance provenanceFromRetrievalDiagnostics(
    const nlohmann::json& metadata,
    const EpisodicLearningExpectations& expectations);

bool liftMatchesExpectation(float lift,
                            EpisodicLiftConstraint constraint,
                            float threshold,
                            bool allow_binary_pass,
                            const std::string& cold_terminal_state,
                            const std::string& warm_terminal_state);

bool retrievalHitMatchesExpectation(bool observed_hit, bool expected_hit);

bool forbiddenTokensAbsent(const EpisodicLearningArmObservation& cold,
                           const EpisodicLearningArmObservation& warm,
                           const std::vector<std::string>& forbidden_tokens,
                           std::string* failure_reason);

EpisodicLearningCaseEvaluation evaluateEpisodicLearningCase(
    const std::string& case_id,
    const EpisodicLearningExpectations& expectations,
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm,
    const E2EvalConfig& config);

EpisodicLearningSummary summarizeEpisodicLearning(
    const std::vector<EpisodicLearningCaseEvaluation>& case_results,
    const std::vector<EpisodicLearningExpectations>& case_expectations,
    const E2EvalConfig& config);

nlohmann::json provenanceToJson(const EpisodicRetrievalProvenance& prov);
nlohmann::json armObservationToJson(const EpisodicLearningArmObservation& arm);
nlohmann::json caseEvaluationToJson(const EpisodicLearningCaseEvaluation& eval);
nlohmann::json retrievedChunkToJson(const RetrievedChunkRecord& chunk);

} // namespace Thoth

#endif // THOTH_EPISODIC_LEARNING_EVAL_H
