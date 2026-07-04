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
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct Plan;

class EmbeddingEngine;
class IndexManager;

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

/**
 * Run-level block — why an evaluation arm/run was invalid for scoring (Phase B).
 * Distinct from arm status and from derived e2_outcome.
 */
enum class E2RunBlockReason {
    NONE,
    RUNTIME_HEURISTIC_GUARD,
    WIRING_GATE,
    STRICT_BOUNDARY_VIOLATION,
    PROVENANCE_VIOLATION,
};

/** Canonical evaluation resolution (Phase B) — computed in B3; unset in B1. */
enum class E2EvaluationResolution {
    SCORED_SUCCESS,
    SCORED_FAILURE,
    NOT_SCORABLE,
};

std::string e2EvalTierToString(E2EvalTier tier);
std::string retrievedChunkSourceToString(RetrievedChunkSource source);
std::string provenanceValidationStatusToString(ProvenanceValidationStatus status);
std::string e2ArmScoringStatusToString(E2ArmScoringStatus status);
std::string e2RunBlockReasonToString(E2RunBlockReason reason);
std::string e2RunBlockReasonToProtocolString(E2RunBlockReason reason);
std::string e2EvaluationResolutionToString(E2EvaluationResolution resolution);

/**
 * B1 stub — always returns NONE. B2 owns non-NONE mapping from guard/wiring throws.
 * Any non-NONE run_block_reason in B1 is invalid unless explicitly set in B2 wiring.
 */
E2RunBlockReason e2RunBlockReasonFromException(const std::exception& e);

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

struct EpisodicLearningCase;
struct E2StrictRetrievalResult;

/**
 * STRICT-only: build a sealed injection log from the frozen case table.
 * @param arm_label "cold" or "warm"
 * @param builder_timestamp_ms shared by every entry in this invocation (deterministic)
 */
SealedEpisodeInjectionLog buildStrictInjectionLogFromCaseTable(
    const EpisodicLearningCase& case_spec,
    const std::string& arm_label,
    std::int64_t builder_timestamp_ms);

/** Test-only: count buildStrictInjectionLogFromCaseTable invocations (E2-09b). */
void setStrictInjectionLogBuilderCallCounterForTests(int* counter);
void clearStrictInjectionLogBuilderCallCounterForTests();

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
    /** B1: default NONE. Non-NONE only legal after B2 explicit assignment. */
    E2RunBlockReason run_block_reason = E2RunBlockReason::NONE;
    /** B3 computes; unset in B1. */
    std::optional<E2EvaluationResolution> evaluation_resolution;
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
    /** B3 rollup placeholders — zero in B1. */
    int scorable_cases = 0;
    int not_scorable_cases = 0;
    /** B3 computes; unset in B1. */
    std::optional<E2EvaluationResolution> evaluation_resolution;
};

std::string e2OutcomeToString(E2Outcome outcome);

/** STRICT: any untraced chunk → false (arm must fail closed). */
bool strictProvenanceValid(const std::vector<RetrievedChunkRecord>& chunks);

EpisodicRetrievalProvenance provenanceFromRetrievalStepResult(
    const nlohmann::json& step_result,
    const EpisodicLearningExpectations& expectations);

/**
 * A3.0a — episodic content expected when arm semantics inject a non-empty plant episode.
 * Used for vacuous-retrieval guard at the STRICT evaluation boundary.
 */
bool strictEpisodicContentRequired(const EpisodicLearningCase& case_spec,
                                   const std::string& arm_label);

/**
 * STRICT evaluation boundary — maps kernel output only (no case-table bypass).
 * @param episodic_content_required from strictEpisodicContentRequired()
 */
EpisodicRetrievalProvenance provenanceFromStrictRetrievalResult(
    const E2StrictRetrievalResult& retrieval,
    const EpisodicLearningExpectations& expectations,
    bool episodic_content_required);

/** Reconstruct kernel result from Executive RETRIEVAL step (STRICT branch output). */
E2StrictRetrievalResult e2StrictRetrievalResultFromRetrievalStep(
    const nlohmann::json& step_result);

/** A4 equivalence — chunk ids, ordering, status (success and failure paths). */
bool e2StrictRetrievalResultsEquivalent(const E2StrictRetrievalResult& harness,
                                        const E2StrictRetrievalResult& executive);

/** First STRICT E2 RETRIEVAL step result in a completed plan (A4). */
std::optional<E2StrictRetrievalResult> executiveStrictRetrievalFromPlan(const Plan& plan);

/** E2 harness corpus helper — accepts cold TfIdf zero-embed via keyword fallback. */
void addEpisodicEvalCorpusChunk(EmbeddingEngine* engine,
                                IndexManager* idx,
                                const std::string& text,
                                const std::string& file_name = "e2-distractor.md");

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

E2ArmScoringStatus caseArmStatusForResolution(
    const EpisodicLearningArmObservation& cold,
    const EpisodicLearningArmObservation& warm);

E2EvaluationResolution resolveEvaluation(E2RunBlockReason run_block_reason,
                                         E2ArmScoringStatus arm_status);

void applyCaseEvaluationResolution(EpisodicLearningCaseEvaluation& eval);

E2Outcome deriveE2OutcomeFromResolution(E2EvaluationResolution resolution, bool table_passes);

/** First completed RETRIEVAL step block reason (struct field read — not JSON inference). */
E2RunBlockReason runBlockReasonFromPlan(const Plan& plan);

EpisodicLearningSummary summarizeEpisodicLearning(
    const std::vector<EpisodicLearningCaseEvaluation>& case_results,
    const std::vector<EpisodicLearningExpectations>& case_expectations,
    const E2EvalConfig& config);

/** B4 — JSONL envelope fields shared by case and summary log rows. */
struct EpisodicLearningLogContext {
    std::int64_t timestamp_ms = 0;
    std::string run_id;
    std::string env_hash;
    nlohmann::json evaluation_fingerprint = nlohmann::json::object();
    nlohmann::json e2_eval_config = nlohmann::json::object();
};

/** B5 — harness run envelope; stage label for JSONL only (never consulted inside scored loop). */
struct EpisodicLearningRunEnvelope {
    bool official_scoring = false;
    bool scoring_enabled = false;
    std::string wiring_stage;
};

/**
 * B4 export-only — derived at serialization time; never a persisted source-of-truth field.
 * Empty when resolution is unset or NOT_SCORABLE.
 */
std::optional<E2Outcome> e2OutcomeForExport(const EpisodicLearningCaseEvaluation& eval);
std::optional<E2Outcome> e2OutcomeForExport(E2EvaluationResolution resolution, bool table_passes);
std::optional<E2Outcome> e2OutcomeForExport(const EpisodicLearningSummary& summary);

nlohmann::json notScorableByReasonMap(
    const std::vector<EpisodicLearningCaseEvaluation>& case_results);
float successRateForExport(const std::vector<EpisodicLearningCaseEvaluation>& case_results);

nlohmann::json provenanceToJson(const EpisodicRetrievalProvenance& prov);
nlohmann::json armObservationToJson(const EpisodicLearningArmObservation& arm);
nlohmann::json caseEvaluationToJson(const EpisodicLearningCaseEvaluation& eval);
nlohmann::json episodicLearningSummaryToJson(const EpisodicLearningSummary& summary);
nlohmann::json episodicLearningCaseLogRow(const EpisodicLearningLogContext& ctx,
                                          const EpisodicLearningCaseEvaluation& eval);
nlohmann::json episodicLearningSummaryLogRow(const EpisodicLearningLogContext& ctx,
                                             const EpisodicLearningSummary& summary,
                                             int cases_passed,
                                             std::size_t case_count,
                                             const EpisodicLearningRunEnvelope& envelope = {});

/** B5 — E2-28 scoped equivalence snapshot (excludes timestamps / log ordering). */
nlohmann::json episodicLearningScopedEquivalenceSnapshot(
    const EpisodicLearningSummary& summary,
    const nlohmann::json& evaluation_fingerprint,
    const nlohmann::json& e2_eval_config);
bool episodicLearningScopedEquivalenceEqual(const nlohmann::json& a, const nlohmann::json& b);

/**
 * B5 — fingerprint mismatch diagnosis bucket when snapshots differ.
 * @return 0 equivalent; 1 config; 2 corpus; 3 retrieval nondeterminism; 4 semantic drift.
 */
int episodicLearningFingerprintMismatchBucket(const nlohmann::json& snapshot_a,
                                              const nlohmann::json& snapshot_b,
                                              const std::string& corpus_hash_a,
                                              const std::string& corpus_hash_b);
nlohmann::json retrievedChunkToJson(const RetrievedChunkRecord& chunk);

} // namespace Thoth

#endif // THOTH_EPISODIC_LEARNING_EVAL_H
