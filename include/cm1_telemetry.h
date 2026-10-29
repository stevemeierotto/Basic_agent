/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — CM-1 Phase 1 observational confidence telemetry builders
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CM1_TELEMETRY_H
#define THOTH_CM1_TELEMETRY_H

#include "json.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct GragDiagnostics;
class IndexManager;

namespace Thoth {

inline constexpr const char* kCm1SchemaVersion = "cm1-phase1-v1";
inline constexpr const char* kCm1EstimatorStatus = "none_active";
inline constexpr const char* kCm1BreadthVersion = "cm1-breadth-2";
inline constexpr const char* kCm1BreadthLabel = "uncalibrated_evidence_breadth";
inline constexpr const char* kCm1EvidenceVersion = "cm1-evidence-1";
inline constexpr int kCm1EvidenceClaimCap = 8;
inline constexpr int kCm1EvidenceNumPredict = 256;
inline constexpr const char* kCm1CorroborationProtocolId = "cm1-corroboration";
inline constexpr const char* kCm1CorroborationProtocolVersion = "v1";
inline constexpr const char* kCm1CorroborationReviewerId = "thoth-corroboration";
inline constexpr const char* kCm1CorroborationVerifierVersion = "cm1-corroboration-1";
inline constexpr const char* kCm1CorroborationMethodId = "held-out-passage-read";
/** One passage. source_ref is sha256Hex of that passage's raw content. */
inline constexpr const char* kCm1HeldOutPassageSource = "held_out_passage";
/**
 * More than one passage was assessed and none was cited.
 * source_ref is the sorted unique content hashes, one per line.
 * It identifies the batch. It is not a single passage.
 */
inline constexpr const char* kCm1HeldOutBatchSource = "held_out_batch";

struct Cm1SourceGroup {
    std::string file_name;
    std::string document_id; // empty means unavailable
    std::string origin_id = "unknown";
    bool indexed_chunk_count_available = false;
    int indexed_chunk_count = 0;
    int pool_chunk_count = 0;
    int scored_chunk_count = 0;
    int selected_chunk_count = 0;
    int included_in_synthesis_count = 0;
    bool included_in_synthesis_available = false;
    bool highest_final_score_available = false;
    float highest_final_score = 0.0f;
    bool highest_query_sim_available = false;
    float highest_query_sim = 0.0f;
    bool mean_final_score_available = false;
    float mean_final_score = 0.0f;
    int duplicate_full_content_hash_count = 0;
    float graph_score_max = 0.0f;
    bool graph_score_present = false;
    std::string agreement_status = "not_evaluated";
};

struct Cm1SnapshotBuildInput {
    std::string plan_id;
    std::string step_id;
    std::string snapshot_type;      // retrieval_observed | synthesis_observed
    std::string lifecycle_stage;    // after_retrieval | after_synthesis
    std::int64_t observed_at_ms = 0;
    const GragDiagnostics* diagnostics = nullptr;
    /** file_name -> indexed slice count from the live index. Empty = denominator unavailable. */
    std::unordered_map<std::string, int> indexed_counts_by_file;
    bool indexed_counts_available = false;
    /** Final top-k selection: file + content for each selected chunk. */
    std::vector<std::pair<std::string, std::string>> selected_chunks;
    /** Content SHA-256 of chunks that entered the synthesis prompt. Empty + flag false = unavailable. */
    std::vector<std::string> included_content_sha256;
    bool included_in_synthesis_available = false;
    bool synthesis_context_truncated = false;
    bool synthesis_context_truncated_available = false;
};

struct Cm1SnapshotRecord {
    std::string snapshot_id;
    std::string plan_id;
    std::string step_id;
    std::string snapshot_type;
    std::string lifecycle_stage;
    std::string features_json;
    std::string measurement_status; // complete | incomplete | empty_retrieval
    std::string schema_version = kCm1SchemaVersion;
    std::string estimator_status = kCm1EstimatorStatus;
    std::int64_t created_at_ms = 0;
};

struct Cm1AnswerVersionRecord {
    std::string answer_id;
    std::string plan_id;
    std::string step_id;
    std::string generation_id;
    std::string content_sha256;
    std::int64_t created_at_ms = 0;
};

/** Deterministic snapshot id. Same inputs always collide on INSERT OR IGNORE. */
std::string makeCm1SnapshotId(const std::string& plan_id,
                              const std::string& step_id,
                              const std::string& snapshot_type,
                              const std::string& lifecycle_stage);

std::string makeCm1AnswerId(const std::string& plan_id,
                            const std::string& step_id,
                            const std::string& generation_id);

/**
 * Build a Phase-1 retrieval or synthesis snapshot from existing diagnostics.
 * Does not call the model, embedder, or vector store.
 */
Cm1SnapshotRecord buildCm1Snapshot(const Cm1SnapshotBuildInput& input);

/**
 * Answer provenance when a synthesis response exists. generation_id must be the
 * real provider id; empty generation_id returns an empty record (not fabricated).
 */
Cm1AnswerVersionRecord buildCm1AnswerVersion(const std::string& plan_id,
                                             const std::string& step_id,
                                             const std::string& generation_id,
                                             const std::string& answer_text,
                                             std::int64_t created_at_ms);

bool cm1TelemetryEnabled();

/** Derived breadth rows. Default off so official EGAR runs stay unchanged. */
bool cm1BreadthEnabled();

/**
 * One derived breadth row for a snapshot. Prior rows are earlier snapshots for
 * the same plan, oldest first, not including current. Raw snapshot JSON is not
 * modified. Does not emit goal confidence or predicted correctness.
 */
struct Cm1BreadthEstimate {
    std::string estimate_id;
    std::string snapshot_id;
    std::string plan_id;
    std::string step_id;
    std::string estimator_version = kCm1BreadthVersion;
    std::string payload_json;
    std::int64_t created_at_ms = 0;
};

Cm1BreadthEstimate buildCm1BreadthEstimate(
    const Cm1SnapshotRecord& current,
    const std::vector<Cm1SnapshotRecord>& prior_snapshots);

/**
 * User impression of one answer version. Thumbs, the 1–10 rating, and the
 * written explanation are optional. Absent means the user did not set that
 * field. This is not a verified factual label.
 */
struct Cm1UserFeedback {
    std::string feedback_id;
    std::string answer_id;
    std::string content_sha256;
    std::string thumbs_status = "absent"; // absent | up | down
    std::string quality_rating_status = "absent"; // absent | present
    int quality_rating = 0;
    std::string explanation;
    std::string supersedes_feedback_id;
    int revision_index = 1;
    std::int64_t created_at_ms = 0;
};

struct Cm1FeedbackSubmission {
    std::string answer_id;
    std::string answer_text;
    std::string thumbs_status = "absent";
    bool include_quality_rating = false;
    int quality_rating = 0;
    std::string explanation;
};

/** Independently verified factual bin for one answer version. */
struct Cm1VerifiedLabel {
    std::string label_id;
    std::string answer_id;
    std::string content_sha256;
    std::string protocol_id;
    std::string reviewer_id;
    int factual_bin = 0;
    std::string supersedes_label_id;
    int revision_index = 1;
    std::int64_t created_at_ms = 0;
};

struct Cm1VerifiedLabelSubmission {
    std::string answer_id;
    std::string answer_text;
    std::string protocol_id;
    std::string reviewer_id;
    int factual_bin = 0;
};

enum class Cm1RecordDisposition {
    Insert,
    NoRecord,
    Rejected
};

struct Cm1UserFeedbackDraft {
    Cm1RecordDisposition disposition = Cm1RecordDisposition::Rejected;
    std::string message;
    Cm1UserFeedback record;
};

struct Cm1VerifiedLabelDraft {
    Cm1RecordDisposition disposition = Cm1RecordDisposition::Rejected;
    std::string message;
    Cm1VerifiedLabel record;
};

/**
 * Decide the next feedback row. A blank submission with no prior row writes
 * nothing. A blank submission after a prior row records an explicit clear.
 * Hash mismatch and a missing answer version are rejected.
 */
Cm1UserFeedbackDraft prepareCm1UserFeedback(
    const Cm1AnswerVersionRecord* answer,
    const Cm1UserFeedback* latest,
    const Cm1FeedbackSubmission& submission,
    std::int64_t created_at_ms);

Cm1VerifiedLabelDraft prepareCm1VerifiedLabel(
    const Cm1AnswerVersionRecord* answer,
    const Cm1VerifiedLabel* latest,
    const Cm1VerifiedLabelSubmission& submission,
    std::int64_t created_at_ms);

/**
 * One claim copied from a persisted evidence-judgment payload.
 * Claim boundaries are not recomputed.
 */
struct Cm1StoredClaim {
    std::string claim_id;
    std::string claim_sha256;
    std::string claim_text;
    std::string answer_id;
    std::string answer_content_sha256;
    std::string evidence_quote;
    std::string evidence_chunk_sha256;
};

/**
 * Append-only finding for one persisted claim.
 * outcome is factual correctness. evidence_label is a passage assessment.
 * A non-empty evidence label requires outcome unverifiable.
 * calibration_eligibility and egar_eligibility are stored separately.
 * This checkpoint's writer accepts only provisional and excluded.
 */
struct Cm1ClaimVerification {
    std::string verification_id;
    std::string claim_id;
    std::string claim_sha256;
    std::string answer_id;
    std::string content_sha256;
    std::string protocol_id;
    std::string protocol_version;
    std::string reviewer_id;
    std::string verifier_kind;
    std::string verifier_version;
    std::string method_id;
    std::string outcome;
    std::string evidence_label;
    std::string source_kind;
    std::string source_ref;
    std::string calibration_eligibility;
    std::string egar_eligibility;
    std::string revision_reason;
    std::string supersedes_verification_id;
    int revision_index = 1;
    std::int64_t created_at_ms = 0;
};

struct Cm1ClaimVerificationSubmission {
    std::string claim_id;
    std::string claim_text;
    std::string answer_id;
    std::string answer_text;
    std::string protocol_id;
    std::string protocol_version;
    std::string reviewer_id;
    std::string verifier_kind;
    std::string verifier_version;
    std::string method_id;
    std::string outcome;
    std::string evidence_label;
    std::string source_kind;
    std::string source_ref;
    std::string calibration_eligibility = "provisional";
    std::string egar_eligibility = "excluded";
    std::string revision_reason;
};

struct Cm1ClaimVerificationDraft {
    Cm1RecordDisposition disposition = Cm1RecordDisposition::Rejected;
    std::string message;
    Cm1ClaimVerification record;
};

/** Derived evidence judgments. Default off so official EGAR runs stay unchanged. */
bool cm1EvidenceEnabled();

/**
 * Split an answer into claim strings. List markers are removed. Decimal points,
 * abbreviations, and backtick spans stay inside a claim. Short claims are kept.
 */
std::vector<std::string> segmentCm1EvidenceClaims(const std::string& answer_text);

struct Cm1EvidenceChunk {
    std::string content_sha256;
    std::string content;
};

struct Cm1EvidenceCompletion {
    bool called = false;
    bool ok = false;
    std::string error;
    std::string text;
    std::string finish_reason;
    std::string generation_id;
};

struct Cm1EvidenceJudgmentRecord {
    std::string judgment_id;
    std::string answer_id;
    std::string content_sha256;
    std::string estimator_version = kCm1EvidenceVersion;
    std::string payload_json;
    std::int64_t created_at_ms = 0;
};

std::string buildCm1EvidencePrompt(const std::string& claim,
                                   const std::vector<Cm1EvidenceChunk>& included_chunks);

/**
 * True when the controller state is still the state observed before judgment
 * released the mutex. A newer state must not be replaced with IDLE.
 */
inline bool cm1EvidenceStateUnchanged(int state_at_release, int state_now) {
    return state_at_release == state_now;
}

/**
 * Copy one claim object out of a persisted judgment payload.
 * A missing claim returns nullopt. The answer is not segmented again.
 */
std::optional<Cm1StoredClaim> findCm1StoredClaim(const Cm1EvidenceJudgmentRecord& judgment,
                                                 const std::string& claim_id);

/**
 * Decide the next claim-verification row from persisted answer and claim
 * records. An initial row starts a chain. A correction appends to that chain.
 * Authorization, and any eligibility other than provisional / excluded, is rejected.
 */
Cm1ClaimVerificationDraft prepareCm1ClaimVerification(
    const Cm1AnswerVersionRecord* answer,
    const Cm1StoredClaim* claim,
    const Cm1ClaimVerification* latest,
    const Cm1ClaimVerificationSubmission& submission,
    std::int64_t created_at_ms);

/**
 * Structural contract for a verification row against one persisted answer and claim.
 * Provisional calibration eligibility and excluded EGAR eligibility are required.
 * A non-empty evidence_label requires outcome unverifiable.
 * Outcome true or false requires an empty evidence_label.
 * Authorization revisions are rejected. This does not read the database.
 */
bool cm1ClaimVerificationConforms(const Cm1ClaimVerification& record,
                                  const Cm1AnswerVersionRecord& answer,
                                  const Cm1StoredClaim& claim);

/**
 * Judge up to kCm1EvidenceClaimCap claims within budget_seconds.
 * caller is not invoked when synthesis was mocked, inclusion is unknown, or an
 * included chunk text is missing. An empty answer id returns nullopt.
 * now_ms drives the deadline. An empty function uses a steady clock.
 * The per-call timeout is the remaining whole seconds. At under one second the
 * remaining claims are not judged.
 */
std::optional<Cm1EvidenceJudgmentRecord> runCm1EvidenceJudgment(
    const Cm1AnswerVersionRecord& answer,
    const std::string& answer_text,
    const std::vector<Cm1EvidenceChunk>& included_chunks,
    bool inclusion_known,
    bool evidence_texts_complete,
    bool mock_synthesis,
    const std::function<Cm1EvidenceCompletion(const std::string& prompt, int timeout_seconds)>& caller,
    std::int64_t created_at_ms,
    int budget_seconds = 900,
    const std::function<std::int64_t()>& now_ms = {});

/**
 * Content hashes of chunks that entered synthesis for one persisted step.
 * An empty vector means the inclusion list is known and nothing was included.
 * nullopt means the list cannot be recovered.
 */
struct Cm1SynthesisInclusion {
    std::vector<std::string> content_sha256;
};

/** Caller-supplied passage. content is hashed with sha256Hex and no other normalization. */
struct Cm1CorroborationPassage {
    std::string content;
    std::string file_name;
};

struct Cm1ClaimCorroborationResult {
    bool wrote = false;
    std::string reason;
    std::optional<Cm1ClaimVerification> record;
};

class MemoryRepository;

/** Default off. Unset, 0, false, and off leave the corroboration writer idle. */
bool cm1CorroborationEnabled();

/**
 * Assess one stored claim against caller-supplied passages.
 * Does not retrieve, re-segment, or change the evidence judgment.
 * A disabled writer returns without reading the model.
 * Timeout, invalid output, ambiguous evidence, and no eligible passage write nothing.
 */
Cm1ClaimCorroborationResult runCm1ClaimCorroboration(
    MemoryRepository& repository,
    const std::string& answer_id,
    const std::string& claim_id,
    const std::string& answer_text,
    const std::vector<Cm1CorroborationPassage>& passages,
    const std::function<Cm1EvidenceCompletion(const std::string& prompt, int timeout_seconds)>& caller,
    std::int64_t created_at_ms,
    int budget_seconds = 900,
    const std::function<std::int64_t()>& now_ms = {});

/**
 * Retrieve index passages for one stored claim and pass the eligible ones to
 * runCm1ClaimCorroboration. The query is the stored claim text.
 * An embedding failure and an empty index both come back as no passages.
 * The writer then reports no_eligible_passage and does not call the model.
 * A model label of insufficient_evidence is a separate stored judgment.
 * Returns reason indexing, without searching, while the index reports that it
 * is indexing. Returns provenance_missing, without searching, when a required
 * synthesis source file cannot be recovered. Does not call the controller.
 */
Cm1ClaimCorroborationResult runCm1ClaimDiscovery(
    IndexManager& index,
    MemoryRepository& repository,
    const std::string& answer_id,
    const std::string& claim_id,
    const std::string& answer_text,
    const std::function<Cm1EvidenceCompletion(const std::string& prompt, int timeout_seconds)>& caller,
    std::int64_t created_at_ms,
    int budget_seconds = 900,
    int top_k = 8,
    const std::function<std::int64_t()>& now_ms = {});

/**
 * One stored claim that can be discovered later.
 * answer_text is the synthesis response recovered from the persisted plan.
 */
struct Cm1DiscoveryCandidate {
    std::string answer_id;
    std::string claim_id;
    std::string answer_text;
};

/**
 * Read stored judgments and return available claims with no CM-1 corroboration
 * row. The answer version must match the judgment hash, the plan must contain
 * that same answer text, and every included or cited source must resolve to a
 * file. Does not retrieve, call the model, write, or read THOTH_CM1_CORROBORATION.
 */
std::vector<Cm1DiscoveryCandidate> selectCm1DiscoveryCandidates(MemoryRepository& repository);

} // namespace Thoth

#endif
