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
#include <string>
#include <unordered_map>
#include <vector>

struct GragDiagnostics;

namespace Thoth {

inline constexpr const char* kCm1SchemaVersion = "cm1-phase1-v1";
inline constexpr const char* kCm1EstimatorStatus = "none_active";
inline constexpr const char* kCm1BreadthVersion = "cm1-breadth-2";
inline constexpr const char* kCm1BreadthLabel = "uncalibrated_evidence_breadth";

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

} // namespace Thoth

#endif
