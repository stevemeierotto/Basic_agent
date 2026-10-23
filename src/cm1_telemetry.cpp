/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — CM-1 Phase 1 observational confidence telemetry builders
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/cm1_telemetry.h"

#include "../include/alp_sha256.h"
#include "../include/grag_diagnostics.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>
#include <unordered_set>

namespace Thoth {

namespace {

struct Accum {
    Cm1SourceGroup group;
    std::vector<float> scores;
    std::unordered_map<std::string, int> content_hash_counts;
};

std::string normalizePathKey(const std::string& file_name) {
    return file_name.empty() ? std::string("<unknown>") : file_name;
}

} // namespace

bool cm1TelemetryEnabled() {
    const char* env = std::getenv("THOTH_CM1_TELEMETRY");
    if (!env || !*env) return true;
    return std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "off";
}

bool cm1BreadthEnabled() {
    const char* env = std::getenv("THOTH_CM1_BREADTH");
    if (!env || !*env) return false;
    return std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "off";
}

bool cm1EvidenceEnabled() {
    const char* env = std::getenv("THOTH_CM1_EVIDENCE");
    if (!env || !*env) return false;
    return std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "off";
}

std::string makeCm1SnapshotId(const std::string& plan_id,
                              const std::string& step_id,
                              const std::string& snapshot_type,
                              const std::string& lifecycle_stage) {
    return "cm1:" + plan_id + ":" + step_id + ":" + snapshot_type + ":" + lifecycle_stage;
}

std::string makeCm1AnswerId(const std::string& plan_id,
                            const std::string& step_id,
                            const std::string& generation_id) {
    return plan_id + ":" + step_id + ":" + generation_id;
}

Cm1AnswerVersionRecord buildCm1AnswerVersion(const std::string& plan_id,
                                             const std::string& step_id,
                                             const std::string& generation_id,
                                             const std::string& answer_text,
                                             std::int64_t created_at_ms) {
    Cm1AnswerVersionRecord record;
    if (plan_id.empty() || step_id.empty() || generation_id.empty() || answer_text.empty()) {
        return record;
    }
    record.plan_id = plan_id;
    record.step_id = step_id;
    record.generation_id = generation_id;
    record.content_sha256 = sha256Hex(answer_text);
    record.answer_id = makeCm1AnswerId(plan_id, step_id, generation_id);
    record.created_at_ms = created_at_ms;
    return record;
}

Cm1SnapshotRecord buildCm1Snapshot(const Cm1SnapshotBuildInput& input) {
    Cm1SnapshotRecord record;
    record.plan_id = input.plan_id;
    record.step_id = input.step_id;
    record.snapshot_type = input.snapshot_type;
    record.lifecycle_stage = input.lifecycle_stage;
    record.created_at_ms = input.observed_at_ms;
    record.schema_version = kCm1SchemaVersion;
    record.estimator_status = kCm1EstimatorStatus;
    record.snapshot_id =
        makeCm1SnapshotId(input.plan_id, input.step_id, input.snapshot_type, input.lifecycle_stage);

    nlohmann::json features = nlohmann::json::object();
    features["schema_version"] = kCm1SchemaVersion;
    features["estimator_status"] = kCm1EstimatorStatus;
    features["agreement_status"] = "not_evaluated";
    features["evidence_state"] = "no_evidence";
    features["origin_independence"] = "unknown";

    std::map<std::string, Accum> by_file;
    bool breakdowns_present = false;
    int pool_chunk_total = 0;
    int scored_chunk_total = 0;
    int graph_activations = 0;
    float graph_max_contribution = 0.0f;
    bool graph_meta_present = false;

    if (input.diagnostics) {
        const GragDiagnostics& d = *input.diagnostics;
        features["chunks_retrieved"] = d.chunks_retrieved;
        features["chunks_reranked"] = d.chunks_reranked;
        features["grag_alpha"] = d.alpha;
        features["grag_routing_mode"] = d.routing_mode;
        features["scoring_type"] = d.scoring_type;
        features["final_scores"] = d.final_scores;
        graph_activations = d.graph_activations;
        graph_max_contribution = d.graph_max_contribution;
        graph_meta_present = true;
        features["graph_activations"] = graph_activations;
        features["graph_max_contribution"] = graph_max_contribution;
        features["graph_total_nodes"] = d.graph_total_nodes;
        features["graph_total_edges"] = d.graph_total_edges;
        features["graph_avg_weight"] = d.graph_avg_weight;

        breakdowns_present = !d.breakdowns.empty();
        for (const auto& b : d.breakdowns) {
            const std::string key = normalizePathKey(b.file_name);
            Accum& acc = by_file[key];
            if (acc.group.file_name.empty()) {
                acc.group.file_name = b.file_name.empty() ? key : b.file_name;
                acc.group.origin_id = "unknown";
                acc.group.agreement_status = "not_evaluated";
            }
            ++acc.group.pool_chunk_count;
            ++acc.group.scored_chunk_count;
            ++pool_chunk_total;
            ++scored_chunk_total;
            acc.scores.push_back(b.final_score);
            if (!acc.group.highest_final_score_available
                || b.final_score > acc.group.highest_final_score) {
                acc.group.highest_final_score = b.final_score;
                acc.group.highest_final_score_available = true;
            }
            if (std::isfinite(b.query_sim)
                && (!acc.group.highest_query_sim_available
                    || b.query_sim > acc.group.highest_query_sim)) {
                acc.group.highest_query_sim = b.query_sim;
                acc.group.highest_query_sim_available = true;
            }
            if (!b.code_text.empty()) {
                const std::string h = sha256Hex(b.code_text);
                acc.content_hash_counts[h] += 1;
            }
            if (b.graph_score > 0.0f) {
                acc.group.graph_score_present = true;
                if (b.graph_score > acc.group.graph_score_max) {
                    acc.group.graph_score_max = b.graph_score;
                }
            }
        }
    } else {
        features["diagnostics_present"] = false;
    }

    std::unordered_map<std::string, int> selected_by_file;
    std::unordered_map<std::string, std::unordered_set<std::string>> selected_hashes_by_file;
    for (const auto& [file, content] : input.selected_chunks) {
        const std::string key = normalizePathKey(file);
        selected_by_file[key] += 1;
        if (by_file.find(key) == by_file.end()) {
            Accum& acc = by_file[key];
            acc.group.file_name = file.empty() ? key : file;
            acc.group.origin_id = "unknown";
            acc.group.agreement_status = "not_evaluated";
        }
        if (!content.empty()) {
            selected_hashes_by_file[key].insert(sha256Hex(content));
        }
    }

    std::unordered_set<std::string> included_set(input.included_content_sha256.begin(),
                                                 input.included_content_sha256.end());

    if (input.indexed_counts_available) {
        for (const auto& [file, count] : input.indexed_counts_by_file) {
            const std::string key = normalizePathKey(file);
            if (by_file.find(key) == by_file.end() && selected_by_file.find(key) == selected_by_file.end()) {
                continue;
            }
            Accum& acc = by_file[key];
            if (acc.group.file_name.empty()) {
                acc.group.file_name = file.empty() ? key : file;
                acc.group.origin_id = "unknown";
            }
            acc.group.indexed_chunk_count_available = true;
            acc.group.indexed_chunk_count = count;
        }
    }

    nlohmann::json source_groups = nlohmann::json::array();
    int selected_total = 0;
    int included_total = 0;
    for (auto& [key, acc] : by_file) {
        acc.group.selected_chunk_count = selected_by_file.count(key) ? selected_by_file[key] : 0;
        selected_total += acc.group.selected_chunk_count;

        if (!acc.scores.empty()) {
            float sum = 0.0f;
            for (float s : acc.scores) sum += s;
            acc.group.mean_final_score = sum / static_cast<float>(acc.scores.size());
            acc.group.mean_final_score_available = true;
        }

        int duplicate_extras = 0;
        for (const auto& [hash, count] : acc.content_hash_counts) {
            if (count > 1) duplicate_extras += (count - 1);
        }
        acc.group.duplicate_full_content_hash_count = duplicate_extras;

        if (input.included_in_synthesis_available) {
            acc.group.included_in_synthesis_available = true;
            int included = 0;
            for (const auto& hash : selected_hashes_by_file[key]) {
                if (included_set.count(hash)) ++included;
            }
            acc.group.included_in_synthesis_count = included;
            included_total += included;
        } else {
            acc.group.included_in_synthesis_available = false;
        }

        if (input.indexed_counts_available && input.indexed_counts_by_file.count(key)) {
            acc.group.indexed_chunk_count_available = true;
            acc.group.indexed_chunk_count = input.indexed_counts_by_file.at(key);
        }

        nlohmann::json g = {
            {"file_name", acc.group.file_name},
            {"document_id", nlohmann::json()},
            {"origin_id", acc.group.origin_id},
            {"agreement_status", acc.group.agreement_status},
            {"pool_chunk_count", acc.group.pool_chunk_count},
            {"scored_chunk_count", acc.group.scored_chunk_count},
            {"selected_chunk_count", acc.group.selected_chunk_count},
            {"duplicate_full_content_hash_count", acc.group.duplicate_full_content_hash_count},
        };
        if (acc.group.document_id.empty()) {
            g["document_id"] = nullptr;
            g["document_id_status"] = "unavailable";
        } else {
            g["document_id"] = acc.group.document_id;
            g["document_id_status"] = "available";
        }
        if (acc.group.indexed_chunk_count_available) {
            g["indexed_chunk_count"] = acc.group.indexed_chunk_count;
            g["indexed_chunk_count_status"] = "available";
            // Coverage ratio intentionally omitted: Phase 1 does not invent a
            // percentage. Callers must not treat selected/pool counts as a census.
        } else {
            g["indexed_chunk_count_status"] = "unavailable";
        }
        if (acc.group.highest_final_score_available) {
            g["highest_final_score"] = acc.group.highest_final_score;
        } else {
            g["highest_final_score_status"] = "unavailable";
        }
        if (acc.group.highest_query_sim_available) {
            g["highest_query_sim"] = acc.group.highest_query_sim;
            g["highest_query_sim_status"] = "available";
        } else {
            g["highest_query_sim_status"] = "unavailable";
        }
        if (acc.group.mean_final_score_available) {
            g["mean_final_score"] = acc.group.mean_final_score;
            g["score_sample_size"] = static_cast<int>(acc.scores.size());
        }
        if (acc.group.graph_score_present) {
            g["graph_score_max"] = acc.group.graph_score_max;
            g["graph_independent_evidence"] = false;
        }
        if (acc.group.included_in_synthesis_available) {
            g["included_in_synthesis_count"] = acc.group.included_in_synthesis_count;
            g["included_in_synthesis_status"] = "available";
        } else {
            g["included_in_synthesis_status"] = "unavailable";
        }
        source_groups.push_back(std::move(g));
    }

    features["source_groups"] = source_groups;
    features["pool_chunk_total"] = pool_chunk_total;
    features["scored_chunk_total"] = scored_chunk_total;
    features["selected_chunk_total"] = selected_total;
    features["distinct_source_paths_in_pool"] = static_cast<int>(by_file.size());
    features["breakdowns_present"] = breakdowns_present;
    features["indexed_counts_available"] = input.indexed_counts_available;
    features["graph_meta_present"] = graph_meta_present;

    if (input.included_in_synthesis_available) {
        features["included_in_synthesis_total"] = included_total;
        features["included_in_synthesis_status"] = "available";
    } else {
        features["included_in_synthesis_status"] = "unavailable";
    }

    if (input.synthesis_context_truncated_available) {
        features["synthesis_context_truncated"] = input.synthesis_context_truncated;
        features["synthesis_context_truncated_status"] = "available";
    } else {
        features["synthesis_context_truncated_status"] = "unavailable";
    }

    if (selected_total > 0 || pool_chunk_total > 0) {
        features["evidence_state"] = "evidence_recorded";
    }

    std::string status = "complete";
    nlohmann::json incomplete_reasons = nlohmann::json::array();
    if (!input.diagnostics) {
        status = "incomplete";
        incomplete_reasons.push_back("diagnostics_missing");
    } else if (!breakdowns_present && selected_total == 0) {
        if (input.diagnostics->chunks_retrieved == 0) {
            status = "empty_retrieval";
            features["evidence_state"] = "no_evidence";
        } else {
            status = "incomplete";
            incomplete_reasons.push_back("breakdowns_missing");
        }
    } else if (!breakdowns_present) {
        status = "incomplete";
        incomplete_reasons.push_back("breakdowns_missing");
    }
    if (!input.indexed_counts_available) {
        incomplete_reasons.push_back("indexed_denominator_unavailable");
        // Denominator missing does not by itself mark the whole snapshot incomplete
        // when scores are present; it is recorded on each source group.
    }
    if (input.snapshot_type == "synthesis_observed" && !input.included_in_synthesis_available) {
        incomplete_reasons.push_back("synthesis_inclusion_unavailable");
    }
    if (!incomplete_reasons.empty() && status == "complete") {
        // Soft incompleteness flags only; keep complete when core scores exist.
        features["soft_incomplete_reasons"] = incomplete_reasons;
    } else if (!incomplete_reasons.empty()) {
        features["incomplete_reasons"] = incomplete_reasons;
    }

    record.measurement_status = status;
    features["measurement_status"] = status;
    record.features_json = features.dump();
    return record;
}

namespace {

float clampUnit(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

float saturateMass(float mass) {
    return mass / (mass + 1.0f);
}

struct PathObs {
    bool query_known = false;
    float query_sim = 0.0f;
    bool inclusion_known = false;
    int included = 0;
};

void absorbGroup(PathObs& path, const nlohmann::json& group) {
    if (group.contains("highest_query_sim") && group["highest_query_sim"].is_number()) {
        const float query = group["highest_query_sim"].get<float>();
        if (!path.query_known || query > path.query_sim) {
            path.query_sim = query;
            path.query_known = true;
        }
    }
    if (group.value("included_in_synthesis_status", "") == "available") {
        path.inclusion_known = true;
        path.included = std::max(path.included, group.value("included_in_synthesis_count", 0));
    }
}

std::map<std::string, PathObs> pathsFromFeatures(const nlohmann::json& features) {
    std::map<std::string, PathObs> paths;
    if (!features.contains("source_groups") || !features["source_groups"].is_array()) {
        return paths;
    }
    for (const auto& group : features["source_groups"]) {
        if (!group.is_object()) continue;
        const std::string file = group.value("file_name", "");
        if (file.empty()) continue;
        absorbGroup(paths[file], group);
    }
    return paths;
}

struct MassResult {
    bool available = false;
    float mass = 0.0f;
    float index = 0.0f;
    int path_count = 0;
};

MassResult massFromPaths(const std::map<std::string, PathObs>& paths, bool used_only) {
    MassResult result;
    bool missing_strength = false;
    bool any_selected = false;
    for (const auto& [file, path] : paths) {
        (void)file;
        if (used_only) {
            if (!path.inclusion_known) {
                return result;
            }
            if (path.included <= 0) continue;
            any_selected = true;
            if (!path.query_known) {
                missing_strength = true;
                continue;
            }
        } else if (!path.query_known) {
            continue;
        }
        result.mass += clampUnit(path.query_sim);
        result.path_count += 1;
    }
    if (used_only && missing_strength) {
        return result;
    }
    if (!used_only && result.path_count == 0) {
        return result;
    }
    if (used_only && !any_selected) {
        result.available = true;
        result.mass = 0.0f;
        result.index = 0.0f;
        result.path_count = 0;
        return result;
    }
    result.available = true;
    result.index = saturateMass(result.mass);
    return result;
}

nlohmann::json featuresOf(const Cm1SnapshotRecord& snapshot) {
    try {
        const auto parsed = nlohmann::json::parse(snapshot.features_json);
        if (parsed.is_object()) return parsed;
    } catch (...) {
    }
    return nlohmann::json::object();
}

} // namespace

Cm1BreadthEstimate buildCm1BreadthEstimate(
    const Cm1SnapshotRecord& current,
    const std::vector<Cm1SnapshotRecord>& prior_snapshots) {
    Cm1BreadthEstimate estimate;
    estimate.snapshot_id = current.snapshot_id;
    estimate.plan_id = current.plan_id;
    estimate.step_id = current.step_id;
    estimate.estimator_version = kCm1BreadthVersion;
    estimate.created_at_ms = current.created_at_ms;
    estimate.estimate_id = std::string("cm1b2:") + current.snapshot_id;

    const nlohmann::json features = featuresOf(current);
    const auto current_paths = pathsFromFeatures(features);
    const bool empty_retrieval = current.measurement_status == "empty_retrieval";
    const bool inclusion_known =
        features.value("included_in_synthesis_status", "") == "available";

    bool current_missing_query = false;
    for (const auto& [file, path] : current_paths) {
        (void)file;
        if (!path.query_known) current_missing_query = true;
    }

    MassResult retrieved;
    if (current_paths.empty() && empty_retrieval) {
        retrieved.available = true;
        retrieved.mass = 0.0f;
        retrieved.index = 0.0f;
    } else if (!current_missing_query && !current_paths.empty()) {
        retrieved = massFromPaths(current_paths, false);
    }

    std::map<std::string, PathObs> goal_paths;
    bool saw_retrieval = false;
    bool all_retrievals_empty = true;
    bool goal_query_incomplete = false;
    auto absorbRetrieval = [&](const Cm1SnapshotRecord& snapshot) {
        if (snapshot.snapshot_type != "retrieval_observed") return;
        saw_retrieval = true;
        if (snapshot.measurement_status != "empty_retrieval") {
            all_retrievals_empty = false;
        }
        for (const auto& [file, path] : pathsFromFeatures(featuresOf(snapshot))) {
            if (!path.query_known) {
                goal_query_incomplete = true;
                continue;
            }
            PathObs& dest = goal_paths[file];
            if (!dest.query_known || path.query_sim > dest.query_sim) {
                dest.query_sim = path.query_sim;
                dest.query_known = true;
            }
        }
    };
    for (const auto& prior : prior_snapshots) {
        absorbRetrieval(prior);
    }
    if (current.snapshot_type == "retrieval_observed") {
        absorbRetrieval(current);
    }

    MassResult used;
    if (inclusion_known) {
        auto used_paths = current_paths;
        for (auto& [file, path] : used_paths) {
            const auto found = goal_paths.find(file);
            if (found != goal_paths.end() && found->second.query_known
                && (!path.query_known || found->second.query_sim > path.query_sim)) {
                path.query_sim = found->second.query_sim;
                path.query_known = true;
            }
        }
        used = massFromPaths(used_paths, true);
    }

    MassResult goal_retrieved;
    if (saw_retrieval && !goal_query_incomplete) {
        if (all_retrievals_empty && goal_paths.empty()) {
            goal_retrieved.available = true;
            goal_retrieved.mass = 0.0f;
            goal_retrieved.index = 0.0f;
        } else {
            goal_retrieved = massFromPaths(goal_paths, false);
        }
    }

    const Cm1SnapshotRecord* latest_synthesis = nullptr;
    if (current.snapshot_type == "synthesis_observed") {
        latest_synthesis = &current;
    } else {
        for (auto it = prior_snapshots.rbegin(); it != prior_snapshots.rend(); ++it) {
            if (it->snapshot_type == "synthesis_observed") {
                latest_synthesis = &(*it);
                break;
            }
        }
    }

    MassResult goal_used;
    if (latest_synthesis) {
        const nlohmann::json synth = featuresOf(*latest_synthesis);
        if (synth.value("included_in_synthesis_status", "") == "available") {
            auto included_paths = pathsFromFeatures(synth);
            for (auto& [file, path] : included_paths) {
                const auto found = goal_paths.find(file);
                if (found != goal_paths.end() && found->second.query_known) {
                    path.query_sim = found->second.query_sim;
                    path.query_known = true;
                }
            }
            goal_used = massFromPaths(included_paths, true);
        }
    }

    auto putIndex = [](nlohmann::json& payload, const char* status_key, const char* value_key,
                       const char* mass_key, const char* count_key, const MassResult& result) {
        if (!result.available) {
            payload[status_key] = "unavailable";
            return;
        }
        payload[status_key] = "available";
        payload[value_key] = result.index;
        payload[mass_key] = result.mass;
        payload[count_key] = result.path_count;
    };

    nlohmann::json payload = {
        {"estimator_version", kCm1BreadthVersion},
        {"label", kCm1BreadthLabel},
        {"agreement_status", "not_evaluated"},
        {"snapshot_id", current.snapshot_id},
        {"plan_id", current.plan_id},
        {"step_id", current.step_id},
    };
    if (features.contains("synthesis_context_truncated_status")) {
        payload["synthesis_context_truncated_status"] =
            features["synthesis_context_truncated_status"];
        if (features.contains("synthesis_context_truncated")) {
            payload["synthesis_context_truncated"] = features["synthesis_context_truncated"];
        }
    }
    putIndex(payload, "evidence_breadth_retrieved_status", "evidence_breadth_retrieved",
             "retrieved_mass", "retrieved_path_count", retrieved);
    putIndex(payload, "evidence_breadth_used_status", "evidence_breadth_used",
             "used_mass", "used_path_count", used);

    MassResult step;
    if (current.snapshot_type == "retrieval_observed") {
        step = retrieved;
    } else if (current.snapshot_type == "synthesis_observed") {
        step = used;
    }
    putIndex(payload, "step_evidence_breadth_status", "step_evidence_breadth",
             "step_mass", "step_path_count", step);
    if (!saw_retrieval) {
        payload["goal_evidence_breadth_retrieved_status"] = "not_recorded";
    } else {
        putIndex(payload, "goal_evidence_breadth_retrieved_status",
                 "goal_evidence_breadth_retrieved", "goal_retrieved_mass",
                 "goal_retrieved_path_count", goal_retrieved);
    }
    if (!latest_synthesis) {
        payload["goal_evidence_breadth_used_status"] = "not_recorded";
    } else {
        putIndex(payload, "goal_evidence_breadth_used_status", "goal_evidence_breadth_used",
                 "goal_used_mass", "goal_used_path_count", goal_used);
    }

    estimate.payload_json = payload.dump();
    return estimate;
}

namespace {

std::string trimCopy(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool validThumbs(const std::string& status) {
    return status == "absent" || status == "up" || status == "down";
}

} // namespace

Cm1UserFeedbackDraft prepareCm1UserFeedback(
    const Cm1AnswerVersionRecord* answer,
    const Cm1UserFeedback* latest,
    const Cm1FeedbackSubmission& submission,
    std::int64_t created_at_ms) {
    Cm1UserFeedbackDraft draft;
    if (!answer || answer->answer_id.empty() || answer->content_sha256.empty()) {
        draft.message = "No answer version is recorded for this synthesis.";
        return draft;
    }
    if (submission.answer_id != answer->answer_id) {
        draft.message = "Feedback does not match the answer version.";
        return draft;
    }
    if (!validThumbs(submission.thumbs_status)) {
        draft.message = "Thumbs must be absent, up, or down.";
        return draft;
    }
    if (submission.include_quality_rating
        && (submission.quality_rating < 1 || submission.quality_rating > 10)) {
        draft.message = "Quality rating must be from 1 to 10.";
        return draft;
    }
    const std::string presented = sha256Hex(submission.answer_text);
    if (presented != answer->content_sha256) {
        draft.message = "The displayed answer does not match the stored version.";
        return draft;
    }

    const std::string explanation = trimCopy(submission.explanation);
    const bool blank = submission.thumbs_status == "absent"
        && !submission.include_quality_rating
        && explanation.empty();
    if (blank && !latest) {
        draft.disposition = Cm1RecordDisposition::NoRecord;
        draft.message = "No feedback recorded.";
        return draft;
    }

    draft.disposition = Cm1RecordDisposition::Insert;
    draft.record.answer_id = answer->answer_id;
    draft.record.content_sha256 = answer->content_sha256;
    draft.record.thumbs_status = submission.thumbs_status;
    if (submission.include_quality_rating) {
        draft.record.quality_rating_status = "present";
        draft.record.quality_rating = submission.quality_rating;
    }
    draft.record.explanation = explanation;
    draft.record.revision_index = latest ? latest->revision_index + 1 : 1;
    if (latest) {
        draft.record.supersedes_feedback_id = latest->feedback_id;
    }
    draft.record.feedback_id =
        answer->answer_id + ":f:" + std::to_string(draft.record.revision_index);
    draft.record.created_at_ms = created_at_ms;
    draft.message = blank ? "Feedback cleared." : "Feedback recorded.";
    return draft;
}

Cm1VerifiedLabelDraft prepareCm1VerifiedLabel(
    const Cm1AnswerVersionRecord* answer,
    const Cm1VerifiedLabel* latest,
    const Cm1VerifiedLabelSubmission& submission,
    std::int64_t created_at_ms) {
    Cm1VerifiedLabelDraft draft;
    if (!answer || answer->answer_id.empty() || answer->content_sha256.empty()) {
        draft.message = "No answer version is recorded for this synthesis.";
        return draft;
    }
    if (submission.answer_id != answer->answer_id) {
        draft.message = "Verified label does not match the answer version.";
        return draft;
    }
    if (submission.protocol_id.empty() || submission.reviewer_id.empty()) {
        draft.message = "A verified label needs a protocol and a reviewer.";
        return draft;
    }
    if (submission.factual_bin < 1 || submission.factual_bin > 10) {
        draft.message = "Verified factual bin must be from 1 to 10.";
        return draft;
    }
    if (sha256Hex(submission.answer_text) != answer->content_sha256) {
        draft.message = "The labeled answer does not match the stored version.";
        return draft;
    }
    draft.disposition = Cm1RecordDisposition::Insert;
    draft.record.answer_id = answer->answer_id;
    draft.record.content_sha256 = answer->content_sha256;
    draft.record.protocol_id = submission.protocol_id;
    draft.record.reviewer_id = submission.reviewer_id;
    draft.record.factual_bin = submission.factual_bin;
    draft.record.revision_index = latest ? latest->revision_index + 1 : 1;
    if (latest) {
        draft.record.supersedes_label_id = latest->label_id;
    }
    draft.record.label_id = answer->answer_id + ":v:" + submission.protocol_id + ":"
        + std::to_string(draft.record.revision_index);
    draft.record.created_at_ms = created_at_ms;
    draft.message = "Verified label recorded.";
    return draft;
}

namespace {

std::string trimEnds(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool isListMarkerLine(const std::string& line, std::size_t& content_begin) {
    std::size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) {
        ++i;
    }
    if (i >= line.size()) return false;
    if ((line[i] == '-' || line[i] == '*') && i + 1 < line.size()
        && std::isspace(static_cast<unsigned char>(line[i + 1]))) {
        content_begin = i + 2;
        while (content_begin < line.size()
               && std::isspace(static_cast<unsigned char>(line[content_begin]))) {
            ++content_begin;
        }
        return true;
    }
    std::size_t j = i;
    if (!std::isdigit(static_cast<unsigned char>(line[j]))) return false;
    while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) {
        ++j;
    }
    if (j >= line.size() || (line[j] != '.' && line[j] != ')')) return false;
    if (j + 1 >= line.size() || !std::isspace(static_cast<unsigned char>(line[j + 1]))) {
        return false;
    }
    content_begin = j + 2;
    while (content_begin < line.size()
           && std::isspace(static_cast<unsigned char>(line[content_begin]))) {
        ++content_begin;
    }
    return true;
}

bool abbreviationBeforePeriod(const std::string& text, std::size_t period_index) {
    std::size_t begin = period_index;
    while (begin > 0 && std::isalpha(static_cast<unsigned char>(text[begin - 1]))) {
        --begin;
    }
    std::string token = text.substr(begin, period_index - begin);
    for (char& ch : token) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (token.size() == 1) return true;
    return token == "vs" || token == "fig" || token == "eq" || token == "eg" || token == "ie";
}

std::vector<std::string> splitClaimSentences(const std::string& block) {
    std::vector<std::string> claims;
    bool in_ticks = false;
    std::size_t start = 0;
    for (std::size_t i = 0; i < block.size(); ++i) {
        const char ch = block[i];
        if (ch == '`') {
            in_ticks = !in_ticks;
            continue;
        }
        if (in_ticks) continue;
        if (ch != '.' && ch != '!' && ch != '?') continue;
        const bool boundary = (i + 1 == block.size())
            || std::isspace(static_cast<unsigned char>(block[i + 1]));
        if (!boundary) continue;
        if (ch == '.' && i > 0 && i + 1 < block.size()
            && std::isdigit(static_cast<unsigned char>(block[i - 1]))
            && std::isdigit(static_cast<unsigned char>(block[i + 1]))) {
            continue;
        }
        if (ch == '.' && abbreviationBeforePeriod(block, i)) continue;
        const std::string piece = trimEnds(block.substr(start, i + 1 - start));
        if (!piece.empty()) claims.push_back(piece);
        start = i + 1;
    }
    const std::string tail = trimEnds(block.substr(start));
    if (!tail.empty()) claims.push_back(tail);
    return claims;
}

std::optional<std::string> oneBalancedObject(const std::string& text) {
    std::optional<std::string> found;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch != '{') continue;
        int depth = 0;
        bool inner_string = false;
        bool inner_escaped = false;
        bool closed = false;
        std::size_t end = i;
        for (std::size_t j = i; j < text.size(); ++j) {
            const char c = text[j];
            if (inner_string) {
                if (inner_escaped) {
                    inner_escaped = false;
                } else if (c == '\\') {
                    inner_escaped = true;
                } else if (c == '"') {
                    inner_string = false;
                }
                continue;
            }
            if (c == '"') {
                inner_string = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth == 0) {
                    closed = true;
                    end = j;
                    break;
                }
            }
        }
        // An unclosed brace is malformed. A second object is also rejected.
        // Ordinary text around the one object is kept out of the parsed value.
        if (!closed || found.has_value()) return std::nullopt;
        found = text.substr(i, end - i + 1);
        i = end;
    }
    return found;
}

nlohmann::json claimJson(int index,
                         const std::string& answer_id,
                         const std::string& claim_text,
                         const std::string& evaluation_status,
                         const std::string& unavailable_reason,
                         const std::string& label,
                         const std::string& evidence_quote,
                         const std::string& evidence_chunk_sha256,
                         const std::string& generation_id) {
    const std::string claim_sha = sha256Hex(claim_text);
    nlohmann::json claim = {
        {"claim_index", index},
        {"claim_id", answer_id + ":" + std::to_string(index) + ":" + claim_sha},
        {"claim_sha256", claim_sha},
        {"claim_text", claim_text},
        {"evaluation_status", evaluation_status},
    };
    if (!generation_id.empty()) claim["generation_id"] = generation_id;
    if (evaluation_status == "available") {
        claim["label"] = label;
        claim["evidence_quote"] = evidence_quote;
        if (!evidence_chunk_sha256.empty()) {
            claim["evidence_chunk_sha256"] = evidence_chunk_sha256;
        }
    } else {
        claim["unavailable_reason"] = unavailable_reason;
    }
    return claim;
}

nlohmann::json unavailableClaim(int index,
                                const std::string& answer_id,
                                const std::string& claim_text,
                                const std::string& reason,
                                const std::string& generation_id) {
    return claimJson(index, answer_id, claim_text, "unavailable", reason, "", "", "", generation_id);
}

} // namespace

std::vector<std::string> segmentCm1EvidenceClaims(const std::string& answer_text) {
    std::vector<std::string> blocks;
    std::string current;
    std::size_t line_begin = 0;
    const auto flush = [&]() {
        if (!current.empty()) blocks.push_back(current);
        current.clear();
    };
    const auto take_line = [&](const std::string& line) {
        if (trimEnds(line).empty()) {
            flush();
            return;
        }
        std::size_t content_begin = 0;
        if (isListMarkerLine(line, content_begin)) {
            flush();
            blocks.push_back(line.substr(content_begin));
            return;
        }
        if (!current.empty()) current.push_back('\n');
        current += line;
    };
    while (line_begin <= answer_text.size()) {
        const std::size_t line_end = answer_text.find('\n', line_begin);
        if (line_end == std::string::npos) {
            take_line(answer_text.substr(line_begin));
            break;
        }
        std::string line = answer_text.substr(line_begin, line_end - line_begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        take_line(line);
        line_begin = line_end + 1;
    }
    flush();

    std::vector<std::string> claims;
    for (const auto& block : blocks) {
        for (const auto& claim : splitClaimSentences(block)) {
            if (!claim.empty()) claims.push_back(claim);
        }
    }
    return claims;
}

std::string buildCm1EvidencePrompt(const std::string& claim,
                                   const std::vector<Cm1EvidenceChunk>& included_chunks) {
    std::ostringstream prompt;
    prompt << "Judge whether the claim is established by the evidence passages.\n";
    prompt << "Use only those passages.\n";
    prompt << "Reply with one JSON object and no other text:\n";
    prompt << "{\"label\":\"supported\",\"evidence_quote\":\"exact span from one passage\"}\n";
    prompt << "{\"label\":\"contradicted\",\"evidence_quote\":\"exact span from one passage\"}\n";
    prompt << "{\"label\":\"insufficient_evidence\",\"evidence_quote\":\"\"}\n";
    prompt << "supported means the passages establish the claim.\n";
    prompt << "contradicted means the passages conflict with the claim.\n";
    prompt << "insufficient_evidence means the passages establish neither conclusion.\n";
    prompt << "Copy the quote exactly from one passage. Leave it empty for insufficient_evidence.\n\n";
    prompt << "EVIDENCE PASSAGE START\n";
    if (included_chunks.empty()) {
        prompt << "No evidence passages were included.\n\n";
    } else {
        for (const auto& chunk : included_chunks) {
            prompt << "[passage " << chunk.content_sha256 << "]\n";
            prompt << chunk.content << "\n\n";
        }
    }
    prompt << "EVIDENCE PASSAGE END\n\n";
    prompt << "CLAIM START\n" << claim << "\n";
    prompt << "CLAIM END\n\n";
    prompt << "Emit exactly one JSON object, stop immediately afterward, and copy any quotation only from the evidence passage, not from the claim.\n";
    return prompt.str();
}

std::optional<Cm1EvidenceJudgmentRecord> runCm1EvidenceJudgment(
    const Cm1AnswerVersionRecord& answer,
    const std::string& answer_text,
    const std::vector<Cm1EvidenceChunk>& included_chunks,
    bool inclusion_known,
    bool evidence_texts_complete,
    bool mock_synthesis,
    const std::function<Cm1EvidenceCompletion(const std::string& prompt, int timeout_seconds)>& caller,
    std::int64_t created_at_ms,
    int budget_seconds,
    const std::function<std::int64_t()>& now_ms) {
    if (answer.answer_id.empty() || answer.content_sha256.empty()) {
        return std::nullopt;
    }
    const std::vector<std::string> claims = segmentCm1EvidenceClaims(answer_text);
    const int judged_count = std::min(static_cast<int>(claims.size()), kCm1EvidenceClaimCap);

    const auto clock = now_ms ? now_ms : []() -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    const std::int64_t budget_ms =
        static_cast<std::int64_t>(std::max(0, budget_seconds)) * 1000;
    const std::int64_t started = clock();

    std::string skip_reason;
    if (mock_synthesis) {
        skip_reason = "mock_synthesis";
    } else if (!inclusion_known) {
        skip_reason = "inclusion_unavailable";
    } else if (!evidence_texts_complete) {
        skip_reason = "evidence_text_missing";
    }

    nlohmann::json claim_rows = nlohmann::json::array();
    for (int index = 0; index < judged_count; ++index) {
        const std::string& claim_text = claims[static_cast<std::size_t>(index)];
        if (!skip_reason.empty()) {
            claim_rows.push_back(unavailableClaim(index, answer.answer_id, claim_text, skip_reason, ""));
            continue;
        }
        const std::int64_t remain_ms = budget_ms - (clock() - started);
        if (remain_ms < 1000) break;
        const int timeout_seconds = static_cast<int>(remain_ms / 1000);
        Cm1EvidenceCompletion completion;
        try {
            completion = caller(buildCm1EvidencePrompt(claim_text, included_chunks), timeout_seconds);
        } catch (...) {
            completion.ok = false;
            completion.error = "exception";
        }
        if (!completion.ok && completion.error == "timeout") {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "timeout", completion.generation_id));
            break;
        }
        if (!completion.ok) {
            const std::string reason = completion.error == "llm_unavailable"
                ? "llm_unavailable"
                : "provider_error";
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, reason, completion.generation_id));
            continue;
        }
        if (completion.text.empty()) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "empty_completion", completion.generation_id));
            continue;
        }
        if (completion.finish_reason.empty()) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "finish_reason_missing",
                completion.generation_id));
            continue;
        }
        if (completion.finish_reason == "length") {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "length", completion.generation_id));
            continue;
        }
        const std::optional<std::string> object = oneBalancedObject(completion.text);
        if (!object.has_value()) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "parse", completion.generation_id));
            continue;
        }
        nlohmann::json parsed;
        try {
            parsed = nlohmann::json::parse(*object);
        } catch (...) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "parse", completion.generation_id));
            continue;
        }
        if (!parsed.is_object() || parsed.size() != 2 || !parsed.contains("label")
            || !parsed["label"].is_string() || !parsed.contains("evidence_quote")
            || !parsed["evidence_quote"].is_string()) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "schema", completion.generation_id));
            continue;
        }
        const std::string label = parsed["label"].get<std::string>();
        const std::string quote = parsed["evidence_quote"].get<std::string>();
        if (label != "supported" && label != "contradicted" && label != "insufficient_evidence") {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "schema", completion.generation_id));
            continue;
        }
        if (label == "insufficient_evidence") {
            if (!quote.empty()) {
                claim_rows.push_back(unavailableClaim(
                    index, answer.answer_id, claim_text, "quote_present", completion.generation_id));
                continue;
            }
            claim_rows.push_back(claimJson(
                index, answer.answer_id, claim_text, "available", "", label, "", "",
                completion.generation_id));
            continue;
        }
        if (quote.empty()) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "quote_required", completion.generation_id));
            continue;
        }
        int hits = 0;
        std::string hit_hash;
        for (const auto& chunk : included_chunks) {
            if (chunk.content.find(quote) != std::string::npos) {
                ++hits;
                hit_hash = chunk.content_sha256;
            }
        }
        if (hits == 0) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "quote_not_in_evidence",
                completion.generation_id));
            continue;
        }
        if (hits != 1) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, "quote_ambiguous", completion.generation_id));
            continue;
        }
        claim_rows.push_back(claimJson(
            index, answer.answer_id, claim_text, "available", "", label, quote, hit_hash,
            completion.generation_id));
    }

    const int claims_total = static_cast<int>(claims.size());
    const int claims_evaluated = static_cast<int>(claim_rows.size());
    const int claims_not_judged = claims_total - claims_evaluated;
    nlohmann::json payload = {
        {"estimator_version", kCm1EvidenceVersion},
        {"answer_id", answer.answer_id},
        {"content_sha256", answer.content_sha256},
        {"claims_total", claims_total},
        {"claims_evaluated", claims_evaluated},
        {"claims_not_judged", claims_not_judged},
        {"claims", claim_rows},
    };
    if (claims.empty()) {
        payload["unavailable_reason"] = "no_claims";
    }

    Cm1EvidenceJudgmentRecord record;
    record.judgment_id = std::string("cm1e1:") + answer.answer_id;
    record.answer_id = answer.answer_id;
    record.content_sha256 = answer.content_sha256;
    record.estimator_version = kCm1EvidenceVersion;
    record.payload_json = payload.dump();
    record.created_at_ms = created_at_ms;
    return record;
}

} // namespace Thoth
