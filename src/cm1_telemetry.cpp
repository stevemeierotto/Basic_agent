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
#include "../include/index_manager.h"
#include "../include/memory_repository.h"

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

bool cm1CorroborationEnabled() {
    const char* env = std::getenv("THOTH_CM1_CORROBORATION");
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

bool sameVerificationChain(const Cm1ClaimVerification& latest,
                           const Cm1StoredClaim& claim,
                           const std::string& protocol_id,
                           const std::string& protocol_version,
                           const std::string& reviewer_id,
                           const std::string& verifier_kind,
                           const std::string& verifier_version,
                           const std::string& method_id) {
    return latest.claim_id == claim.claim_id
        && latest.protocol_id == protocol_id
        && latest.protocol_version == protocol_version
        && latest.reviewer_id == reviewer_id
        && latest.verifier_kind == verifier_kind
        && latest.verifier_version == verifier_version
        && latest.method_id == method_id;
}

bool persistedClaimIdentity(const Cm1AnswerVersionRecord& answer, const Cm1StoredClaim& claim) {
    const std::string prefix = answer.answer_id + ":";
    const std::string suffix = ":" + claim.claim_sha256;
    if (claim.claim_sha256.empty() || claim.answer_id != answer.answer_id) return false;
    if (claim.claim_id.compare(0, prefix.size(), prefix) != 0) return false;
    if (claim.claim_id.size() <= prefix.size() + suffix.size()) return false;
    return claim.claim_id.compare(claim.claim_id.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool internalEvidenceSource(const std::string& source_kind,
                            const std::string& source_ref,
                            const Cm1StoredClaim& claim) {
    if (source_kind == "cm1_judgment" || source_kind == "evidence_passage"
        || source_kind == "evidence_quote" || source_kind == "grag"
        || source_kind == "warm_memory") {
        return true;
    }
    if (!claim.evidence_quote.empty() && source_ref == claim.evidence_quote) return true;
    if (!claim.evidence_chunk_sha256.empty() && source_ref == claim.evidence_chunk_sha256) {
        return true;
    }
    return false;
}

bool knownEvidenceLabel(const std::string& label) {
    return label.empty() || label == "supported" || label == "contradicted"
        || label == "insufficient_evidence";
}

bool evidenceLabelMatchesOutcome(const std::string& label, const std::string& outcome) {
    if (!knownEvidenceLabel(label)) return false;
    if (!label.empty()) return outcome == "unverifiable";
    return outcome == "true" || outcome == "false" || outcome == "unverifiable";
}

} // namespace

std::optional<Cm1StoredClaim> findCm1StoredClaim(const Cm1EvidenceJudgmentRecord& judgment,
                                                 const std::string& claim_id) {
    if (judgment.answer_id.empty() || judgment.content_sha256.empty() || claim_id.empty()
        || judgment.payload_json.empty()) {
        return std::nullopt;
    }
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(judgment.payload_json);
    } catch (...) {
        return std::nullopt;
    }
    if (!payload.is_object() || !payload.contains("claims") || !payload["claims"].is_array()) {
        return std::nullopt;
    }
    for (const auto& row : payload["claims"]) {
        if (!row.is_object() || row.value("claim_id", "") != claim_id) continue;
        if (!row.contains("claim_text") || !row["claim_text"].is_string()) return std::nullopt;
        if (!row.contains("claim_sha256") || !row["claim_sha256"].is_string()) return std::nullopt;
        Cm1StoredClaim claim;
        claim.claim_id = claim_id;
        claim.claim_text = row["claim_text"].get<std::string>();
        claim.claim_sha256 = row["claim_sha256"].get<std::string>();
        claim.answer_id = judgment.answer_id;
        claim.answer_content_sha256 = judgment.content_sha256;
        if (row.contains("evidence_quote") && row["evidence_quote"].is_string()) {
            claim.evidence_quote = row["evidence_quote"].get<std::string>();
        }
        if (row.contains("evidence_chunk_sha256") && row["evidence_chunk_sha256"].is_string()) {
            claim.evidence_chunk_sha256 = row["evidence_chunk_sha256"].get<std::string>();
        }
        return claim;
    }
    return std::nullopt;
}

bool cm1ClaimVerificationConforms(const Cm1ClaimVerification& record,
                                  const Cm1AnswerVersionRecord& answer,
                                  const Cm1StoredClaim& claim) {
    if (record.verification_id.empty() || record.answer_id.empty() || record.claim_id.empty()
        || record.claim_sha256.empty() || record.content_sha256.empty()) {
        return false;
    }
    if (record.answer_id != answer.answer_id || record.content_sha256 != answer.content_sha256
        || answer.content_sha256.empty()) {
        return false;
    }
    if (record.claim_id != claim.claim_id || record.claim_sha256 != claim.claim_sha256) {
        return false;
    }
    if (claim.answer_id != answer.answer_id || claim.answer_content_sha256 != answer.content_sha256) {
        return false;
    }
    if (claim.claim_text.empty() || sha256Hex(claim.claim_text) != claim.claim_sha256) {
        return false;
    }
    if (!persistedClaimIdentity(answer, claim)) return false;
    if (record.protocol_id.empty() || record.protocol_version.empty() || record.reviewer_id.empty()
        || record.method_id.empty() || record.source_kind.empty() || record.source_ref.empty()) {
        return false;
    }
    if (record.verifier_kind != "human" && record.verifier_kind != "automated_agent"
        && record.verifier_kind != "external_service") {
        return false;
    }
    if (record.verifier_kind == "human") {
        if (!record.verifier_version.empty()) return false;
    } else if (record.verifier_version.empty()) {
        return false;
    }
    if (record.outcome != "true" && record.outcome != "false" && record.outcome != "unverifiable") {
        return false;
    }
    if (!evidenceLabelMatchesOutcome(record.evidence_label, record.outcome)) return false;
    if (internalEvidenceSource(record.source_kind, record.source_ref, claim)) return false;
    if (record.calibration_eligibility != "provisional" || record.egar_eligibility != "excluded") {
        return false;
    }
    if (record.revision_reason == "initial") {
        return record.revision_index == 1 && record.supersedes_verification_id.empty();
    }
    if (record.revision_reason == "correction") {
        return record.revision_index >= 2 && !record.supersedes_verification_id.empty();
    }
    return false;
}

Cm1ClaimVerificationDraft prepareCm1ClaimVerification(
    const Cm1AnswerVersionRecord* answer,
    const Cm1StoredClaim* claim,
    const Cm1ClaimVerification* latest,
    const Cm1ClaimVerificationSubmission& submission,
    std::int64_t created_at_ms) {
    Cm1ClaimVerificationDraft draft;
    auto reject = [&](const std::string& message) {
        draft.disposition = Cm1RecordDisposition::Rejected;
        draft.message = message;
        return draft;
    };
    if (!answer || answer->answer_id.empty() || answer->content_sha256.empty()) {
        return reject("No answer version is recorded for this synthesis.");
    }
    if (!claim || claim->claim_id.empty() || claim->claim_sha256.empty() || claim->claim_text.empty()) {
        return reject("No persisted claim is recorded for this verification.");
    }
    if (submission.answer_id != answer->answer_id || claim->answer_id != answer->answer_id) {
        return reject("Verification does not match the answer version.");
    }
    if (submission.claim_id != claim->claim_id) {
        return reject("Verification does not match the persisted claim.");
    }
    if (claim->answer_content_sha256 != answer->content_sha256) {
        return reject("The persisted claim is not for this answer version.");
    }
    if (sha256Hex(submission.answer_text) != answer->content_sha256) {
        return reject("The displayed answer does not match the stored version.");
    }
    if (sha256Hex(claim->claim_text) != claim->claim_sha256) {
        return reject("The persisted claim text does not match its stored hash.");
    }
    if (sha256Hex(submission.claim_text) != claim->claim_sha256) {
        return reject("The claim text does not match the persisted claim.");
    }
    if (!persistedClaimIdentity(*answer, *claim)) {
        return reject("The persisted claim id does not match the answer and claim hash.");
    }

    const std::string protocol_id = trimCopy(submission.protocol_id);
    const std::string protocol_version = trimCopy(submission.protocol_version);
    const std::string reviewer_id = trimCopy(submission.reviewer_id);
    const std::string method_id = trimCopy(submission.method_id);
    const std::string verifier_version = trimCopy(submission.verifier_version);
    const std::string source_kind = trimCopy(submission.source_kind);
    const std::string source_ref = trimCopy(submission.source_ref);
    if (protocol_id.empty() || protocol_version.empty() || reviewer_id.empty() || method_id.empty()
        || source_kind.empty() || source_ref.empty()) {
        return reject("A claim verification needs a protocol, version, reviewer, method, and external source.");
    }

    const std::string& kind = submission.verifier_kind;
    if (kind != "human" && kind != "automated_agent" && kind != "external_service") {
        return reject("Verifier kind must be human, automated_agent, or external_service.");
    }
    if (kind == "human") {
        if (!verifier_version.empty()) {
            return reject("A human verifier has no verifier version.");
        }
    } else if (verifier_version.empty()) {
        return reject("An automated agent or external service needs a verifier version.");
    }

    if (submission.outcome != "true" && submission.outcome != "false"
        && submission.outcome != "unverifiable") {
        return reject("Claim outcome must be true, false, or unverifiable.");
    }
    const std::string evidence_label = trimCopy(submission.evidence_label);
    if (!knownEvidenceLabel(evidence_label)) {
        return reject("Evidence label must be empty, supported, contradicted, or insufficient_evidence.");
    }
    if (!evidence_label.empty() && submission.outcome != "unverifiable") {
        return reject("An evidence label requires an unverifiable outcome.");
    }
    if ((submission.outcome == "true" || submission.outcome == "false") && !evidence_label.empty()) {
        return reject("A true or false outcome requires an empty evidence label.");
    }
    if (internalEvidenceSource(source_kind, source_ref, *claim)) {
        return reject("Verification cannot cite the claim's internal evidence.");
    }
    if (submission.calibration_eligibility != "provisional"
        || submission.egar_eligibility != "excluded") {
        return reject("This checkpoint accepts only provisional calibration eligibility and excluded EGAR eligibility.");
    }
    if (submission.revision_reason == "authorization") {
        return reject("Authorization revisions are not accepted in this checkpoint.");
    }

    const bool chain_match = latest && sameVerificationChain(
        *latest, *claim, protocol_id, protocol_version, reviewer_id, kind, verifier_version, method_id);
    if (latest && !chain_match) {
        return reject("The latest verification is for a different chain.");
    }

    int revision_index = 1;
    std::string supersedes;
    if (submission.revision_reason == "initial") {
        if (latest) {
            return reject("An initial verification requires an empty chain.");
        }
    } else if (submission.revision_reason == "correction") {
        if (!latest) {
            return reject("A correction requires an existing verification.");
        }
        revision_index = latest->revision_index + 1;
        supersedes = latest->verification_id;
    } else {
        return reject("Revision reason must be initial, correction, or authorization.");
    }

    draft.record.claim_id = claim->claim_id;
    draft.record.claim_sha256 = claim->claim_sha256;
    draft.record.answer_id = answer->answer_id;
    draft.record.content_sha256 = answer->content_sha256;
    draft.record.protocol_id = protocol_id;
    draft.record.protocol_version = protocol_version;
    draft.record.reviewer_id = reviewer_id;
    draft.record.verifier_kind = kind;
    draft.record.verifier_version = verifier_version;
    draft.record.method_id = method_id;
    draft.record.outcome = submission.outcome;
    draft.record.evidence_label = evidence_label;
    draft.record.source_kind = source_kind;
    draft.record.source_ref = source_ref;
    draft.record.calibration_eligibility = "provisional";
    draft.record.egar_eligibility = "excluded";
    draft.record.revision_reason = submission.revision_reason;
    draft.record.supersedes_verification_id = supersedes;
    draft.record.revision_index = revision_index;
    draft.record.verification_id = claim->claim_id + ":c:" + protocol_id + ":" + protocol_version + ":"
        + kind + ":" + verifier_version + ":" + method_id + ":" + reviewer_id + ":"
        + std::to_string(revision_index);
    draft.record.created_at_ms = created_at_ms;
    if (!cm1ClaimVerificationConforms(draft.record, *answer, *claim)) {
        return reject("The verification record does not match the persisted claim.");
    }
    draft.disposition = Cm1RecordDisposition::Insert;
    draft.message = "Claim verification recorded.";
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

struct Cm1EvidenceAssessment {
    bool available = false;
    bool stop = false;
    std::string label;
    std::string quote;
    std::string chunk_sha256;
    std::string unavailable_reason;
};

Cm1EvidenceAssessment assessCm1EvidenceCompletion(
    const Cm1EvidenceCompletion& completion,
    const std::vector<Cm1EvidenceChunk>& included_chunks) {
    Cm1EvidenceAssessment assessment;
    if (!completion.ok && completion.error == "timeout") {
        assessment.unavailable_reason = "timeout";
        assessment.stop = true;
        return assessment;
    }
    if (!completion.ok) {
        assessment.unavailable_reason = completion.error == "llm_unavailable"
            ? "llm_unavailable"
            : "provider_error";
        return assessment;
    }
    if (completion.text.empty()) {
        assessment.unavailable_reason = "empty_completion";
        return assessment;
    }
    if (completion.finish_reason.empty()) {
        assessment.unavailable_reason = "finish_reason_missing";
        return assessment;
    }
    if (completion.finish_reason == "length") {
        assessment.unavailable_reason = "length";
        return assessment;
    }
    const std::optional<std::string> object = oneBalancedObject(completion.text);
    if (!object.has_value()) {
        assessment.unavailable_reason = "parse";
        return assessment;
    }
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(*object);
    } catch (...) {
        assessment.unavailable_reason = "parse";
        return assessment;
    }
    if (!parsed.is_object() || parsed.size() != 2 || !parsed.contains("label")
        || !parsed["label"].is_string() || !parsed.contains("evidence_quote")
        || !parsed["evidence_quote"].is_string()) {
        assessment.unavailable_reason = "schema";
        return assessment;
    }
    const std::string label = parsed["label"].get<std::string>();
    const std::string quote = parsed["evidence_quote"].get<std::string>();
    if (label != "supported" && label != "contradicted" && label != "insufficient_evidence") {
        assessment.unavailable_reason = "schema";
        return assessment;
    }
    if (label == "insufficient_evidence") {
        if (!quote.empty()) {
            assessment.unavailable_reason = "quote_present";
            return assessment;
        }
        assessment.available = true;
        assessment.label = label;
        return assessment;
    }
    if (quote.empty()) {
        assessment.unavailable_reason = "quote_required";
        return assessment;
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
        assessment.unavailable_reason = "quote_not_in_evidence";
        return assessment;
    }
    if (hits != 1) {
        assessment.unavailable_reason = "quote_ambiguous";
        return assessment;
    }
    assessment.available = true;
    assessment.label = label;
    assessment.quote = quote;
    assessment.chunk_sha256 = hit_hash;
    return assessment;
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
        const Cm1EvidenceAssessment assessment = assessCm1EvidenceCompletion(
            completion, included_chunks);
        if (!assessment.available) {
            claim_rows.push_back(unavailableClaim(
                index, answer.answer_id, claim_text, assessment.unavailable_reason,
                completion.generation_id));
            if (assessment.stop) break;
            continue;
        }
        claim_rows.push_back(claimJson(
            index, answer.answer_id, claim_text, "available", "", assessment.label,
            assessment.quote, assessment.chunk_sha256, completion.generation_id));
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

namespace {

constexpr const char* kWarmMemoryPrefix = "warm_memory:";

bool warmMemoryFile(const std::string& file_name) {
    const std::string prefix = kWarmMemoryPrefix;
    return file_name.compare(0, prefix.size(), prefix) == 0;
}

std::string heldOutBatchRef(const std::vector<std::string>& content_hashes) {
    std::vector<std::string> ordered = content_hashes;
    std::sort(ordered.begin(), ordered.end());
    ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
    std::string joined;
    for (const auto& hash : ordered) {
        if (!joined.empty()) joined.push_back('\n');
        joined += hash;
    }
    return joined;
}

Cm1ClaimCorroborationResult corroborationIdle(const std::string& reason) {
    Cm1ClaimCorroborationResult result;
    result.reason = reason;
    return result;
}

void appendPersistedRow(const nlohmann::json& item,
                        std::vector<std::pair<std::string, std::string>>& rows) {
    if (!item.is_object()) return;
    std::string file = item.value("file", "");
    if (file.empty()) file = item.value("file_name", "");
    std::string content = item.value("content", "");
    if (content.empty()) content = item.value("code_text", "");
    if (content.empty()) return;
    rows.emplace_back(std::move(file), std::move(content));
}

void appendPersistedArray(const nlohmann::json& parent,
                          const char* key,
                          std::vector<std::pair<std::string, std::string>>& rows) {
    if (!parent.is_object() || !parent.contains(key) || !parent[key].is_array()) return;
    for (const auto& item : parent[key]) appendPersistedRow(item, rows);
}

std::vector<std::pair<std::string, std::string>> persistedRetrievalRows(const std::string& body) {
    std::vector<std::pair<std::string, std::string>> rows;
    const auto parsed = nlohmann::json::parse(body);
    if (!parsed.is_object() || !parsed.contains("steps") || !parsed["steps"].is_array()) return rows;
    for (const auto& step : parsed["steps"]) {
        if (!step.is_object() || !step.contains("result") || !step["result"].is_object()) continue;
        const auto& result = step["result"];
        if (result.contains("data") && result["data"].is_object()) {
            appendPersistedArray(result["data"], "chunks", rows);
            if (result["data"].contains("data") && result["data"]["data"].is_object()) {
                appendPersistedArray(result["data"]["data"], "chunks", rows);
            }
        }
        if (result.contains("diagnostics") && result["diagnostics"].is_object()) {
            appendPersistedArray(result["diagnostics"], "breakdowns", rows);
        }
    }
    return rows;
}

bool sourceFileForHash(const std::vector<std::pair<std::string, std::string>>& rows,
                       const std::string& hash,
                       std::unordered_set<std::string>& files) {
    bool found = false;
    for (const auto& row : rows) {
        if (sha256Hex(row.second) != hash) continue;
        if (row.first.empty()) return false;
        files.insert(row.first);
        found = true;
    }
    return found;
}

} // namespace

Cm1ClaimCorroborationResult runCm1ClaimCorroboration(
    MemoryRepository& repository,
    const std::string& answer_id,
    const std::string& claim_id,
    const std::string& answer_text,
    const std::vector<Cm1CorroborationPassage>& passages,
    const std::function<Cm1EvidenceCompletion(const std::string& prompt, int timeout_seconds)>& caller,
    std::int64_t created_at_ms,
    int budget_seconds,
    const std::function<std::int64_t()>& now_ms) {
    if (!cm1CorroborationEnabled()) return corroborationIdle("disabled");

    const auto answer = repository.loadCm1AnswerVersion(answer_id);
    if (!answer) return corroborationIdle("missing_answer");
    if (sha256Hex(answer_text) != answer->content_sha256) return corroborationIdle("answer_mismatch");

    const auto judgment = repository.loadCm1EvidenceJudgment(std::string("cm1e1:") + answer_id);
    if (!judgment) return corroborationIdle("missing_claim");
    const auto claim = findCm1StoredClaim(*judgment, claim_id);
    if (!claim) return corroborationIdle("missing_claim");

    const auto inclusion = repository.loadCm1SynthesisInclusion(answer->plan_id, answer->step_id);
    if (!inclusion) return corroborationIdle("inclusion_unknown");
    std::unordered_set<std::string> included(
        inclusion->content_sha256.begin(), inclusion->content_sha256.end());

    std::vector<Cm1EvidenceChunk> eligible;
    std::unordered_set<std::string> seen_hash;
    for (const auto& passage : passages) {
        if (passage.content.empty() || warmMemoryFile(passage.file_name)) continue;
        if (!claim->evidence_quote.empty() && passage.content == claim->evidence_quote) continue;
        const std::string hash = sha256Hex(passage.content);
        if (!claim->evidence_chunk_sha256.empty() && hash == claim->evidence_chunk_sha256) continue;
        if (included.count(hash) || seen_hash.count(hash)) continue;
        seen_hash.insert(hash);
        Cm1EvidenceChunk chunk;
        chunk.content = passage.content;
        chunk.content_sha256 = hash;
        eligible.push_back(std::move(chunk));
    }
    if (eligible.empty()) return corroborationIdle("no_eligible_passage");

    const auto clock = now_ms ? now_ms : []() -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    const std::int64_t budget_ms = static_cast<std::int64_t>(std::max(0, budget_seconds)) * 1000;
    if (budget_ms < 1000) return corroborationIdle("timeout");

    Cm1EvidenceCompletion completion;
    try {
        completion = caller(buildCm1EvidencePrompt(claim->claim_text, eligible),
                            static_cast<int>(budget_ms / 1000));
    } catch (...) {
        completion.ok = false;
        completion.error = "exception";
    }
    const Cm1EvidenceAssessment assessment = assessCm1EvidenceCompletion(completion, eligible);
    if (!assessment.available) {
        if (assessment.unavailable_reason == "timeout") return corroborationIdle("timeout");
        if (assessment.unavailable_reason == "quote_ambiguous") return corroborationIdle("ambiguous");
        return corroborationIdle("invalid_output");
    }
    if ((assessment.label == "supported" || assessment.label == "contradicted")
        && (assessment.chunk_sha256.empty() || included.count(assessment.chunk_sha256)
            || assessment.chunk_sha256 == claim->evidence_chunk_sha256)) {
        return corroborationIdle("invalid_output");
    }

    const bool batch = eligible.size() > 1 && assessment.label == "insufficient_evidence";
    std::vector<std::string> member_hashes;
    member_hashes.reserve(eligible.size());
    for (const auto& chunk : eligible) member_hashes.push_back(chunk.content_sha256);

    Cm1ClaimVerificationSubmission submission;
    submission.claim_id = claim->claim_id;
    submission.claim_text = claim->claim_text;
    submission.answer_id = answer->answer_id;
    submission.answer_text = answer_text;
    submission.protocol_id = kCm1CorroborationProtocolId;
    submission.protocol_version = kCm1CorroborationProtocolVersion;
    submission.reviewer_id = kCm1CorroborationReviewerId;
    submission.verifier_kind = "automated_agent";
    submission.verifier_version = kCm1CorroborationVerifierVersion;
    submission.method_id = kCm1CorroborationMethodId;
    submission.outcome = "unverifiable";
    submission.evidence_label = assessment.label;
    submission.source_kind = batch ? kCm1HeldOutBatchSource : kCm1HeldOutPassageSource;
    submission.source_ref = batch ? heldOutBatchRef(member_hashes) : (
        assessment.label == "insufficient_evidence" ? eligible.front().content_sha256
                                                    : assessment.chunk_sha256);
    submission.calibration_eligibility = "provisional";
    submission.egar_eligibility = "excluded";

    const auto latest = repository.latestCm1ClaimVerification(
        claim->claim_id, submission.protocol_id, submission.protocol_version, submission.reviewer_id,
        submission.verifier_kind, submission.verifier_version, submission.method_id);
    if (latest && latest->evidence_label == submission.evidence_label
        && latest->outcome == submission.outcome
        && latest->source_kind == submission.source_kind
        && latest->source_ref == submission.source_ref) {
        Cm1ClaimCorroborationResult result;
        result.reason = "unchanged";
        result.record = latest;
        return result;
    }
    submission.revision_reason = latest ? "correction" : "initial";
    const Cm1ClaimVerificationDraft draft = prepareCm1ClaimVerification(
        &*answer, &*claim, latest ? &*latest : nullptr, submission, created_at_ms);
    if (draft.disposition != Cm1RecordDisposition::Insert) return corroborationIdle("rejected");
    if (!repository.insertCm1ClaimVerification(draft.record)) return corroborationIdle("rejected");
    Cm1ClaimCorroborationResult result;
    result.wrote = true;
    result.reason = "stored";
    result.record = draft.record;
    return result;
}

Cm1ClaimCorroborationResult runCm1ClaimDiscovery(
    IndexManager& index,
    MemoryRepository& repository,
    const std::string& answer_id,
    const std::string& claim_id,
    const std::string& answer_text,
    const std::function<Cm1EvidenceCompletion(const std::string& prompt, int timeout_seconds)>& caller,
    std::int64_t created_at_ms,
    int budget_seconds,
    int top_k,
    const std::function<std::int64_t()>& now_ms) {
    if (index.isIndexing()) return corroborationIdle("indexing");

    const auto answer = repository.loadCm1AnswerVersion(answer_id);
    if (!answer) return corroborationIdle("missing_answer");
    if (sha256Hex(answer_text) != answer->content_sha256) return corroborationIdle("answer_mismatch");

    const auto judgment = repository.loadCm1EvidenceJudgment(std::string("cm1e1:") + answer_id);
    if (!judgment) return corroborationIdle("missing_claim");
    const auto claim = findCm1StoredClaim(*judgment, claim_id);
    if (!claim) return corroborationIdle("missing_claim");

    const auto inclusion = repository.loadCm1SynthesisInclusion(answer->plan_id, answer->step_id);
    if (!inclusion) return corroborationIdle("inclusion_unknown");
    const auto plan_body = repository.loadCm1PersistedPlanBody(answer->plan_id, answer->step_id);
    if (!plan_body) return corroborationIdle("provenance_missing");

    std::vector<std::pair<std::string, std::string>> persisted;
    try {
        persisted = persistedRetrievalRows(*plan_body);
    } catch (...) {
        return corroborationIdle("provenance_missing");
    }

    std::unordered_set<std::string> synthesis_files;
    for (const auto& hash : inclusion->content_sha256) {
        if (hash.empty() || !sourceFileForHash(persisted, hash, synthesis_files)) {
            return corroborationIdle("provenance_missing");
        }
    }
    if (!claim->evidence_chunk_sha256.empty()
        && !sourceFileForHash(persisted, claim->evidence_chunk_sha256, synthesis_files)) {
        return corroborationIdle("provenance_missing");
    }

    std::unordered_set<std::string> included(
        inclusion->content_sha256.begin(), inclusion->content_sha256.end());
    // retrieveChunks returns an empty list when the query embedding fails.
    const int recall = top_k > 0 ? top_k : 8;
    const auto hits = index.retrieveChunks(claim->claim_text, recall);
    std::vector<Cm1CorroborationPassage> passages;
    std::unordered_set<std::string> seen_hash;
    for (const auto& hit : hits) {
        const std::string content = hit.first;
        const auto chunk = index.copyChunkByCode(content);
        if (!chunk) continue;
        const std::string file_name = chunk->fileName;
        if (content.empty() || warmMemoryFile(file_name) || synthesis_files.count(file_name)) continue;
        if (!claim->evidence_quote.empty() && content == claim->evidence_quote) continue;
        const std::string hash = sha256Hex(content);
        if (!claim->evidence_chunk_sha256.empty() && hash == claim->evidence_chunk_sha256) continue;
        if (included.count(hash) || seen_hash.count(hash)) continue;
        seen_hash.insert(hash);
        passages.push_back(Cm1CorroborationPassage{content, file_name});
    }

    return runCm1ClaimCorroboration(
        repository, answer_id, claim_id, answer_text, passages, caller,
        created_at_ms, budget_seconds, now_ms);
}

} // namespace Thoth
