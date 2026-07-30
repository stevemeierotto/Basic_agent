/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-D0 migration dry-run analyzer (read-only)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "alp_migration_analyzer.h"

#include "alp_index_reader.h"
#include "alp_sha256.h"
#include "corpus_documents.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace Thoth {
namespace {

constexpr const char* kPreviewUuidNamespace = "thoth-alp-d0-preview-v1";

struct CandidateRow {
    std::string path;
    std::string filename;
    std::string content_sha256;
    std::int64_t mtime_sec = 0;
    int chunk_count = 0;
    std::vector<std::string> registry_sessions;
    std::string stable_document_id_preview;
    std::string tier = "operator_candidate";
    bool file_exists = false;
    bool empty_content = false;
    bool valid_for_winner = false;
    bool in_registry = false;
    bool in_session_paths = false;
};

struct GroupRow {
    std::string canonical_stem;
    std::vector<CandidateRow> candidates;
    bool hash_merge = false;
};

nlohmann::json sortJsonRecursive(const nlohmann::json& node) {
    if (node.is_object()) {
        nlohmann::json out = nlohmann::json::object();
        std::vector<std::string> keys;
        keys.reserve(node.size());
        for (auto it = node.begin(); it != node.end(); ++it) {
            keys.push_back(it.key());
        }
        std::sort(keys.begin(), keys.end());
        for (const auto& key : keys) {
            out[key] = sortJsonRecursive(node[key]);
        }
        return out;
    }
    if (node.is_array()) {
        nlohmann::json out = nlohmann::json::array();
        for (const auto& item : node) {
            out.push_back(sortJsonRecursive(item));
        }
        return out;
    }
    return node;
}

bool isWhitespaceOnlyFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return true;
    }
    char ch = 0;
    while (in.get(ch)) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return true;
}

bool isDigits(const std::string& s) {
    return !s.empty()
        && std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
}

std::string canonicalStemFromFilename(const std::string& filename) {
    const fs::path p(filename);
    std::string stem = p.stem().string();
    const auto pos = stem.rfind('_');
    if (pos != std::string::npos && pos + 1 < stem.size()) {
        const std::string suffix = stem.substr(pos + 1);
        if (isDigits(suffix)) {
            return stem.substr(0, pos);
        }
    }
    return stem;
}

std::string proposedCanonicalName(const std::string& canonical_stem, const std::string& winner_filename) {
    const fs::path winner(winner_filename);
    const std::string ext = winner.extension().string();
    if (ext.empty()) {
        return canonical_stem;
    }
    return canonical_stem + ext;
}

bool isExcludedRagSubpath(const fs::path& relative) {
    if (relative.empty()) {
        return false;
    }
    const auto first = relative.begin()->string();
    return first == "seed" || first == "attachments" || first == "revisions"
        || first == "migration_archive";
}

bool isSeedFilenameHeuristic(const std::string& filename) {
    const std::string lower = [&]() {
        std::string out = filename;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }();
    return lower.find("grag") != std::string::npos || lower.find("benchmark") != std::string::npos
        || lower.find("system_reference") != std::string::npos;
}

std::string classifyTier(const fs::path& relative, bool in_registry, bool in_session,
                         const std::string& filename) {
    if (!relative.empty() && relative.begin()->string() == "seed") {
        return "seed_definite";
    }
    const bool seed_hint = isSeedFilenameHeuristic(filename);
    const bool operator_hint = in_registry || in_session;
    if (seed_hint && operator_hint) {
        return "ambiguous";
    }
    if (seed_hint) {
        return "seed_candidate";
    }
    if (operator_hint) {
        return "operator_definite";
    }
    return "operator_candidate";
}

std::string previewDocumentId(const std::string& canonical_stem, const std::string& winner_hash) {
    const std::string seed = std::string(kPreviewUuidNamespace) + ":" + canonical_stem + ":" + winner_hash;
    const std::string hex = sha256Hex(seed);
    if (hex.size() < 32) {
        return "00000000-0000-4000-8000-000000000000";
    }
    return hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-4" + hex.substr(13, 3) + "-a"
        + hex.substr(17, 3) + "-" + hex.substr(20, 12);
}

std::string iso8601NowUtc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream out;
    out << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string normalizePath(const fs::path& p) {
    try {
        return fs::absolute(p).lexically_normal().string();
    } catch (...) {
        return p.string();
    }
}

int compareWinner(const CandidateRow& a, const CandidateRow& b) {
    const bool a_committed = a.chunk_count > 0;
    const bool b_committed = b.chunk_count > 0;
    if (a_committed != b_committed) {
        return a_committed ? -1 : 1;
    }
    if (a_committed && b_committed && a.chunk_count != b.chunk_count) {
        return a.chunk_count > b.chunk_count ? -1 : 1;
    }
    if (a.mtime_sec != b.mtime_sec) {
        return a.mtime_sec > b.mtime_sec ? -1 : 1;
    }
    if (a.chunk_count != b.chunk_count) {
        return a.chunk_count > b.chunk_count ? -1 : 1;
    }
    if (a.path != b.path) {
        return a.path < b.path ? -1 : 1;
    }
    return 0;
}

void updateValidity(CandidateRow& row) {
    row.empty_content = row.file_exists && (row.content_sha256.empty() || isWhitespaceOnlyFile(row.path));
    const bool seed_block = row.tier == "seed_definite" || row.tier == "ambiguous";
    row.valid_for_winner =
        row.file_exists && !row.empty_content && !row.content_sha256.empty() && !seed_block;
}

nlohmann::json candidateJson(const CandidateRow& c) {
    return nlohmann::json{
        {"path", c.path},
        {"content_sha256", c.content_sha256},
        {"mtime", c.mtime_sec},
        {"chunk_count", c.chunk_count},
        {"registry_sessions", c.registry_sessions},
        {"stable_document_id_preview", c.stable_document_id_preview},
        {"tier_classification", c.tier},
        {"valid_for_winner", c.valid_for_winner},
        {"file_exists", c.file_exists},
    };
}

} // namespace

AlpMigrationAnalyzer::AlpMigrationAnalyzer(std::string workspace_root)
    : workspace_root_(std::move(workspace_root)) {
    rag_root_ = normalizePath(fs::path(workspace_root_) / "rag");
    index_path_ = normalizePath(fs::path(workspace_root_) / "rag" / "rag_index.bin");
    registry_path_ = normalizePath(fs::path(workspace_root_) / "rag_attachment_registry.json");
    sessions_path_ = normalizePath(fs::path(workspace_root_) / "chat_sessions.json");
}

nlohmann::json AlpMigrationAnalyzer::canonicalPayload(const nlohmann::json& report) {
    nlohmann::json payload = report;
    payload.erase("generated_at");
    payload.erase("migration_run_id");
    payload.erase("report_hash");
    return sortJsonRecursive(payload);
}

std::string AlpMigrationAnalyzer::computeReportHash(const nlohmann::json& report) {
    return sha256Hex(canonicalPayload(report).dump());
}

nlohmann::json AlpMigrationAnalyzer::runDryRun() {
    std::map<std::string, std::string> registry_owners;
    {
        std::ifstream in(registry_path_);
        if (in.is_open()) {
            try {
                nlohmann::json body;
                in >> body;
                if (body.contains("owners") && body["owners"].is_object()) {
                    for (auto& item : body["owners"].items()) {
                        if (item.value().is_string()) {
                            registry_owners[normalizePath(fs::path(item.key()))] =
                                item.value().get<std::string>();
                        }
                    }
                }
            } catch (...) {
            }
        }
    }

    std::set<std::string> session_rag_paths;
    {
        std::ifstream in(sessions_path_);
        if (in.is_open()) {
            try {
                nlohmann::json body;
                in >> body;
                const auto scan_paths = [&](const nlohmann::json& node) {
                    if (node.is_array()) {
                        for (const auto& p : node) {
                            if (p.is_string()) {
                                session_rag_paths.insert(normalizePath(fs::path(p.get<std::string>())));
                            }
                        }
                    }
                };
                if (body.is_object()) {
                    for (auto& item : body.items()) {
                        if (!item.value().is_object()) {
                            continue;
                        }
                        if (item.value().contains("ragFilePaths")) {
                            scan_paths(item.value()["ragFilePaths"]);
                        }
                    }
                }
            } catch (...) {
            }
        }
    }

    const auto chunk_counts = AlpIndexReader::loadChunkCountsByPath(index_path_);

    std::map<std::string, GroupRow> groups;
    std::vector<nlohmann::json> warnings;

    auto ensureCandidate = [&](const std::string& abs_path, bool registry_only) {
        const fs::path path(abs_path);
        const std::string filename = path.filename().string();
        fs::path relative;
        try {
            relative = fs::relative(path, rag_root_);
        } catch (...) {
            relative = filename;
        }
        if (!registry_only && isExcludedRagSubpath(relative)) {
            return;
        }
        if (filename == "rag_index.bin") {
            return;
        }

        const std::string stem = canonicalStemFromFilename(filename);
        GroupRow& group = groups[stem];
        group.canonical_stem = stem;

        CandidateRow row;
        row.path = abs_path;
        row.filename = filename;
        row.file_exists = fs::is_regular_file(path);
        row.in_registry = registry_owners.count(abs_path) > 0;
        row.in_session_paths = session_rag_paths.count(abs_path) > 0;
        row.tier = classifyTier(relative, row.in_registry, row.in_session_paths, filename);
        if (row.in_registry) {
            row.registry_sessions.push_back(registry_owners.at(abs_path));
        }
        if (row.file_exists) {
            row.content_sha256 = sha256HexFile(abs_path);
            std::error_code ec;
            const auto ftime = fs::last_write_time(abs_path, ec);
            if (!ec) {
                row.mtime_sec = static_cast<std::int64_t>(
                    std::chrono::duration_cast<std::chrono::seconds>(ftime.time_since_epoch()).count());
            }
        }
        auto count_it = chunk_counts.find(abs_path);
        if (count_it != chunk_counts.end()) {
            row.chunk_count = count_it->second;
        }
        try {
            const std::string storage_key = relative.lexically_normal().string();
            row.stable_document_id_preview = CorpusDocuments::stableDocumentId(storage_key);
        } catch (...) {
            row.stable_document_id_preview = CorpusDocuments::stableDocumentId(filename);
        }
        updateValidity(row);
        group.candidates.push_back(std::move(row));
    };

    if (fs::is_directory(rag_root_)) {
        for (auto it = fs::recursive_directory_iterator(
                 rag_root_, fs::directory_options::skip_permission_denied);
             it != fs::recursive_directory_iterator();
             ++it) {
            if (!it->is_regular_file()) {
                continue;
            }
            ensureCandidate(normalizePath(it->path()), false);
        }
    }

    for (const auto& [reg_path, session_id] : registry_owners) {
        (void)session_id;
        bool found = false;
        for (const auto& [stem, group] : groups) {
            (void)stem;
            for (const auto& c : group.candidates) {
                if (c.path == reg_path) {
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
        if (!found) {
            ensureCandidate(reg_path, true);
            warnings.push_back({{"code", "missing_storage"},
                                {"path", reg_path},
                                {"detail", "registry row without readable file"}});
        }
    }

    nlohmann::json groups_json = nlohmann::json::array();
    nlohmann::json chunk_retag = nlohmann::json::array();
    nlohmann::json legacy_map = nlohmann::json::object();
    int archive_files = 0;
    int hash_merge_groups = 0;
    int suffix_groups = 0;

    for (auto& [stem, group] : groups) {
        std::sort(group.candidates.begin(), group.candidates.end(),
                  [](const CandidateRow& a, const CandidateRow& b) { return a.path < b.path; });

        if (group.candidates.size() > 1) {
            suffix_groups++;
        }

        std::set<std::string> unique_hashes;
        for (const auto& c : group.candidates) {
            if (!c.content_sha256.empty()) {
                unique_hashes.insert(c.content_sha256);
            }
        }
        group.hash_merge = unique_hashes.size() <= 1 && unique_hashes.size() > 0;

        std::vector<const CandidateRow*> valid_ptrs;
        for (const auto& c : group.candidates) {
            if (c.valid_for_winner) {
                valid_ptrs.push_back(&c);
            }
            if (c.in_registry && c.chunk_count == 0 && c.file_exists && !c.empty_content) {
                warnings.push_back({{"code", "registry_index_mismatch"},
                                    {"path", c.path},
                                    {"detail", "registry bound but index chunk_count=0"}});
            }
        }

        if (valid_ptrs.empty()) {
            warnings.push_back({{"code", "no_valid_winner"},
                                {"path", stem},
                                {"detail", "no valid operator revision candidate in group"}});
            nlohmann::json group_json = nlohmann::json::object();
            group_json["canonical_stem"] = stem;
            group_json["hash_merge"] = group.hash_merge;
            group_json["candidates"] = nlohmann::json::array();
            for (const auto& c : group.candidates) {
                group_json["candidates"].push_back(candidateJson(c));
            }
            group_json["no_valid_winner"] = true;
            for (const auto& c : group.candidates) {
                if (c.tier == "ambiguous") {
                    warnings.push_back({{"code", "seed_operator_ambiguous"},
                                        {"path", c.path},
                                        {"detail", "requires_operator_review"}});
                }
            }
            groups_json.push_back(group_json);
            continue;
        }

        const CandidateRow* winner = valid_ptrs.front();
        for (const CandidateRow* ptr : valid_ptrs) {
            if (compareWinner(*ptr, *winner) < 0) {
                winner = ptr;
            }
        }

        if (group.hash_merge) {
            hash_merge_groups++;
        }

        bool all_zero = true;
        for (const auto& c : group.candidates) {
            if (c.chunk_count > 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            warnings.push_back({{"code", "degraded_index_all_zero"},
                                {"path", winner->path},
                                {"detail", "all candidates have chunk_count=0"}});
        }

        if (winner->tier == "ambiguous") {
            warnings.push_back({{"code", "seed_operator_ambiguous"},
                                {"path", winner->path},
                                {"detail", "requires_operator_review"}});
        }
        for (const auto& c : group.candidates) {
            if (c.tier == "ambiguous" && c.path != winner->path) {
                warnings.push_back({{"code", "seed_operator_ambiguous"},
                                    {"path", c.path},
                                    {"detail", "requires_operator_review"}});
            }
        }

        const std::string preview_id = previewDocumentId(stem, winner->content_sha256);
        const std::string canonical_name = proposedCanonicalName(stem, winner->filename);

        nlohmann::json losers = nlohmann::json::array();
        for (const auto& c : group.candidates) {
            if (c.path == winner->path) {
                continue;
            }
            archive_files++;
            losers.push_back({{"path", c.path},
                              {"archive_path_preview",
                               "rag/migration_archive/{migration_run_id}/" + c.filename}});
        }

        std::set<std::string> session_links;
        for (const auto& c : group.candidates) {
            for (const auto& s : c.registry_sessions) {
                session_links.insert(s);
            }
        }

        std::string winner_reason = "lexicographic_path";
        if (winner->chunk_count > 0) {
            winner_reason = "committed_index_preferred";
        } else if (winner->mtime_sec > 0) {
            winner_reason = "newest_mtime";
        }

        nlohmann::json group_json = nlohmann::json::object();
        group_json["canonical_stem"] = stem;
        group_json["hash_merge"] = group.hash_merge;
        group_json["candidates"] = nlohmann::json::array();
        group_json["winner"] = {
            {"path", winner->path},
            {"reason", winner_reason},
            {"proposed_document_id_preview", preview_id},
            {"preview_only", true},
            {"proposed_canonical_name", canonical_name},
            {"target_path_preview", "rag/attachments/" + canonical_name},
        };
        group_json["losers"] = losers;
        group_json["session_links_preview"] = nlohmann::json::array();
        for (const auto& c : group.candidates) {
            group_json["candidates"].push_back(candidateJson(c));
        }
        for (const auto& s : session_links) {
            group_json["session_links_preview"].push_back(s);
        }
        groups_json.push_back(group_json);

        chunk_retag.push_back({{"legacy_fileName", winner->path},
                               {"proposed_document_id_preview", preview_id},
                               {"preview_only", true},
                               {"chunk_count", winner->chunk_count},
                               {"action", "retag"}});
        legacy_map[winner->stable_document_id_preview] = preview_id;
    }

    std::sort(warnings.begin(), warnings.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
        const auto key = [](const nlohmann::json& w) {
            return w.value("code", "") + "|" + w.value("path", "") + "|" + w.value("detail", "");
        };
        return key(a) < key(b);
    });

    nlohmann::json report{
        {"schema_version", kReportSchemaVersion},
        {"dry_run", true},
        {"workspace_root", workspace_root_},
        {"summary",
         {{"candidate_groups", static_cast<int>(groups.size())},
          {"proposed_documents", static_cast<int>(groups_json.size())},
          {"archive_files", archive_files},
          {"hash_merge_groups", hash_merge_groups},
          {"suffix_duplicate_groups", suffix_groups},
          {"warnings", static_cast<int>(warnings.size())}}},
        {"groups", groups_json},
        {"chunk_retag_plan", chunk_retag},
        {"legacy_id_map_preview", legacy_map},
        {"warnings", warnings},
        {"planned_steps", nlohmann::json::array({"M1", "M2", "M3", "M4", "M5", "M7", "M8"})},
        {"explicit_non_actions",
         nlohmann::json::array({"no writes to rag/attachments/",
                                "no writes to document_registry.json",
                                "no writes to rag_index.bin",
                                "no archive copies"})},
    };

    report["report_hash"] = computeReportHash(report);
    report["migration_run_id"] = std::string("alp-d0-") + report["report_hash"].get<std::string>().substr(0, 12);
    report["generated_at"] = iso8601NowUtc();

    return report;
}

} // namespace Thoth
