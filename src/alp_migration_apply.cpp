/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-D1 migration apply (brownfield gate)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "alp_migration_apply.h"

#include "alp_index_mutator.h"
#include "alp_migration_analyzer.h"
#include "alp_sha256.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace Thoth {
namespace {

std::string normalizePath(const fs::path& p) {
    try {
        return fs::absolute(p).lexically_normal().string();
    } catch (...) {
        return p.string();
    }
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

std::string generateUuidV4() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned> dis(0, 255);
    unsigned char bytes[16];
    for (auto& b : bytes) {
        b = static_cast<unsigned char>(dis(gen));
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3F) | 0x80);
    char buf[37];
    std::snprintf(buf,
                  sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  bytes[0],
                  bytes[1],
                  bytes[2],
                  bytes[3],
                  bytes[4],
                  bytes[5],
                  bytes[6],
                  bytes[7],
                  bytes[8],
                  bytes[9],
                  bytes[10],
                  bytes[11],
                  bytes[12],
                  bytes[13],
                  bytes[14],
                  bytes[15]);
    return buf;
}

bool atomicWriteJson(const fs::path& path, const nlohmann::json& body) {
    const std::string temp =
        path.string() + ".tmp." + std::to_string(static_cast<unsigned long>(getpid()));
    std::error_code ec;
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path(), ec);
    }
    {
        std::ofstream out(temp, std::ios::trunc);
        if (!out) {
            return false;
        }
        out << body.dump(2);
        if (!out) {
            fs::remove(temp, ec);
            return false;
        }
    }
    fs::rename(temp, path, ec);
    if (ec) {
        fs::remove(temp, ec);
        return false;
    }
    return true;
}

bool copyFileAtomic(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    if (!src.parent_path().empty()) {
        fs::create_directories(dst.parent_path(), ec);
    }
    const fs::path temp = dst.string() + ".tmp." + std::to_string(getpid());
    fs::copy_file(src, temp, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        return false;
    }
    fs::rename(temp, dst, ec);
    if (ec) {
        fs::remove(temp, ec);
        return false;
    }
    return true;
}

void copyRecursive(const fs::path& src, const fs::path& dst) {
    std::error_code ec;
    if (!fs::exists(src)) {
        return;
    }
    fs::create_directories(dst, ec);
    for (auto it = fs::recursive_directory_iterator(
             src, fs::directory_options::skip_permission_denied);
         it != fs::recursive_directory_iterator();
         ++it) {
        const fs::path rel = fs::relative(it->path(), src, ec);
        if (ec) {
            continue;
        }
        const fs::path target = dst / rel;
        if (it->is_directory()) {
            fs::create_directories(target, ec);
        } else if (it->is_regular_file()) {
            fs::create_directories(target.parent_path(), ec);
            fs::copy_file(it->path(), target, fs::copy_options::overwrite_existing, ec);
        }
    }
}

std::string tierForPath(const std::string& path,
                        const nlohmann::json* resolution_file) {
    if (resolution_file == nullptr || !resolution_file->contains("resolutions")) {
        return {};
    }
    for (const auto& row : (*resolution_file)["resolutions"]) {
        if (!row.is_object()) {
            continue;
        }
        if (row.value("path", "") == path) {
            return row.value("tier", "");
        }
    }
    return {};
}

bool isExcludedLiveRagSubpath(const fs::path& relative) {
    if (relative.empty()) {
        return false;
    }
    const std::string first = relative.begin()->string();
    return first == "seed" || first == "attachments" || first == "revisions"
        || first == "migration_archive";
}

} // namespace

AlpMigrationApply::AlpMigrationApply(std::string workspace_root)
    : workspace_root_(std::move(workspace_root)) {
    rag_root_ = normalizePath(fs::path(workspace_root_) / "rag");
    index_path_ = normalizePath(fs::path(rag_root_) / "rag_index.bin");
    registry_path_ = normalizePath(fs::path(workspace_root_) / "rag_attachment_registry.json");
    sessions_path_ = normalizePath(fs::path(workspace_root_) / "chat_sessions.json");
    document_registry_path_ =
        normalizePath(fs::path(workspace_root_) / "document_registry.json");
    legacy_map_path_ = normalizePath(fs::path(workspace_root_) / "legacy_id_map.json");
    migration_state_path_ =
        normalizePath(fs::path(workspace_root_) / "migration_state.json");
    snapshots_root_ =
        normalizePath(fs::path(workspace_root_) / "migration_snapshots");
}

bool AlpMigrationApply::writeMigrationState(const std::string& run_id,
                                            const std::string& phase,
                                            const std::string& status,
                                            const std::string& report_hash) const {
    nlohmann::json state{
        {"run_id", run_id},
        {"phase", phase},
        {"status", status},
        {"report_hash", report_hash},
        {"last_checkpoint_at", iso8601NowUtc()},
    };
    return atomicWriteJson(fs::path(migration_state_path_), state);
}

bool AlpMigrationApply::createM0Snapshot(const std::string& run_id,
                                         const std::string& report_hash,
                                         std::string& error_out) const {
    const fs::path m0_root = fs::path(snapshots_root_) / run_id / "M0";
    std::error_code ec;
    fs::create_directories(m0_root, ec);

    if (fs::exists(rag_root_)) {
        copyRecursive(rag_root_, m0_root / "rag");
    }

    nlohmann::json manifest{
        {"run_id", run_id},
        {"report_hash", report_hash},
        {"created_at", iso8601NowUtc()},
        {"files", nlohmann::json::array()},
    };

    const auto snapshot_file = [&](const fs::path& live, const std::string& rel_key) {
        if (!fs::is_regular_file(live)) {
            return;
        }
        const fs::path dest = m0_root / rel_key;
        fs::create_directories(dest.parent_path(), ec);
        fs::copy_file(live, dest, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            error_out = "M0 copy failed: " + live.string();
            return;
        }
        manifest["files"].push_back(
            {{"relative", rel_key}, {"sha256", sha256HexFile(live.string())}});
    };

    snapshot_file(fs::path(index_path_), "rag/rag_index.bin");
    snapshot_file(fs::path(registry_path_), "rag_attachment_registry.json");
    snapshot_file(fs::path(sessions_path_), "chat_sessions.json");
    snapshot_file(fs::path(document_registry_path_), "document_registry.json");

    if (!error_out.empty()) {
        return false;
    }

    if (!atomicWriteJson(m0_root / "manifest.json", manifest)) {
        error_out = "Failed to write M0 manifest";
        return false;
    }
    return true;
}

bool AlpMigrationApply::rollbackM0(const std::string& run_id, std::string& error_out) const {
    const fs::path m0_root = fs::path(snapshots_root_) / run_id / "M0";
    if (!fs::exists(m0_root / "manifest.json")) {
        error_out = "M0 manifest missing for run_id: " + run_id;
        return false;
    }

    std::error_code ec;

    if (fs::exists(m0_root / "rag")) {
        if (fs::exists(rag_root_)) {
            fs::remove_all(rag_root_, ec);
        }
        copyRecursive(m0_root / "rag", rag_root_);
    }

    const auto restore = [&](const char* rel, const fs::path& live) {
        const fs::path src = m0_root / rel;
        if (!fs::exists(src)) {
            if (fs::exists(live)) {
                fs::remove(live, ec);
            }
            return true;
        }
        fs::create_directories(live.parent_path(), ec);
        fs::copy_file(src, live, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            error_out = "rollback copy failed: " + live.string();
            return false;
        }
        return true;
    };

    if (!restore("rag/rag_index.bin", fs::path(index_path_))) {
        return false;
    }
    if (!restore("rag_attachment_registry.json", fs::path(registry_path_))) {
        return false;
    }
    if (!restore("chat_sessions.json", fs::path(sessions_path_))) {
        return false;
    }
    if (!restore("document_registry.json", fs::path(document_registry_path_))) {
        return false;
    }

    if (fs::exists(legacy_map_path_)) {
        fs::remove(legacy_map_path_, ec);
    }

    const fs::path legacy_renamed = registry_path_ + ".legacy";
    if (fs::exists(legacy_renamed)) {
        fs::remove(legacy_renamed, ec);
    }

    writeMigrationState(run_id, "ROLLBACK_M0", "rolled_back", "");

    const fs::path log_path =
        fs::path(workspace_root_) / "migration_logs" / (run_id + ".json");
    if (fs::exists(log_path)) {
        try {
            nlohmann::json log;
            std::ifstream in(log_path);
            in >> log;
            if (!log.contains("events")) {
                log["events"] = nlohmann::json::array();
            }
            log["events"].push_back(
                {{"event", "rolled_back"}, {"at", iso8601NowUtc()}});
            atomicWriteJson(log_path, log);
        } catch (...) {
        }
    }

    return true;
}

bool AlpMigrationApply::rollback(const std::string& run_id, std::string& error_out) {
    return rollbackM0(run_id, error_out);
}

bool AlpMigrationApply::validateReport(const nlohmann::json& approved_report,
                                       const nlohmann::json* resolution_file,
                                       std::string& error_out) const {
    if (!approved_report.contains("report_hash")
        || !approved_report["report_hash"].is_string()) {
        error_out = "Approved report missing report_hash";
        return false;
    }

    AlpMigrationAnalyzer analyzer(workspace_root_);
    const nlohmann::json live = analyzer.runDryRun();
    const std::string approved_hash = approved_report["report_hash"].get<std::string>();
    const std::string live_hash = live["report_hash"].get<std::string>();
    if (approved_hash != live_hash) {
        error_out = "Corpus drift: live report_hash does not match approved report";
        return false;
    }

    std::set<std::string> resolved_paths;
    if (resolution_file != nullptr && resolution_file->contains("resolutions")) {
        for (const auto& row : (*resolution_file)["resolutions"]) {
            if (row.is_object() && row.contains("path")) {
                resolved_paths.insert(row["path"].get<std::string>());
            }
        }
    }

    if (approved_report.contains("warnings")) {
        for (const auto& w : approved_report["warnings"]) {
            if (w.value("code", "") != "seed_operator_ambiguous") {
                continue;
            }
            const std::string path = w.value("path", "");
            if (!path.empty() && resolved_paths.count(path) == 0) {
                error_out = "Unresolved seed_operator_ambiguous: " + path;
                return false;
            }
        }
    }

    if (approved_report.contains("groups")) {
        for (const auto& group : approved_report["groups"]) {
            for (const auto& c : group.value("candidates", nlohmann::json::array())) {
                if (c.value("tier_classification", "") != "ambiguous") {
                    continue;
                }
                const std::string path = c.value("path", "");
                if (!path.empty() && resolved_paths.count(path) == 0) {
                    error_out = "Unresolved ambiguous tier: " + path;
                    return false;
                }
            }
            if (!group.contains("winner")) {
                continue;
            }
            const auto& winner = group["winner"];
            const std::string winner_path = winner.value("path", "");
            for (const auto& c : group.value("candidates", nlohmann::json::array())) {
                if (c.value("path", "") == winner_path
                    && c.value("tier_classification", "") == "ambiguous") {
                    if (resolved_paths.count(winner_path) == 0) {
                        error_out = "Winner has unresolved ambiguous tier: " + winner_path;
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

AlpMigrationApplyResult AlpMigrationApply::apply(const nlohmann::json& approved_report,
                                                 const std::string& run_id,
                                                 const nlohmann::json* resolution_file) {
    AlpMigrationApplyResult result;
    result.run_id = run_id;

    std::string validate_err;
    if (!validateReport(approved_report, resolution_file, validate_err)) {
        result.error = validate_err;
        return result;
    }

    const std::string report_hash = approved_report["report_hash"].get<std::string>();

    nlohmann::json migration_log{
        {"run_id", run_id},
        {"report_hash", report_hash},
        {"started_at", iso8601NowUtc()},
        {"events", nlohmann::json::array()},
        {"deferred_repairs", nlohmann::json::array()},
        {"document_assignments", nlohmann::json::array()},
    };

    const auto log_event = [&](const char* phase, const char* detail) {
        migration_log["events"].push_back(
            {{"phase", phase}, {"detail", detail}, {"at", iso8601NowUtc()}});
    };

    const auto fail = [&](const std::string& phase, const std::string& msg) {
        result.error = msg;
        log_event(phase.c_str(), msg.c_str());
        const fs::path log_path =
            fs::path(workspace_root_) / "migration_logs" / (run_id + ".json");
        fs::create_directories(log_path.parent_path());
        atomicWriteJson(log_path, migration_log);
        if (phase != std::string("PRE_VALIDATE") && phase != std::string("M0_SNAPSHOT")) {
            std::string rb_err;
            rollbackM0(run_id, rb_err);
            if (!rb_err.empty()) {
                result.error += "; rollback: " + rb_err;
            }
        }
        writeMigrationState(run_id, phase, "failed", report_hash);
        return result;
    };

    log_event("PRE_VALIDATE", "report_hash matched live corpus");

    std::string m0_err;
    if (!createM0Snapshot(run_id, report_hash, m0_err)) {
        return fail("M0_SNAPSHOT", m0_err.empty() ? "M0 snapshot failed" : m0_err);
    }
    writeMigrationState(run_id, "M0_SNAPSHOT", "in_progress", report_hash);
    log_event("M0_SNAPSHOT", "snapshot complete");

    const fs::path archive_root =
        fs::path(rag_root_) / "migration_archive" / run_id;
    std::error_code ec;

    for (const auto& group : approved_report["groups"]) {
        for (const auto& loser : group.value("losers", nlohmann::json::array())) {
            const std::string loser_path = loser.value("path", "");
            if (loser_path.empty() || !fs::is_regular_file(loser_path)) {
                continue;
            }
            const fs::path dest =
                archive_root / fs::path(loser_path).filename();
            fs::create_directories(dest.parent_path(), ec);
            fs::copy_file(loser_path, dest, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                return fail("ARCHIVE_LOSERS", "archive copy failed: " + loser_path);
            }
        }
    }
    writeMigrationState(run_id, "ARCHIVE_LOSERS", "in_progress", report_hash);
    log_event("ARCHIVE_LOSERS", "loser copies complete");

    struct GroupApply {
        std::string document_id;
        std::string revision_id;
        std::string attachment_path;
        std::string canonical_name;
        std::string winner_path;
        std::string winner_hash;
        int chunk_count = 0;
        std::vector<std::string> legacy_paths;
        std::set<std::string> session_links;
        bool degraded = false;
    };
    std::vector<GroupApply> applied_groups;

    const fs::path attachments_dir = fs::path(rag_root_) / "attachments";
    fs::create_directories(attachments_dir, ec);
    const fs::path tmp_dir = attachments_dir / ".tmp";
    fs::create_directories(tmp_dir, ec);

    for (const auto& group : approved_report["groups"]) {
        if (!group.contains("winner")) {
            continue;
        }
        const auto& winner = group["winner"];
        const std::string winner_path = winner.value("path", "");
        const std::string canonical_name = winner.value("proposed_canonical_name", "");
        if (winner_path.empty() || canonical_name.empty()) {
            continue;
        }

        std::string winner_tier;
        for (const auto& c : group.value("candidates", nlohmann::json::array())) {
            if (c.value("path", "") == winner_path) {
                winner_tier = c.value("tier_classification", "");
                break;
            }
        }
        const std::string resolved = tierForPath(winner_path, resolution_file);
        if (!resolved.empty()) {
            winner_tier = resolved == "seed" ? "seed_definite" : "operator_definite";
        }
        if (winner_tier == "seed_definite" || winner_tier == "seed_candidate") {
            continue;
        }
        if (winner_tier == "ambiguous") {
            return fail("PROMOTE_WINNERS", "ambiguous winner reached apply: " + winner_path);
        }
        if (!fs::is_regular_file(winner_path)) {
            return fail("PROMOTE_WINNERS", "winner file missing: " + winner_path);
        }

        GroupApply ga;
        ga.document_id = generateUuidV4();
        ga.revision_id = generateUuidV4();
        ga.canonical_name = canonical_name;
        ga.winner_path = winner_path;
        ga.attachment_path =
            normalizePath(attachments_dir / canonical_name);
        ga.winner_hash = sha256HexFile(winner_path);

        for (const auto& row : approved_report.value("chunk_retag_plan", nlohmann::json::array())) {
            if (row.value("legacy_fileName", "") == winner_path) {
                ga.chunk_count = row.value("chunk_count", 0);
                break;
            }
        }

        for (const auto& c : group.value("candidates", nlohmann::json::array())) {
            ga.legacy_paths.push_back(c.value("path", ""));
            for (const auto& s : c.value("registry_sessions", nlohmann::json::array())) {
                if (s.is_string()) {
                    ga.session_links.insert(s.get<std::string>());
                }
            }
        }
        for (const auto& s : group.value("session_links_preview", nlohmann::json::array())) {
            if (s.is_string()) {
                ga.session_links.insert(s.get<std::string>());
            }
        }

        for (const auto& w : approved_report["warnings"]) {
            if (w.value("code", "") == "degraded_index_all_zero"
                && w.value("path", "") == winner_path) {
                ga.degraded = true;
            }
        }

        const fs::path tmp_file = tmp_dir / canonical_name;
        if (!copyFileAtomic(fs::path(winner_path), tmp_file)) {
            return fail("PROMOTE_WINNERS", "promote temp copy failed: " + winner_path);
        }
        fs::rename(tmp_file, fs::path(ga.attachment_path), ec);
        if (ec) {
            return fail("PROMOTE_WINNERS", "promote rename failed: " + ga.attachment_path);
        }

        applied_groups.push_back(std::move(ga));
    }
    writeMigrationState(run_id, "PROMOTE_WINNERS", "in_progress", report_hash);
    log_event("PROMOTE_WINNERS", "winners promoted to rag/attachments/");

    nlohmann::json registry = nlohmann::json{
        {"schema_version", 1},
        {"documents", nlohmann::json::array()},
        {"revisions", nlohmann::json::array()},
        {"session_links", nlohmann::json::array()},
    };
    nlohmann::json legacy_map = nlohmann::json::object();

    for (const auto& group : approved_report["groups"]) {
        if (!group.contains("winner")) {
            continue;
        }
        const auto& winner = group["winner"];
        const std::string winner_path = winner.value("path", "");
        const std::string preview_id = winner.value("proposed_document_id_preview", "");

        const GroupApply* ga = nullptr;
        for (const auto& g : applied_groups) {
            if (g.winner_path == winner_path) {
                ga = &g;
                break;
            }
        }
        if (ga == nullptr) {
            continue;
        }

        registry["documents"].push_back({
            {"document_id", ga->document_id},
            {"canonical_name", ga->canonical_name},
            {"current_revision_id", ga->revision_id},
            {"storage_path", ga->attachment_path},
        });
        registry["revisions"].push_back({
            {"document_id", ga->document_id},
            {"revision_id", ga->revision_id},
            {"state", "committed"},
            {"storage_path", ga->attachment_path},
            {"chunk_count", ga->chunk_count},
            {"content_sha256", ga->winner_hash},
        });
        for (const auto& session_id : ga->session_links) {
            registry["session_links"].push_back(
                {{"document_id", ga->document_id}, {"session_id", session_id}});
        }

        for (const auto& c : group.value("candidates", nlohmann::json::array())) {
            const std::string stable = c.value("stable_document_id_preview", "");
            if (!stable.empty()) {
                legacy_map[stable] = ga->document_id;
            }
            const std::string path = c.value("path", "");
            if (!path.empty()) {
                legacy_map[path] = ga->document_id;
            }
        }
        if (!preview_id.empty()) {
            legacy_map[preview_id] = ga->document_id;
        }

        migration_log["document_assignments"].push_back({
            {"document_id", ga->document_id},
            {"revision_id", ga->revision_id},
            {"canonical_name", ga->canonical_name},
            {"winner_path", winner_path},
            {"attachment_path", ga->attachment_path},
        });

        if (ga->degraded) {
            migration_log["deferred_repairs"].push_back({
                {"document_id", ga->document_id},
                {"storage_path", ga->attachment_path},
                {"reason", "degraded_index_all_zero"},
            });
        }
    }

    if (!atomicWriteJson(fs::path(document_registry_path_), registry)) {
        return fail("REGISTRY_COMMIT", "document_registry.json write failed");
    }
    if (!atomicWriteJson(fs::path(legacy_map_path_), legacy_map)) {
        return fail("REGISTRY_COMMIT", "legacy_id_map.json write failed");
    }
    writeMigrationState(run_id, "REGISTRY_COMMIT", "in_progress", report_hash);
    log_event("REGISTRY_COMMIT", "registry and legacy_id_map committed");

    std::unordered_map<std::string, std::string> path_retag_map;
    for (const auto& ga : applied_groups) {
        for (const auto& legacy : ga.legacy_paths) {
            if (!legacy.empty()) {
                path_retag_map[legacy] = ga.attachment_path;
            }
        }
    }

    if (fs::is_regular_file(index_path_) && !path_retag_map.empty()) {
        auto snapshot = AlpIndexMutator::loadIndex(index_path_);
        if (!snapshot.header_json.empty() || !snapshot.chunks.empty()) {
            AlpIndexMutator::applyPathRetag(snapshot, path_retag_map);
            if (!AlpIndexMutator::saveIndexAtomic(index_path_, snapshot)) {
                return fail("INDEX_RETAG", "rag_index.bin save failed");
            }
        }
    }
    writeMigrationState(run_id, "INDEX_RETAG", "in_progress", report_hash);
    log_event("INDEX_RETAG", "index paths retagged");

    for (const auto& group : approved_report["groups"]) {
        for (const auto& loser : group.value("losers", nlohmann::json::array())) {
            const std::string loser_path = loser.value("path", "");
            if (loser_path.empty()) {
                continue;
            }
            fs::path p(loser_path);
            try {
                if (fs::exists(p) && !isExcludedLiveRagSubpath(fs::relative(p, rag_root_))) {
                    fs::remove(p, ec);
                }
            } catch (...) {
            }
        }
        if (group.contains("winner")) {
            const std::string winner_path = group["winner"].value("path", "");
            const std::string canonical = group["winner"].value("proposed_canonical_name", "");
            if (!winner_path.empty() && !canonical.empty()) {
                const fs::path wp(winner_path);
                const fs::path promoted = attachments_dir / canonical;
                try {
                    if (fs::exists(wp) && normalizePath(wp) != normalizePath(promoted)
                        && !isExcludedLiveRagSubpath(fs::relative(wp, rag_root_))) {
                        fs::remove(wp, ec);
                    }
                } catch (...) {
                }
            }
        }
    }
    writeMigrationState(run_id, "LEGACY_CLEANUP", "in_progress", report_hash);
    log_event("LEGACY_CLEANUP", "legacy flat duplicates removed");

    if (approved_report.contains("groups")) {
        for (const auto& group : approved_report["groups"]) {
            for (const auto& c : group.value("candidates", nlohmann::json::array())) {
                const std::string path = c.value("path", "");
                const std::string tier = c.value("tier_classification", "");
                const std::string resolved = tierForPath(path, resolution_file);
                const std::string effective =
                    !resolved.empty()
                        ? (resolved == "seed" ? "seed_definite" : tier)
                        : tier;
                if (effective != "seed_definite" && effective != "seed_candidate") {
                    continue;
                }
                if (path.empty() || !fs::is_regular_file(path)) {
                    continue;
                }
                const fs::path seed_dest =
                    fs::path(rag_root_) / "seed" / fs::path(path).filename();
                fs::create_directories(seed_dest.parent_path(), ec);
                if (!fs::exists(seed_dest)) {
                    fs::copy_file(path, seed_dest, fs::copy_options::overwrite_existing, ec);
                }
            }
        }
    }
    writeMigrationState(run_id, "SEED_RELOCATE", "in_progress", report_hash);
    log_event("SEED_RELOCATE", "seed candidates relocated");

    if (approved_report.contains("warnings")) {
        for (const auto& w : approved_report["warnings"]) {
            if (w.value("code", "") == "missing_storage") {
                migration_log["orphans"] = migration_log.value("orphans", nlohmann::json::array());
                migration_log["orphans"].push_back(w);
            }
        }
    }
    log_event("ORPHAN_LOG", "orphan warnings recorded");

    if (fs::exists(registry_path_)) {
        const fs::path legacy_renamed = registry_path_ + ".legacy";
        if (!fs::exists(legacy_renamed)) {
            fs::rename(registry_path_, legacy_renamed, ec);
        }
    }

    migration_log["completed_at"] = iso8601NowUtc();
    migration_log["status"] = "completed";
    const fs::path log_path =
        fs::path(workspace_root_) / "migration_logs" / (run_id + ".json");
    fs::create_directories(log_path.parent_path());
    atomicWriteJson(log_path, migration_log);

    writeMigrationState(run_id, "VERIFY_COMPLETE", "completed", report_hash);

    result.success = true;
    result.migration_log = migration_log;
    return result;
}

} // namespace Thoth
