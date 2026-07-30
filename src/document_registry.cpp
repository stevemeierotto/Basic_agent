/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP1 document registry (Attachment Lifecycle Protocol)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "document_registry.h"

#include "file_handler.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace Thoth {

namespace {

std::string jsonStringField(const nlohmann::json& row, const char* key) {
    if (!row.contains(key) || row[key].is_null() || !row[key].is_string()) {
        return "";
    }
    return row[key].get<std::string>();
}

} // namespace

nlohmann::json DocumentRegistry::emptyV1() {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"documents", nlohmann::json::array()},
        {"revisions", nlohmann::json::array()},
        {"session_links", nlohmann::json::array()},
    };
}

std::string DocumentRegistry::defaultRegistryPath() {
    FileHandler fh;
    return fh.getAgentWorkspacePath("document_registry.json");
}

bool DocumentRegistry::validateSchema(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "document registry must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (body["schema_version"].get<int>() < kSchemaVersion) {
        error_out = "schema_version too old";
        return false;
    }
    static const char* kArrays[] = {"documents", "revisions", "session_links"};
    for (const char* key : kArrays) {
        if (!body.contains(key) || !body[key].is_array()) {
            error_out = std::string(key) + " must be an array";
            return false;
        }
    }
    return true;
}

void DocumentRegistry::clear() {
    body_ = emptyV1();
    loaded_ = true;
}

bool DocumentRegistry::load(const std::string& path) {
    loaded_ = false;
    std::ifstream in(path);
    if (!in.is_open()) {
        body_ = emptyV1();
        loaded_ = true;
        return true;
    }
    try {
        nlohmann::json parsed;
        in >> parsed;
        std::string err;
        if (!validateSchema(parsed, err)) {
            std::cerr << "[ALP] document registry load failed: " << err << "\n";
            return false;
        }
        body_ = std::move(parsed);
        loaded_ = true;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[ALP] document registry parse failed: " << ex.what() << "\n";
        return false;
    }
}

bool DocumentRegistry::save(const std::string& path) const {
    const std::string temp_path = path + ".tmp." + std::to_string(getpid());
    try {
        if (!path.empty()) {
            const fs::path parent = fs::path(path).parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                fs::create_directories(parent, ec);
            }
        }
        {
            std::ofstream out(temp_path, std::ios::trunc);
            if (!out) {
                return false;
            }
            out << body_.dump(2);
            if (!out) {
                std::error_code ec;
                fs::remove(temp_path, ec);
                return false;
            }
        }
        std::error_code ec;
        fs::rename(temp_path, path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
            return false;
        }
        return true;
    } catch (...) {
        std::error_code ec;
        fs::remove(temp_path, ec);
        return false;
    }
}

nlohmann::json* DocumentRegistry::findRevisionRow(nlohmann::json& body,
                                                   const std::string& document_id,
                                                   const std::string& revision_id) {
    if (!body.contains("revisions") || !body["revisions"].is_array()) {
        return nullptr;
    }
    for (auto& row : body["revisions"]) {
        if (jsonStringField(row, "document_id") == document_id
            && jsonStringField(row, "revision_id") == revision_id) {
            return &row;
        }
    }
    return nullptr;
}

const nlohmann::json* DocumentRegistry::findRevisionRow(const nlohmann::json& body,
                                                        const std::string& document_id,
                                                        const std::string& revision_id) {
    return findRevisionRow(const_cast<nlohmann::json&>(body), document_id, revision_id);
}

nlohmann::json* DocumentRegistry::findDocumentRow(nlohmann::json& body,
                                                  const std::string& document_id) {
    if (!body.contains("documents") || !body["documents"].is_array()) {
        return nullptr;
    }
    for (auto& row : body["documents"]) {
        if (jsonStringField(row, "document_id") == document_id) {
            return &row;
        }
    }
    return nullptr;
}

const nlohmann::json* DocumentRegistry::findDocumentRow(const nlohmann::json& body,
                                                        const std::string& document_id) {
    return findDocumentRow(const_cast<nlohmann::json&>(body), document_id);
}

RegistryRevisionView DocumentRegistry::revisionViewFromRow(const nlohmann::json& row) {
    RegistryRevisionView view;
    view.document_id = jsonStringField(row, "document_id");
    view.revision_id = jsonStringField(row, "revision_id");
    view.state = jsonStringField(row, "state");
    view.content_hash = jsonStringField(row, "content_hash");
    view.storage_path = jsonStringField(row, "storage_path");
    view.chunk_count = row.value("chunk_count", 0);
    view.indexed_at_ms = row.value("indexed_at_ms", static_cast<std::int64_t>(0));
    view.failure_reason = jsonStringField(row, "reason");
    return view;
}

bool DocumentRegistry::beginRevision(const std::string& document_id,
                                     const std::string& revision_id,
                                     const std::string& storage_path,
                                     const std::string& initial_state,
                                     const std::string& content_hash,
                                     std::int64_t local_source_mtime_sec) {
    if (document_id.empty() || revision_id.empty()) {
        return true;
    }
    if (findRevisionRow(body_, document_id, revision_id) != nullptr) {
        return false;
    }
    nlohmann::json row{
        {"document_id", document_id},
        {"revision_id", revision_id},
        {"state", initial_state.empty() ? "indexing" : initial_state},
        {"storage_path", storage_path},
        {"chunk_count", 0},
        {"reason", ""},
    };
    if (!content_hash.empty()) {
        row["content_hash"] = content_hash;
    }
    if (local_source_mtime_sec > 0) {
        row["local_source_mtime"] = local_source_mtime_sec;
    }
    body_["revisions"].push_back(std::move(row));
    return true;
}

bool DocumentRegistry::markRevisionIndexing(const std::string& document_id,
                                            const std::string& revision_id) {
    if (document_id.empty() || revision_id.empty()) {
        return true;
    }
    nlohmann::json* row = findRevisionRow(body_, document_id, revision_id);
    if (row == nullptr) {
        return false;
    }
    (*row)["state"] = "indexing";
    return true;
}

bool DocumentRegistry::markRevisionCommitted(const std::string& document_id,
                                             const std::string& revision_id,
                                             int chunk_count) {
    if (document_id.empty() || revision_id.empty()) {
        return true;
    }
    nlohmann::json* row = findRevisionRow(body_, document_id, revision_id);
    if (row == nullptr) {
        if (!beginRevision(document_id, revision_id)) {
            return false;
        }
        row = findRevisionRow(body_, document_id, revision_id);
    }
    if (row == nullptr) {
        return false;
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    (*row)["state"] = "committed";
    (*row)["chunk_count"] = chunk_count;
    (*row)["reason"] = "";
    (*row)["indexed_at_ms"] = now_ms;

    if (nlohmann::json* doc = findDocumentRow(body_, document_id)) {
        (*doc)["current_revision_id"] = revision_id;
    }
    return true;
}

bool DocumentRegistry::markRevisionFailed(const std::string& document_id,
                                          const std::string& revision_id,
                                          const std::string& reason) {
    if (document_id.empty() || revision_id.empty()) {
        return true;
    }
    nlohmann::json* row = findRevisionRow(body_, document_id, revision_id);
    if (row == nullptr) {
        if (!beginRevision(document_id, revision_id)) {
            return false;
        }
        row = findRevisionRow(body_, document_id, revision_id);
    }
    if (row == nullptr) {
        return false;
    }
    (*row)["state"] = "failed";
    (*row)["reason"] = reason;
    return true;
}

bool DocumentRegistry::supersedePriorRevisions(const std::string& document_id,
                                               const std::string& active_revision_id) {
    if (document_id.empty() || active_revision_id.empty()) {
        return true;
    }
    if (!body_.contains("revisions") || !body_["revisions"].is_array()) {
        return true;
    }
    for (auto& row : body_["revisions"]) {
        if (jsonStringField(row, "document_id") != document_id) {
            continue;
        }
        if (jsonStringField(row, "revision_id") == active_revision_id) {
            continue;
        }
        if (jsonStringField(row, "state") == "committed") {
            row["state"] = "superseded";
        }
    }
    return true;
}

std::optional<std::string> DocumentRegistry::findDocumentIdByCanonicalName(
    const std::string& canonical_name) const {
    if (!body_.contains("documents") || !body_["documents"].is_array()) {
        return std::nullopt;
    }
    for (const auto& row : body_["documents"]) {
        if (jsonStringField(row, "canonical_name") == canonical_name) {
            return jsonStringField(row, "document_id");
        }
    }
    return std::nullopt;
}

std::optional<RegistryRevisionView> DocumentRegistry::findCommittedRevision(
    const std::string& document_id) const {
    if (!body_.contains("documents") || !body_.contains("revisions")) {
        return std::nullopt;
    }
    std::string current_rev;
    if (const nlohmann::json* doc = findDocumentRow(body_, document_id)) {
        current_rev = jsonStringField(*doc, "current_revision_id");
    }
    if (!current_rev.empty()) {
        if (const nlohmann::json* row = findRevisionRow(body_, document_id, current_rev)) {
            if (jsonStringField(*row, "state") == "committed") {
                return revisionViewFromRow(*row);
            }
        }
    }
    for (const auto& row : body_["revisions"]) {
        if (jsonStringField(row, "document_id") == document_id
            && jsonStringField(row, "state") == "committed") {
            return revisionViewFromRow(row);
        }
    }
    return std::nullopt;
}

std::optional<RegistryRevisionView> DocumentRegistry::findInFlightRevision(
    const std::string& document_id) const {
    if (!body_.contains("revisions")) {
        return std::nullopt;
    }
    for (const auto& row : body_["revisions"]) {
        if (jsonStringField(row, "document_id") != document_id) {
            continue;
        }
        const std::string state = jsonStringField(row, "state");
        if (state == "pending" || state == "indexing") {
            return revisionViewFromRow(row);
        }
    }
    return std::nullopt;
}

bool DocumentRegistry::ensureDocument(const std::string& document_id,
                                      const std::string& canonical_name,
                                      const std::string& storage_path) {
    if (document_id.empty() || canonical_name.empty()) {
        return false;
    }
    if (const auto existing = findDocumentIdByCanonicalName(canonical_name)) {
        if (*existing != document_id) {
            return false;
        }
    }
    if (nlohmann::json* doc = findDocumentRow(body_, document_id)) {
        (*doc)["canonical_name"] = canonical_name;
        (*doc)["storage_path"] = storage_path;
        return true;
    }
    body_["documents"].push_back({
        {"document_id", document_id},
        {"canonical_name", canonical_name},
        {"current_revision_id", nullptr},
        {"storage_path", storage_path},
    });
    return true;
}

bool DocumentRegistry::addSessionLink(const std::string& document_id,
                                      const std::string& session_id) {
    if (document_id.empty() || session_id.empty()) {
        return true;
    }
    if (hasSessionLink(document_id, session_id)) {
        return true;
    }
    body_["session_links"].push_back(
        {{"document_id", document_id}, {"session_id", session_id}});
    return true;
}

bool DocumentRegistry::hasSessionLink(const std::string& document_id,
                                      const std::string& session_id) const {
    if (!body_.contains("session_links")) {
        return false;
    }
    for (const auto& row : body_["session_links"]) {
        if (jsonStringField(row, "document_id") == document_id
            && jsonStringField(row, "session_id") == session_id) {
            return true;
        }
    }
    return false;
}

bool DocumentRegistry::lastRevisionFailed(const std::string& document_id) const {
    if (!body_.contains("revisions")) {
        return false;
    }
    const nlohmann::json* latest = nullptr;
    for (const auto& row : body_["revisions"]) {
        if (jsonStringField(row, "document_id") != document_id) {
            continue;
        }
        latest = &row;
    }
    return latest != nullptr && jsonStringField(*latest, "state") == "failed";
}

std::string DocumentRegistry::normalizeStoragePath(const std::string& path) {
    if (path.empty()) {
        return path;
    }
    try {
        return std::filesystem::absolute(path).lexically_normal().string();
    } catch (...) {
        return path;
    }
}

std::optional<std::string> DocumentRegistry::findDocumentIdByStoragePath(
    const std::string& storage_path) const {
    if (!body_.contains("documents") || !body_["documents"].is_array()) {
        return std::nullopt;
    }
    const std::string needle = normalizeStoragePath(storage_path);
    for (const auto& row : body_["documents"]) {
        if (normalizeStoragePath(jsonStringField(row, "storage_path")) == needle) {
            const std::string id = jsonStringField(row, "document_id");
            if (!id.empty()) {
                return id;
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> DocumentRegistry::findCanonicalName(
    const std::string& document_id) const {
    if (const nlohmann::json* doc = findDocumentRow(body_, document_id)) {
        const std::string name = jsonStringField(*doc, "canonical_name");
        if (!name.empty()) {
            return name;
        }
    }
    return std::nullopt;
}

std::optional<std::string> DocumentRegistry::currentCommittedStoragePath(
    const std::string& document_id) const {
    if (!body_.contains("documents") || !body_.contains("revisions")) {
        return std::nullopt;
    }
    const nlohmann::json* doc = findDocumentRow(body_, document_id);
    if (!doc) {
        return std::nullopt;
    }
    std::string current_rev = jsonStringField(*doc, "current_revision_id");
    if (current_rev.empty()) {
        return std::nullopt;
    }
    const nlohmann::json* row = findRevisionRow(body_, document_id, current_rev);
    if (!row || jsonStringField(*row, "state") != "committed") {
        return std::nullopt;
    }
    std::string path = jsonStringField(*row, "storage_path");
    if (path.empty()) {
        path = jsonStringField(*doc, "storage_path");
    }
    if (path.empty()) {
        return std::nullopt;
    }
    return path;
}

std::vector<std::string> DocumentRegistry::listLinkedDocumentIds(
    const std::string& session_id) const {
    std::vector<std::string> out;
    if (session_id.empty() || !body_.contains("session_links")) {
        return out;
    }
    for (const auto& row : body_["session_links"]) {
        if (jsonStringField(row, "session_id") != session_id) {
            continue;
        }
        const std::string id = jsonStringField(row, "document_id");
        if (id.empty()) {
            continue;
        }
        if (std::find(out.begin(), out.end(), id) == out.end()) {
            out.push_back(id);
        }
    }
    return out;
}

} // namespace Thoth
