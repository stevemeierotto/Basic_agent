/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 8 corpus document list resource (Engine-authored; pure helpers)
 *
 * Contract: structured JSON with mandatory schema_version. Storage layout
 * (filesystem, DB, etc.) is an implementation detail — never part of the GUI API.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CORPUS_DOCUMENTS_H
#define THOTH_CORPUS_DOCUMENTS_H

#include "json.hpp"

#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

namespace Thoth {
namespace CorpusDocuments {

inline constexpr int kSchemaVersion = 1;

/** Locked HTTP path (resource-oriented — not a directory listing). */
inline constexpr const char* kHttpPath = "/v1/rag/corpus";

/** Engine /ready capability token when this resource is served. */
inline constexpr const char* kReadyCapability = "corpus";

/** Phase 8 locked presentation strings (GUI may use verbatim). */
inline constexpr const char* kLoadingLabel = "Loading\u2026";
inline constexpr const char* kEmptyLabel = "Corpus is empty.";
inline constexpr const char* kUnavailableLabel = "Corpus listing unavailable.";

/** R5-G4 — populated inventory header (unscoped; not grounded Agent Context). */
inline constexpr const char* kInventoryPopulatedLabel = "Engine inventory (unscoped)";

inline nlohmann::json emptyV1List() {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"documents", nlohmann::json::array()},
    };
}

/** Invalid payload — forces GUI Unavailable (not Empty) after failed remote fetch. */
inline nlohmann::json unavailableFetchResult() {
    return nlohmann::json::object();
}

inline nlohmann::json makeDocument(const std::string& id,
                                   const std::string& name,
                                   const std::string& status,
                                   const std::optional<std::string>& indexed_at = std::nullopt,
                                   const std::optional<int>& chunk_count = std::nullopt,
                                   const std::optional<std::string>& failure_reason = std::nullopt) {
    nlohmann::json doc = nlohmann::json{
        {"id", id},
        {"name", name},
        {"status", status},
    };
    if (indexed_at.has_value()) {
        doc["indexed_at"] = *indexed_at;
    } else {
        doc["indexed_at"] = nullptr;
    }
    if (chunk_count.has_value()) {
        doc["chunk_count"] = *chunk_count;
    } else {
        doc["chunk_count"] = nullptr;
    }
    if (failure_reason.has_value() && !failure_reason->empty()) {
        doc["reason"] = *failure_reason;
    }
    return doc;
}

/** R2 — corpus document terminal failure (closed set enforced at Engine). */
inline bool isAllowedDocumentStatus(const std::string& status) {
    return status == "indexed" || status == "pending" || status == "failed";
}

/** Stable document id from an internal storage key (never exposed to GUI as a path). */
inline std::string stableDocumentId(const std::string& storage_key) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char ch : storage_key) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "doc-" << std::hex << hash;
    return out.str();
}

inline std::optional<std::string> formatIndexedAtIso(std::int64_t epoch_ms) {
    if (epoch_ms <= 0) {
        return std::nullopt;
    }
    const std::time_t t = static_cast<std::time_t>(epoch_ms / 1000);
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

inline bool hasRequiredV1Fields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "corpus list must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (body["schema_version"].get<int>() < 1) {
        error_out = "schema_version must be >= 1";
        return false;
    }
    if (!body.contains("documents") || !body["documents"].is_array()) {
        error_out = "documents must be an array";
        return false;
    }
    for (const auto& doc : body["documents"]) {
        if (!doc.is_object()) {
            error_out = "document entry must be an object";
            return false;
        }
        static const char* kRequired[] = {"id", "name", "status"};
        for (const char* key : kRequired) {
            if (!doc.contains(key) || !doc[key].is_string()) {
                error_out = std::string("document missing string field: ") + key;
                return false;
            }
        }
        const std::string status = doc["status"].get<std::string>();
        if (!isAllowedDocumentStatus(status)) {
            error_out = "document status must be indexed, pending, or failed";
            return false;
        }
        if (doc.contains("reason")
            && !(doc["reason"].is_null() || doc["reason"].is_string())) {
            error_out = "reason must be null or string";
            return false;
        }
        if (doc.contains("chunk_count")
            && !(doc["chunk_count"].is_null() || doc["chunk_count"].is_number_integer())) {
            error_out = "chunk_count must be null or integer";
            return false;
        }
        if (doc.contains("indexed_at")
            && !(doc["indexed_at"].is_null() || doc["indexed_at"].is_string())) {
            error_out = "indexed_at must be null or string";
            return false;
        }
    }
    return true;
}

inline bool isEffectivelyEmpty(const nlohmann::json& body) {
    if (!body.is_object()) {
        return true;
    }
    if (!body.contains("documents") || !body["documents"].is_array()) {
        return true;
    }
    return body["documents"].empty();
}

} // namespace CorpusDocuments
} // namespace Thoth

#endif // THOTH_CORPUS_DOCUMENTS_H
