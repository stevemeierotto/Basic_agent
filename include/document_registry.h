/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP1 document registry (Attachment Lifecycle Protocol)
 *
 * Engine-authoritative attachment identity store. ALP-A: load/save scaffold only;
 * legacy ingest continues via path-key registry until THOTH_ALP_ENABLED=1.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_DOCUMENT_REGISTRY_H
#define THOTH_DOCUMENT_REGISTRY_H

#include "json.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace Thoth {

struct RegistryRevisionView {
    std::string document_id;
    std::string revision_id;
    std::string state;
    std::string content_hash;
    std::string storage_path;
    int chunk_count = 0;
    std::int64_t indexed_at_ms = 0;
    std::string failure_reason;
};

class DocumentRegistry {
public:
    static constexpr int kSchemaVersion = 1;

    DocumentRegistry() = default;

    const nlohmann::json& body() const { return body_; }
    bool loaded() const { return loaded_; }

    /** Load from agent_workspace/document_registry.json; creates empty store if missing. */
    bool load(const std::string& path);

    /** Persist current body (atomic write). */
    bool save(const std::string& path) const;

    /** Reset to empty v1 document. */
    void clear();

    static nlohmann::json emptyV1();

    static std::string defaultRegistryPath();

    /** ALP-B — revision lifecycle (no-op when ids empty). Returns false on schema conflict. */
    bool beginRevision(const std::string& document_id,
                       const std::string& revision_id,
                       const std::string& storage_path = "",
                       const std::string& initial_state = "indexing",
                       const std::string& content_hash = "",
                       std::int64_t local_source_mtime_sec = 0);

    bool markRevisionIndexing(const std::string& document_id, const std::string& revision_id);
    bool markRevisionCommitted(const std::string& document_id,
                               const std::string& revision_id,
                               int chunk_count);
    bool markRevisionFailed(const std::string& document_id,
                            const std::string& revision_id,
                            const std::string& reason);
    bool supersedePriorRevisions(const std::string& document_id,
                                 const std::string& active_revision_id);

    /** ALP-C — document slot + session links. */
    std::optional<std::string> findDocumentIdByCanonicalName(
        const std::string& canonical_name) const;

    std::optional<RegistryRevisionView> findCommittedRevision(
        const std::string& document_id) const;

    std::optional<RegistryRevisionView> findInFlightRevision(
        const std::string& document_id) const;

    bool ensureDocument(const std::string& document_id,
                        const std::string& canonical_name,
                        const std::string& storage_path);

    bool addSessionLink(const std::string& document_id, const std::string& session_id);

    bool hasSessionLink(const std::string& document_id, const std::string& session_id) const;

    bool lastRevisionFailed(const std::string& document_id) const;

    /** ALP-F — normalized path match against documents[].storage_path. */
    std::optional<std::string> findDocumentIdByStoragePath(
        const std::string& storage_path) const;

    std::optional<std::string> findCanonicalName(const std::string& document_id) const;

    /** Committed storage path for document.current_revision_id only. */
    std::optional<std::string> currentCommittedStoragePath(
        const std::string& document_id) const;

    std::vector<std::string> listLinkedDocumentIds(const std::string& session_id) const;

    static std::string normalizeStoragePath(const std::string& path);

private:
    nlohmann::json body_ = emptyV1();
    bool loaded_ = false;

    static nlohmann::json* findDocumentRow(nlohmann::json& body,
                                           const std::string& document_id);
    static const nlohmann::json* findDocumentRow(const nlohmann::json& body,
                                                 const std::string& document_id);

    static nlohmann::json* findRevisionRow(nlohmann::json& body,
                                           const std::string& document_id,
                                           const std::string& revision_id);
    static const nlohmann::json* findRevisionRow(const nlohmann::json& body,
                                                 const std::string& document_id,
                                                 const std::string& revision_id);

    static RegistryRevisionView revisionViewFromRow(const nlohmann::json& row);

    static bool validateSchema(const nlohmann::json& body, std::string& error_out);
};

} // namespace Thoth

#endif // THOTH_DOCUMENT_REGISTRY_H
