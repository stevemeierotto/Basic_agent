/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 9 / ALP-C create corpus document resource (Engine-authored; pure helpers)
 *
 * Contract: GUI initiates create; Engine owns id, filename, storage, chunking,
 * embedding, and atomicity. Acceptance JSON is separate from SSE INDEXING_*.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CORPUS_CREATE_H
#define THOTH_CORPUS_CREATE_H

#include "corpus_documents.h"

#include <cstdint>
#include <string>

namespace Thoth {
namespace CorpusCreate {

inline constexpr int kSchemaVersion = 1;
inline constexpr int kAlpSchemaVersion = 2;

/** Locked HTTP path (resource-oriented — not upload/multipart contract). */
inline constexpr const char* kHttpPath = "/v1/rag/documents";

/** Remove session↔document link (Local Note X). Document rows unchanged. */
inline constexpr const char* kHttpPathSessionLinkRemove = "/v1/rag/session-links/remove";

/** Engine /ready capability token when create-document is served. */
inline constexpr const char* kReadyCapability = "ingest";

inline constexpr const char* kAcceptedStatus = "accepted";

/** GUI Phase 9 — OperationResult operation token. */
inline constexpr const char* kOperationName = "create_document";

struct CreateDocumentRequest {
    std::string suggested_name;
    std::string content;
    std::string owner_context_id;
    std::string content_hash;
    std::int64_t local_source_mtime_sec = 0;
    std::string local_source_path;
    bool force_replace = false;
    bool dry_run = false;
};

inline nlohmann::json makeAcceptedResponse(const std::string& document_id,
                                           const std::string& document_name) {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"status", kAcceptedStatus},
        {"document",
         nlohmann::json{{"id", document_id}, {"name", document_name}}},
    };
}

inline nlohmann::json makeAlpAcceptedResponse(const std::string& document_id,
                                              const std::string& document_name,
                                              const std::string& revision_id,
                                              const std::string& action) {
    return nlohmann::json{
        {"schema_version", kAlpSchemaVersion},
        {"status", kAcceptedStatus},
        {"action", action},
        {"document",
         nlohmann::json{{"id", document_id},
                        {"name", document_name},
                        {"revision_id", revision_id}}},
    };
}

inline nlohmann::json makeDryRunResponse(const std::string& action,
                                         const std::string& document_id,
                                         const std::string& document_name,
                                         const std::string& reason) {
    nlohmann::json body{
        {"schema_version", kAlpSchemaVersion},
        {"dry_run", true},
        {"action", action},
        {"reason", reason},
    };
    if (!document_id.empty()) {
        body["document"] = {{"id", document_id}, {"name", document_name}};
    }
    return body;
}

inline bool hasRequiredAcceptedFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "create response must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    const int schema = body["schema_version"].get<int>();
    if (schema < 1) {
        error_out = "schema_version must be >= 1";
        return false;
    }
    if (body.value("dry_run", false) == true) {
        if (!body.contains("action") || !body["action"].is_string()) {
            error_out = "dry_run response requires action";
            return false;
        }
        return true;
    }
    if (!body.contains("status") || !body["status"].is_string()
        || body["status"].get<std::string>() != kAcceptedStatus) {
        error_out = "status must be \"accepted\"";
        return false;
    }
    if (!body.contains("document") || !body["document"].is_object()) {
        error_out = "document object required";
        return false;
    }
    const auto& doc = body["document"];
    if (!doc.contains("id") || !doc["id"].is_string() || doc["id"].get<std::string>().empty()) {
        error_out = "document.id required";
        return false;
    }
    if (!doc.contains("name") || !doc["name"].is_string() || doc["name"].get<std::string>().empty()) {
        error_out = "document.name required";
        return false;
    }
    if (schema >= kAlpSchemaVersion) {
        if (!body.contains("action") || !body["action"].is_string()) {
            error_out = "action required for schema_version >= 2";
            return false;
        }
        if (!doc.contains("revision_id") || !doc["revision_id"].is_string()
            || doc["revision_id"].get<std::string>().empty()) {
            error_out = "document.revision_id required for schema_version >= 2";
            return false;
        }
    }
    return true;
}

/** True when /ready capabilities array includes ingest. */
inline bool readyCapabilitiesIncludeIngest(const nlohmann::json& body) {
    if (!body.contains("capabilities") || !body["capabilities"].is_array()) {
        return false;
    }
    for (const auto& cap : body["capabilities"]) {
        if (cap.is_string() && cap.get<std::string>() == kReadyCapability) {
            return true;
        }
    }
    return false;
}

/** Sanitize a GUI-provided name to a single filename segment (Engine-owned). */
inline std::string sanitizeSuggestedFilename(std::string suggested) {
    while (!suggested.empty()
           && (suggested.front() == ' ' || suggested.front() == '\t')) {
        suggested.erase(suggested.begin());
    }
    while (!suggested.empty()
           && (suggested.back() == ' ' || suggested.back() == '\t')) {
        suggested.pop_back();
    }
    const auto pos = suggested.find_last_of("/\\");
    if (pos != std::string::npos) {
        suggested = suggested.substr(pos + 1);
    }
    if (suggested.empty() || suggested == "." || suggested == "..") {
        return "document.txt";
    }
    return suggested;
}

/** TCB4 — POST body for create document; optional session bind (TCB3 / P4). */
inline nlohmann::json makeCreateDocumentRequestBody(const std::string& name,
                                                      const std::string& content,
                                                      const std::string& backend_session_id) {
    nlohmann::json req = nlohmann::json{{"name", name}, {"content", content}};
    std::string sid = backend_session_id;
    while (!sid.empty() && (sid.front() == ' ' || sid.front() == '\t')) {
        sid.erase(sid.begin());
    }
    while (!sid.empty() && (sid.back() == ' ' || sid.back() == '\t')) {
        sid.pop_back();
    }
    if (!sid.empty()) {
        req["session_id"] = sid;
    }
    return req;
}

inline nlohmann::json makeCreateDocumentRequestBodyAlp(
    const CreateDocumentRequest& request) {
    nlohmann::json req = makeCreateDocumentRequestBody(
        request.suggested_name, request.content, request.owner_context_id);
    if (!request.content_hash.empty()) {
        req["content_hash"] = request.content_hash;
    }
    if (request.local_source_mtime_sec > 0) {
        req["local_source_mtime"] = request.local_source_mtime_sec;
    }
    if (!request.local_source_path.empty()) {
        req["local_source_path"] = request.local_source_path;
    }
    if (request.force_replace) {
        req["force_replace"] = true;
    }
    if (request.dry_run) {
        req["dry_run"] = true;
    }
    return req;
}

} // namespace CorpusCreate
} // namespace Thoth

#endif // THOTH_CORPUS_CREATE_H
