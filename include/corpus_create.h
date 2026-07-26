/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 9 create corpus document resource (Engine-authored; pure helpers)
 *
 * Contract: GUI initiates create; Engine owns id, filename, storage, chunking,
 * embedding, and atomicity. Acceptance JSON is separate from SSE INDEXING_*.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CORPUS_CREATE_H
#define THOTH_CORPUS_CREATE_H

#include "corpus_documents.h"

#include <string>

namespace Thoth {
namespace CorpusCreate {

inline constexpr int kSchemaVersion = 1;

/** Locked HTTP path (resource-oriented — not upload/multipart contract). */
inline constexpr const char* kHttpPath = "/v1/rag/documents";

/** Engine /ready capability token when create-document is served. */
inline constexpr const char* kReadyCapability = "ingest";

inline constexpr const char* kAcceptedStatus = "accepted";

/** GUI Phase 9 — OperationResult operation token. */
inline constexpr const char* kOperationName = "create_document";

inline nlohmann::json makeAcceptedResponse(const std::string& document_id,
                                           const std::string& document_name) {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"status", kAcceptedStatus},
        {"document",
         nlohmann::json{{"id", document_id}, {"name", document_name}}},
    };
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
    if (body["schema_version"].get<int>() < 1) {
        error_out = "schema_version must be >= 1";
        return false;
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

} // namespace CorpusCreate
} // namespace Thoth

#endif // THOTH_CORPUS_CREATE_H
