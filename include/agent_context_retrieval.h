/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — TCB2 Agent Context retrieval scope & trace
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "json.hpp"
#include <string>
#include <unordered_map>
#include <vector>

class IndexManager;
struct CodeChunk;
struct GragDiagnostics;

namespace Thoth {

class DocumentRegistry;

constexpr int kContextPolicyVersionV1 = 1;

struct ChunkClassificationContext {
    std::string attachment_owner_context_id;
    const DocumentRegistry* registry = nullptr;
    bool alp_enabled = false;
};

struct RetrievalScope {
    std::string retrieval_scope_id;
    int context_policy_version = kContextPolicyVersionV1;
    std::string active_context_key;
    std::string scope_type;
    std::vector<std::string> allowed_tiers;
    std::vector<std::string> selected_documents;
    std::vector<std::string> excluded_documents;
    /** ALP-F — session-link filter active (populated once at scope resolve). */
    bool alp_session_link_filter = false;
    std::vector<std::string> linked_document_ids;
    std::unordered_map<std::string, std::string> alp_committed_storage_by_document_id;

    nlohmann::json toJson() const;
};

struct RetrievalTrace {
    std::string request_id;
    RetrievalScope retrieval_scope;
    nlohmann::json grag = nlohmann::json::object();
    nlohmann::json grounding = nlohmann::json::object();

    nlohmann::json toJson() const;
    nlohmann::json toRetrievalTraceEnvelope() const;
};

/** TCB-P1 — default GUI / chat Agent Context (v1: active_context_key = session_id). */
RetrievalScope resolveAgentContextRetrievalScope(const std::string& active_context_key,
                                                 const IndexManager* indexManager = nullptr);

/** TCB-R4 — harness / Plan L explicit benchmark tier. */
RetrievalScope resolveBenchmarkExplicitScope();

bool chunkPassesRetrievalScope(const CodeChunk& chunk, const RetrievalScope& scope);

void classifyChunkMetadata(CodeChunk& chunk, const ChunkClassificationContext& ctx);

/** Legacy TCB3 helper — owner path lookup only (flags off). */
inline void classifyChunkMetadata(CodeChunk& chunk,
                                  const std::string& attachmentOwnerContextId) {
    ChunkClassificationContext ctx;
    ctx.attachment_owner_context_id = attachmentOwnerContextId;
    classifyChunkMetadata(chunk, ctx);
}

RetrievalTrace buildRetrievalTrace(const RetrievalScope& scope,
                                   const std::string& request_id,
                                   const GragDiagnostics* diagnostics = nullptr,
                                   const nlohmann::json& grounding = nlohmann::json::object());

std::string makeRetrievalScopeId(const std::string& active_context_key, int policyVersion);

} // namespace Thoth
