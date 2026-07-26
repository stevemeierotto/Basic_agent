/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — TCB2 Agent Context retrieval scope & trace
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/agent_context_retrieval.h"
#include "../include/chunkers/code_chunk.h"
#include "../include/grag_diagnostics.h"
#include "../include/index_manager.h"

#include <atomic>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace Thoth {
namespace {

std::atomic<std::uint64_t> g_scopeSerial{0};

std::string lowerBasename(const std::string& path) {
    try {
        std::string name = fs::path(path).filename().string();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return name;
    } catch (...) {
        return path;
    }
}

bool isPlanLBenchmarkBasename(const std::string& lowerName) {
    return lowerName == "grag.md" || lowerName == "howto.md" || lowerName == "agents.md" ||
           lowerName == "cognate.md";
}

bool pathIndicatesPlanLSeed(const std::string& filePath) {
    const std::string lower = [&] {
        std::string s = filePath;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }();
    return lower.find("seed_rag") != std::string::npos ||
           lower.find("/docker/") != std::string::npos;
}

std::string documentStemFromPath(const std::string& filePath) {
    try {
        return fs::path(filePath).filename().string();
    } catch (...) {
        return filePath;
    }
}

void appendUnique(std::vector<std::string>& docs, const std::string& doc) {
    if (doc.empty()) {
        return;
    }
    if (std::find(docs.begin(), docs.end(), doc) == docs.end()) {
        docs.push_back(doc);
    }
}

} // namespace

std::string makeRetrievalScopeId(const std::string& active_context_key, int policyVersion) {
    const auto serial = ++g_scopeSerial;
    return "rs_" + std::to_string(policyVersion) + "_" + active_context_key + "_" +
           std::to_string(serial);
}

nlohmann::json RetrievalScope::toJson() const {
    return {{"retrieval_scope_id", retrieval_scope_id},
            {"context_policy_version", context_policy_version},
            {"active_context_key", active_context_key},
            {"scope_type", scope_type},
            {"allowed_tiers", allowed_tiers},
            {"selected_documents", selected_documents},
            {"excluded_documents", excluded_documents}};
}

nlohmann::json RetrievalTrace::toJson() const {
    return {{"request_id", request_id},
            {"retrieval_scope", retrieval_scope.toJson()},
            {"grag", grag},
            {"grounding", grounding}};
}

nlohmann::json RetrievalTrace::toRetrievalTraceEnvelope() const {
    return {{"retrieval_trace", toJson()}};
}

void classifyChunkMetadata(CodeChunk& chunk, const std::string& attachmentOwnerContextId) {
    const std::string base = lowerBasename(chunk.fileName);
    chunk.owner_context_id.clear();

    if (!attachmentOwnerContextId.empty()) {
        chunk.corpus_tier = "session_attachment";
        chunk.owner_context_id = attachmentOwnerContextId;
        return;
    }

    if (isPlanLBenchmarkBasename(base) && pathIndicatesPlanLSeed(chunk.fileName)) {
        chunk.corpus_tier = "benchmark";
        return;
    }

    if (isPlanLBenchmarkBasename(base)) {
        chunk.corpus_tier = "system_reference";
        return;
    }

    chunk.corpus_tier = "legacy_orphan";
}

bool chunkPassesRetrievalScope(const CodeChunk& chunk, const RetrievalScope& scope) {
    if (scope.allowed_tiers.empty()) {
        return true;
    }

    const std::string& tier = chunk.corpus_tier;
    auto tierAllowed = [&](const char* allowed) {
        return std::find(scope.allowed_tiers.begin(), scope.allowed_tiers.end(), allowed) !=
               scope.allowed_tiers.end();
    };

    if (tier == "session_attachment") {
        if (!tierAllowed("session_attachment")) {
            return false;
        }
        return !scope.active_context_key.empty() &&
               chunk.owner_context_id == scope.active_context_key;
    }

    if (tier == "benchmark") {
        return tierAllowed("benchmark");
    }
    if (tier == "system_reference") {
        return tierAllowed("system_reference");
    }
    if (tier == "legacy_orphan") {
        return tierAllowed("legacy_orphan");
    }
    if (tier == "shared_knowledge") {
        return tierAllowed("shared_knowledge");
    }

    return false;
}

RetrievalScope resolveAgentContextRetrievalScope(const std::string& active_context_key,
                                                 const IndexManager* indexManager) {
    RetrievalScope scope;
    scope.context_policy_version = kContextPolicyVersionV1;
    scope.active_context_key = active_context_key;
    scope.scope_type = "default_agent_context";
    scope.allowed_tiers = {"session_attachment"};
    scope.retrieval_scope_id = makeRetrievalScopeId(active_context_key, scope.context_policy_version);

    if (indexManager) {
        for (const auto& chunk : indexManager->getChunks()) {
            if (chunk.corpus_tier == "session_attachment" &&
                chunk.owner_context_id == active_context_key) {
                appendUnique(scope.selected_documents, documentStemFromPath(chunk.fileName));
            }
        }
    }
    return scope;
}

RetrievalScope resolveBenchmarkExplicitScope() {
    RetrievalScope scope;
    scope.context_policy_version = kContextPolicyVersionV1;
    scope.active_context_key = "harness";
    scope.scope_type = "benchmark_explicit";
    scope.allowed_tiers = {"benchmark", "system_reference"};
    scope.retrieval_scope_id = makeRetrievalScopeId("harness", scope.context_policy_version);
    return scope;
}

RetrievalTrace buildRetrievalTrace(const RetrievalScope& scope,
                                   const std::string& request_id,
                                   const GragDiagnostics* diagnostics,
                                   const nlohmann::json& grounding) {
    RetrievalTrace trace;
    trace.request_id = request_id;
    trace.retrieval_scope = scope;
    trace.grounding = grounding;
    if (diagnostics) {
        trace.grag = diagnostics->to_json();
    }
    return trace;
}

} // namespace Thoth
