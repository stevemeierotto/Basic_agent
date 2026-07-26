/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Phase 10 conversation/session authority (Engine-authored; pure helpers)
 *
 * Contract: Engine owns conversation state; GUI displays and navigates only.
 * Append-only — no replace-conversation.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CONVERSATION_AUTHORITY_H
#define THOTH_CONVERSATION_AUTHORITY_H

#include "json.hpp"

#include <cstdint>
#include <string>

namespace Thoth {
namespace ConversationAuthority {

inline constexpr int kSchemaVersion = 1;

inline constexpr const char* kHttpPathSessions = "/v1/conversation/sessions";
inline constexpr const char* kHttpPathTurns = "/v1/conversation/turns";

/** Engine /ready capability token when conversation authority is served. */
inline constexpr const char* kReadyCapability = "conversation";

inline nlohmann::json makeMessage(const std::string& role,
                                  const std::string& content,
                                  std::int64_t timestamp_ms) {
    return nlohmann::json{
        {"role", role},
        {"content", content},
        {"timestamp_ms", timestamp_ms},
    };
}

inline nlohmann::json makeCreateSessionResponse(const std::string& session_id) {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"session_id", session_id},
    };
}

inline nlohmann::json makeAppendTurnResponse(const std::string& session_id,
                                           const nlohmann::json& assistant_message) {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"session_id", session_id},
        {"assistant", assistant_message},
    };
}

inline nlohmann::json makeConversationResponse(const std::string& session_id,
                                               const nlohmann::json& messages) {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"session_id", session_id},
        {"messages", messages},
    };
}

inline nlohmann::json makeSummaryResponse(const std::string& session_id,
                                          const std::string& summary) {
    return nlohmann::json{
        {"schema_version", kSchemaVersion},
        {"session_id", session_id},
        {"summary", summary},
    };
}

inline nlohmann::json emptyConversation(const std::string& session_id) {
    return makeConversationResponse(session_id, nlohmann::json::array());
}

inline bool hasRequiredCreateSessionFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "create session response must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (!body.contains("session_id") || !body["session_id"].is_string()
        || body["session_id"].get<std::string>().empty()) {
        error_out = "session_id required";
        return false;
    }
    return true;
}

inline bool hasRequiredMessageFields(const nlohmann::json& msg, std::string& error_out) {
    if (!msg.is_object()) {
        error_out = "message must be an object";
        return false;
    }
    if (!msg.contains("role") || !msg["role"].is_string()) {
        error_out = "message.role required";
        return false;
    }
    if (!msg.contains("content") || !msg["content"].is_string()) {
        error_out = "message.content required";
        return false;
    }
    if (!msg.contains("timestamp_ms") || !msg["timestamp_ms"].is_number_integer()) {
        error_out = "message.timestamp_ms required";
        return false;
    }
    return true;
}

inline bool hasRequiredAppendTurnFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "append turn response must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (!body.contains("session_id") || !body["session_id"].is_string()) {
        error_out = "session_id required";
        return false;
    }
    if (!body.contains("assistant") || !hasRequiredMessageFields(body["assistant"], error_out)) {
        if (error_out.empty()) {
            error_out = "assistant message invalid";
        }
        return false;
    }
    return true;
}

inline bool hasRequiredConversationFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "conversation response must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (!body.contains("session_id") || !body["session_id"].is_string()) {
        error_out = "session_id required";
        return false;
    }
    if (!body.contains("messages") || !body["messages"].is_array()) {
        error_out = "messages must be an array";
        return false;
    }
    for (const auto& msg : body["messages"]) {
        if (!hasRequiredMessageFields(msg, error_out)) {
            return false;
        }
    }
    return true;
}

inline bool hasRequiredSummaryFields(const nlohmann::json& body, std::string& error_out) {
    if (!body.is_object()) {
        error_out = "summary response must be an object";
        return false;
    }
    if (!body.contains("schema_version") || !body["schema_version"].is_number_integer()) {
        error_out = "schema_version missing or not an integer";
        return false;
    }
    if (!body.contains("session_id") || !body["session_id"].is_string()) {
        error_out = "session_id required";
        return false;
    }
    if (!body.contains("summary") || !body["summary"].is_string()) {
        error_out = "summary required";
        return false;
    }
    return true;
}

inline bool readyCapabilitiesIncludeConversation(const nlohmann::json& body) {
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

} // namespace ConversationAuthority
} // namespace Thoth

#endif // THOTH_CONVERSATION_AUTHORITY_H
