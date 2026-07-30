/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineRuntime error schema (Plan F)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_error.h"

namespace Thoth {

std::string engineErrorCodeToString(EngineErrorCode code) {
    switch (code) {
    case EngineErrorCode::INVALID_REQUEST:
        return "INVALID_REQUEST";
    case EngineErrorCode::NOT_FOUND:
        return "NOT_FOUND";
    case EngineErrorCode::ENGINE_BUSY:
        return "ENGINE_BUSY";
    case EngineErrorCode::CONFLICT:
        return "CONFLICT";
    case EngineErrorCode::INTERNAL_ERROR:
        return "INTERNAL_ERROR";
    }
    return "INTERNAL_ERROR";
}

int engineErrorHttpStatus(EngineErrorCode code) {
    switch (code) {
    case EngineErrorCode::INVALID_REQUEST:
        return 400;
    case EngineErrorCode::NOT_FOUND:
        return 404;
    case EngineErrorCode::CONFLICT:
        return 409;
    case EngineErrorCode::ENGINE_BUSY:
        return 503;
    case EngineErrorCode::INTERNAL_ERROR:
        return 500;
    }
    return 500;
}

EngineError EngineError::invalidRequest(const std::string& message) {
    return {EngineErrorCode::INVALID_REQUEST, message, "", nlohmann::json::object()};
}

EngineError EngineError::notFound(const std::string& message) {
    return {EngineErrorCode::NOT_FOUND, message, "", nlohmann::json::object()};
}

EngineError EngineError::engineBusy(const std::string& message) {
    return {EngineErrorCode::ENGINE_BUSY, message, "", nlohmann::json::object()};
}

EngineError EngineError::internalError(const std::string& message) {
    return {EngineErrorCode::INTERNAL_ERROR, message, "", nlohmann::json::object()};
}

EngineError EngineError::conflict(const std::string& machine_code,
                                  const std::string& message,
                                  nlohmann::json details) {
    return {EngineErrorCode::CONFLICT, message, machine_code, std::move(details)};
}

std::string EngineError::toJson() const {
    nlohmann::json err = {{"code", engineErrorCodeToString(code)}, {"message", message}};
    if (!machine_code.empty()) {
        err["machine_code"] = machine_code;
    }
    if (!details.empty()) {
        err["details"] = details;
    }
    return nlohmann::json{{"error", err}}.dump();
}

EngineException::EngineException(EngineError error)
    : std::runtime_error(error.message), error_(std::move(error)) {}

std::string normalizeEngineSessionId(const std::string& session_id) {
    return session_id.empty() ? "default" : session_id;
}

} // namespace Thoth
