/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineRuntime error schema (Plan F)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_error.h"

#include <json.hpp>

namespace Thoth {

std::string engineErrorCodeToString(EngineErrorCode code) {
    switch (code) {
    case EngineErrorCode::INVALID_REQUEST:
        return "INVALID_REQUEST";
    case EngineErrorCode::NOT_FOUND:
        return "NOT_FOUND";
    case EngineErrorCode::ENGINE_BUSY:
        return "ENGINE_BUSY";
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
    case EngineErrorCode::ENGINE_BUSY:
        return 503;
    case EngineErrorCode::INTERNAL_ERROR:
        return 500;
    }
    return 500;
}

EngineError EngineError::invalidRequest(const std::string& message) {
    return {EngineErrorCode::INVALID_REQUEST, message};
}

EngineError EngineError::notFound(const std::string& message) {
    return {EngineErrorCode::NOT_FOUND, message};
}

EngineError EngineError::engineBusy(const std::string& message) {
    return {EngineErrorCode::ENGINE_BUSY, message};
}

EngineError EngineError::internalError(const std::string& message) {
    return {EngineErrorCode::INTERNAL_ERROR, message};
}

std::string EngineError::toJson() const {
    nlohmann::json body = {
        {"error",
         {{"code", engineErrorCodeToString(code)}, {"message", message}}}};
    return body.dump();
}

EngineException::EngineException(EngineError error)
    : std::runtime_error(error.message), error_(std::move(error)) {}

std::string normalizeEngineSessionId(const std::string& session_id) {
    return session_id.empty() ? "default" : session_id;
}

} // namespace Thoth
