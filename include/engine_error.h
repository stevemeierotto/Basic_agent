/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineRuntime error schema (Plan F)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <stdexcept>
#include <string>

namespace Thoth {

enum class EngineErrorCode {
    INVALID_REQUEST,
    NOT_FOUND,
    ENGINE_BUSY,
    INTERNAL_ERROR
};

std::string engineErrorCodeToString(EngineErrorCode code);
int engineErrorHttpStatus(EngineErrorCode code);

struct EngineError {
    EngineErrorCode code{EngineErrorCode::INTERNAL_ERROR};
    std::string message;

    static EngineError invalidRequest(const std::string& message);
    static EngineError notFound(const std::string& message);
    static EngineError engineBusy(const std::string& message);
    static EngineError internalError(const std::string& message);

    std::string toJson() const;
};

class EngineException : public std::runtime_error {
public:
    explicit EngineException(EngineError error);

    const EngineError& error() const { return error_; }

private:
    EngineError error_;
};

/** Plan F session rule: empty → "default". */
std::string normalizeEngineSessionId(const std::string& session_id);

} // namespace Thoth
