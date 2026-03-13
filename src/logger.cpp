#include "logger.h"

#include "file_handler.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <functional>
#include <filesystem>
#include <iostream>
#include <regex>

using json = nlohmann::json;

namespace {
thread_local std::string tlsRequestId;
thread_local std::string tlsSessionId;
}

StructuredLogger& StructuredLogger::instance() {
    static StructuredLogger logger;
    return logger;
}

StructuredLogger::StructuredLogger() {
    FileHandler fileHandler;
    logFilePath = fileHandler.getAgentWorkspacePath("app_log.jsonl");

    const char* levelEnv = std::getenv("THOTH_LOG_LEVEL");
    if (levelEnv && *levelEnv) {
        minLevel = parseLogLevel(levelEnv);
    }

    const char* maxLenEnv = std::getenv("THOTH_LOG_MAX_STRING");
    if (maxLenEnv && *maxLenEnv) {
        try {
            const auto parsed = std::stoull(maxLenEnv);
            if (parsed > 16) {
                maxStringLength = static_cast<std::size_t>(parsed);
            }
        } catch (...) {
        }
    }

    const char* consoleEnv = std::getenv("THOTH_LOG_CONSOLE");
    if (consoleEnv && *consoleEnv) {
        std::string value = consoleEnv;
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        writeToConsole = (value == "1" || value == "true" || value == "yes" || value == "on");
    }

    const char* rotateBytesEnv = std::getenv("THOTH_LOG_ROTATE_BYTES");
    if (rotateBytesEnv && *rotateBytesEnv) {
        try {
            const auto parsed = std::stoull(rotateBytesEnv);
            if (parsed >= 1024) {
                rotateMaxBytes = static_cast<std::size_t>(parsed);
            }
        } catch (...) {
        }
    }

    const char* rotateFilesEnv = std::getenv("THOTH_LOG_ROTATE_FILES");
    if (rotateFilesEnv && *rotateFilesEnv) {
        try {
            const auto parsed = std::stoull(rotateFilesEnv);
            if (parsed >= 1) {
                rotateMaxFiles = static_cast<std::size_t>(parsed);
            }
        } catch (...) {
        }
    }
}

void StructuredLogger::configure(
    LogLevel level,
    bool enableConsole,
    std::size_t maxBytes,
    std::size_t maxFiles,
    std::size_t maxValueLength) {
    std::lock_guard<std::mutex> lock(writeMutex);
    minLevel = level;
    writeToConsole = enableConsole;
    if (maxBytes >= 1024) {
        rotateMaxBytes = maxBytes;
    }
    if (maxFiles >= 1) {
        rotateMaxFiles = maxFiles;
    }
    if (maxValueLength >= 16) {
        maxStringLength = maxValueLength;
    }
}

void StructuredLogger::setContext(const std::string& requestId, const std::string& sessionId) {
    tlsRequestId = requestId;
    tlsSessionId = sessionId;
}

void StructuredLogger::clearContext() {
    tlsRequestId.clear();
    tlsSessionId.clear();
}

std::string StructuredLogger::currentRequestId() const {
    return tlsRequestId;
}

std::string StructuredLogger::currentSessionId() const {
    return tlsSessionId;
}

void StructuredLogger::log(
    LogLevel level,
    const std::string& component,
    const std::string& eventName,
    const std::string& message,
    const json& metadata,
    const std::string& requestId,
    const std::string& sessionId) {
    if (static_cast<int>(level) < static_cast<int>(minLevel)) {
        return;
    }

    const std::string resolvedRequestId = requestId.empty() ? tlsRequestId : requestId;
    const std::string resolvedSessionId = sessionId.empty() ? tlsSessionId : sessionId;

    auto sanitizeString = [&](const std::string& value) {
        std::string output = value;

        output = std::regex_replace(
            output,
            std::regex("Bearer\\s+[A-Za-z0-9._\\-]+", std::regex::icase),
            "Bearer [REDACTED]");

        output = std::regex_replace(
            output,
            std::regex("sk-[A-Za-z0-9]{8,}"),
            "sk-[REDACTED]");

        output = std::regex_replace(
            output,
            std::regex("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"),
            "[REDACTED_EMAIL]");

        output = std::regex_replace(
            output,
            std::regex("(api[_-]?key\\s*[:=]\\s*)([^\\s\\\"']+)", std::regex::icase),
            "$1[REDACTED]");

        if (output.size() > maxStringLength) {
            output = output.substr(0, maxStringLength) + "...[truncated]";
        }

        return output;
    };

    std::function<json(const json&, const std::string&)> sanitizeJson;
    sanitizeJson = [&](const json& node, const std::string& keyHint) -> json {
        if (node.is_object()) {
            json out = json::object();
            for (auto it = node.begin(); it != node.end(); ++it) {
                out[it.key()] = sanitizeJson(it.value(), it.key());
            }
            return out;
        }

        if (node.is_array()) {
            json out = json::array();
            for (const auto& entry : node) {
                out.push_back(sanitizeJson(entry, keyHint));
            }
            return out;
        }

        if (node.is_string()) {
            if (isSensitiveKey(keyHint)) {
                return "[REDACTED]";
            }

            const std::string value = node.get<std::string>();
            if (shouldRedactString(value)) {
                return sanitizeString(value);
            }

            return sanitizeString(value);
        }

        return node;
    };

    const std::string safeMessage = sanitizeString(message);
    const json safeMetadata = sanitizeJson(metadata, "");

    json entry;
    entry["timestamp_ms"] = nowMs();
    entry["level"] = levelToString(level);
    entry["component"] = component;
    entry["event_name"] = eventName;
    entry["request_id"] = resolvedRequestId;
    entry["session_id"] = resolvedSessionId;
    entry["message"] = safeMessage;
    entry["metadata"] = safeMetadata;
    const std::string entryLine = entry.dump();

    std::lock_guard<std::mutex> lock(writeMutex);

    rotateIfNeeded(entryLine.size() + 1);

    std::ofstream out(logFilePath, std::ios::app);
    if (!out.is_open()) {
        return;
    }

    out << entryLine << '\n';

    if (writeToConsole) {
        std::cerr << "[" << levelToString(level) << "] "
                  << component << ": " << safeMessage << std::endl;
    }
}

std::string StructuredLogger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
        default:
            return "INFO";
    }
}

LogLevel StructuredLogger::parseLogLevel(const std::string& level) {
    std::string upper = level;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    if (upper == "DEBUG") return LogLevel::Debug;
    if (upper == "INFO") return LogLevel::Info;
    if (upper == "WARN" || upper == "WARNING") return LogLevel::Warn;
    if (upper == "ERROR") return LogLevel::Error;
    return LogLevel::Info;
}

bool StructuredLogger::isSensitiveKey(const std::string& key) {
    std::string upper = key;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    return upper.find("KEY") != std::string::npos ||
           upper.find("TOKEN") != std::string::npos ||
           upper.find("SECRET") != std::string::npos ||
           upper.find("PASSWORD") != std::string::npos ||
           upper.find("AUTH") != std::string::npos;
}

bool StructuredLogger::shouldRedactString(const std::string& value) {
    static const std::regex sensitivePattern(
        R"((Bearer\s+[A-Za-z0-9._\-]+)|(sk-[A-Za-z0-9]{8,})|([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})|(api[_-]?key\s*[:=]))",
        std::regex::icase);

    return std::regex_search(value, sensitivePattern);
}

std::int64_t StructuredLogger::nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void StructuredLogger::rotateIfNeeded(std::size_t incomingBytes) {
    if (rotateMaxBytes == 0 || rotateMaxFiles == 0) {
        return;
    }

    std::error_code ec;
    const bool exists = std::filesystem::exists(logFilePath, ec);
    if (ec || !exists) {
        return;
    }

    const auto fileSize = std::filesystem::file_size(logFilePath, ec);
    if (ec || fileSize + incomingBytes < rotateMaxBytes) {
        return;
    }

    for (std::size_t idx = rotateMaxFiles; idx >= 1; --idx) {
        const std::string src = logFilePath + "." + std::to_string(idx);
        const std::string dst = logFilePath + "." + std::to_string(idx + 1);

        if (idx == rotateMaxFiles) {
            std::filesystem::remove(src, ec);
            ec.clear();
            continue;
        }

        if (std::filesystem::exists(src, ec) && !ec) {
            std::filesystem::rename(src, dst, ec);
            ec.clear();
        } else {
            ec.clear();
        }
    }

    const std::string firstRotated = logFilePath + ".1";
    std::filesystem::rename(logFilePath, firstRotated, ec);
    ec.clear();
}
