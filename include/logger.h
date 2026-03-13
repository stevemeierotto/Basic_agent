#pragma once

#include <json.hpp>

#include <cstdint>
#include <mutex>
#include <string>

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error
};

class StructuredLogger {
public:
    static StructuredLogger& instance();
    static LogLevel parseLogLevel(const std::string& level);

    void configure(
        LogLevel level,
        bool writeToConsole,
        std::size_t rotateMaxBytes,
        std::size_t rotateMaxFiles,
        std::size_t maxValueLength = 512);

    void setContext(const std::string& requestId, const std::string& sessionId = "");
    void clearContext();
    std::string currentRequestId() const;
    std::string currentSessionId() const;

    void log(
        LogLevel level,
        const std::string& component,
        const std::string& eventName,
        const std::string& message,
        const nlohmann::json& metadata = nlohmann::json::object(),
        const std::string& requestId = "",
        const std::string& sessionId = "");

private:
    StructuredLogger();

    static std::string levelToString(LogLevel level);
    static std::int64_t nowMs();
    static bool isSensitiveKey(const std::string& key);
    static bool shouldRedactString(const std::string& value);
    void rotateIfNeeded(std::size_t incomingBytes);

    std::string logFilePath;
    LogLevel minLevel = LogLevel::Info;
    bool writeToConsole = false;
    std::size_t rotateMaxBytes = 4 * 1024 * 1024;
    std::size_t rotateMaxFiles = 5;
    std::size_t maxStringLength = 512;
    std::mutex writeMutex;
};
