#pragma once
#include <string>
#include <unordered_map>
#include <mutex>

class Config {
public:
    Config();
    // Core LLM parameters
    double temperature = 0.7;
    double top_p = 1.0;
    double similarity_threshold =0.7;
    int max_tokens = 512;
    int max_results = 5;

    // Runtime parameters
    int verbosity = 1;  // 0 = silent, 1 = normal, 2 = debug
    int max_retries = 3;

    // Resource controls
    size_t memory_limit_mb = 256;   // soft cap for memory
    size_t disk_quota_mb = 512;     // max RAG/index size

    // Tool flags
    bool allow_web = true;
    bool allow_file_io = true;

    // Load/Save from JSON or ENV
    bool loadFromJson(const std::string& path);
    bool saveToJson(const std::string& path) const;

    // Query/Update at runtime
    std::string get(const std::string& key) const;
    bool set(const std::string& key, const std::string& value);

    // Utility
    void printConfig() const;

private:
    mutable std::mutex mtx; // to keep thread-safe if needed
};

