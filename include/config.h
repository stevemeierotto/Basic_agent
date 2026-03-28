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
    
    int max_tokens = 512;
    int max_results = 5;
    double similarity_threshold =0.6;
    std::string similarity_metric = "cosine";  // "cosine", "dot", "euclidean", "jaccard"

    std::string llm_model = "qwen2.5:3b";
    std::string embedding_model = "nomic-embed-text:v1.5";

    // Runtime parameters
    int verbosity = 1;  // 0 = silent, 1 = normal, 2 = debug
    bool grag_directional = true; // Phase 7 toggle
    int max_retries = 3;
    std::string log_level = "INFO";
    bool log_to_console = false;
    size_t log_rotate_max_bytes = 4 * 1024 * 1024;
    size_t log_rotate_max_files = 5;
    size_t log_max_string_length = 512;

    // Resource controls
    size_t memory_limit_mb = 256;   // soft cap for memory
    size_t disk_quota_mb = 512;     // max RAG/index size
    std::string database_path = ""; // Path to SQLite database

    // Tool flags
    bool enable_tools = true;
    bool allow_network = true;
    bool allow_shell_exec = false;
    bool allow_web = true;
    bool allow_file_io = true;

    // Phase 5.1: Retrieval Weights
    float wq = 0.4f;
    float wd = 0.4f;
    float wt = 0.0f;
    float keyword_weight = 0.3f;
    float graph_weight = 0.3f;

    // Cognate V2 — Scientific Reasoning Settings
    float convergence_epsilon = 0.05f;
    int stability_window = 2;
    int max_scientific_iterations = 5;

    // Load/Save from JSON or ENV
    bool loadFromJson(const std::string& path);
    bool saveToJson(const std::string& path) const;

    // Phase 5.1: Separate Retrieval Config
    bool loadRetrievalConfig(const std::string& path);
    bool saveRetrievalConfig(const std::string& path) const;

    // Query/Update at runtime
    std::string get(const std::string& key) const;
    bool set(const std::string& key, const std::string& value);

    // Utility
    void printConfig() const;

private:
    mutable std::mutex mtx; // to keep thread-safe if needed
};
