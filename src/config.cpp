#include "../include/config.h"
#include "../include/memory_pruning_config.h"
#include "../include/plan_reuse_config.h"
#include "../include/runtime_latency_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <../include/json.hpp>

using json = nlohmann::json;

Config::Config()
    : temperature(0.7),
      top_p(1.0),
      max_tokens(512),
      max_results(5),
      similarity_threshold(0.7),
      similarity_metric("cosine"),
      verbosity(1),
      grag_directional(true),
      max_retries(3),
      log_level("info"),
      log_to_console(true),
      log_rotate_max_bytes(10 * 1024 * 1024),
      log_rotate_max_files(5),
      log_max_string_length(10000),
      memory_limit_mb(512),
      disk_quota_mb(1024),
      enable_tools(true),
      allow_network(true),
      allow_shell_exec(false),
      allow_web(true),
      allow_file_io(true),
      wq(0.4f),
      wd(0.4f),
      wt(0.2f),
      keyword_weight(0.3f),
      graph_weight(0.3f),
      synthesis_max_context_chars(
          static_cast<int>(Thoth::RuntimeLatency::kDefaultSynthesisMaxContextChars)),
      synthesis_num_predict(Thoth::RuntimeLatency::kDefaultSynthesisNumPredict),
      max_parallel_retrieval(Thoth::RuntimeLatency::kDefaultMaxParallelRetrieval),
      enable_retrieval_prefetch(Thoth::RuntimeLatency::kDefaultEnableRetrievalPrefetch),
      memory_max_hot_messages(Thoth::MemoryPruning::kMaxHotMessages),
      memory_max_hot_age_days(30),
      memory_prune_batch_size(Thoth::MemoryPruning::kPruneBatchSize)
{
    if (const char* maxReflections = std::getenv("THOTH_MAX_REFLECTIONS")) {
        try {
            max_reflections = std::stoi(maxReflections);
        } catch (...) {
        }
    }
}

bool Config::loadFromJson(const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx);
    std::ifstream file(path);
    if (!file.is_open()) return false;

    json j;
    try {
        file >> j;
    } catch (...) {
        return false;
    }

    if (j.contains("temperature")) temperature = j["temperature"];
    if (j.contains("top_p")) top_p = j["top_p"];
    if (j.contains("max_tokens")) max_tokens = j["max_tokens"];
    if (j.contains("verbosity")) verbosity = j["verbosity"];
    if (j.contains("max_retries")) max_retries = j["max_retries"];
    if (j.contains("grag_directional")) grag_directional = j["grag_directional"];
    if (j.contains("log_level") && j["log_level"].is_string()) log_level = j["log_level"];
    if (j.contains("log_to_console")) log_to_console = j["log_to_console"];
    if (j.contains("log_rotate_max_bytes")) log_rotate_max_bytes = j["log_rotate_max_bytes"];
    if (j.contains("log_rotate_max_files")) log_rotate_max_files = j["log_rotate_max_files"];
    if (j.contains("log_max_string_length")) log_max_string_length = j["log_max_string_length"];
    if (j.contains("memory_limit_mb")) memory_limit_mb = j["memory_limit_mb"];
    if (j.contains("disk_quota_mb")) disk_quota_mb = j["disk_quota_mb"];
    if (j.contains("database_path") && j["database_path"].is_string()) {
        database_path = j["database_path"];
    }
    if (j.contains("enable_tools")) enable_tools = j["enable_tools"];
    if (j.contains("allow_network")) allow_network = j["allow_network"];
    if (j.contains("allow_shell_exec")) allow_shell_exec = j["allow_shell_exec"];
    if (j.contains("allow_web")) allow_web = j["allow_web"];
    if (j.contains("allow_file_io")) allow_file_io = j["allow_file_io"];
    if (j.contains("similarity_metric") && j["similarity_metric"].is_string()) {
        similarity_metric = j["similarity_metric"];
    }
    if (j.contains("similarity_threshold")) similarity_threshold = j["similarity_threshold"];
    if (j.contains("llm_model")) llm_model = j["llm_model"];
    if (j.contains("embedding_model")) embedding_model = j["embedding_model"];
    if (j.contains("synthesis_max_context_chars")) {
        synthesis_max_context_chars = j["synthesis_max_context_chars"];
    }
    if (j.contains("synthesis_num_predict")) synthesis_num_predict = j["synthesis_num_predict"];
    if (j.contains("max_parallel_retrieval")) max_parallel_retrieval = j["max_parallel_retrieval"];
    if (j.contains("enable_retrieval_prefetch")) enable_retrieval_prefetch = j["enable_retrieval_prefetch"];

    if (j.contains("memory") && j["memory"].is_object()) {
        const auto& mem = j["memory"];
        if (mem.contains("max_hot_messages")) {
            memory_max_hot_messages = mem["max_hot_messages"].get<std::size_t>();
        }
        if (mem.contains("max_hot_age_days")) {
            memory_max_hot_age_days = mem["max_hot_age_days"].get<int>();
        }
        if (mem.contains("prune_batch_size")) {
            memory_prune_batch_size = mem["prune_batch_size"].get<std::size_t>();
        }
    }

    return true;
}

bool Config::saveToJson(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mtx);
    json j;

    j["temperature"] = temperature;
    j["top_p"] = top_p;
    j["max_tokens"] = max_tokens;
    j["max_results"] = max_results;
    j["similarity_threshold"] = similarity_threshold;
    j["similarity_metric"] = similarity_metric;
    j["llm_model"] = llm_model;
    j["embedding_model"] = embedding_model;
    j["verbosity"] = verbosity;
    j["max_retries"] = max_retries;
    j["grag_directional"] = grag_directional;
    j["log_level"] = log_level;
    j["log_to_console"] = log_to_console;
    j["log_rotate_max_bytes"] = log_rotate_max_bytes;
    j["log_rotate_max_files"] = log_rotate_max_files;
    j["log_max_string_length"] = log_max_string_length;
    j["memory_limit_mb"] = memory_limit_mb;
    j["disk_quota_mb"] = disk_quota_mb;
    j["database_path"] = database_path;
    j["enable_tools"] = enable_tools;
    j["allow_network"] = allow_network;
    j["allow_shell_exec"] = allow_shell_exec;
    j["allow_web"] = allow_web;
    j["allow_file_io"] = allow_file_io;
    j["synthesis_max_context_chars"] = synthesis_max_context_chars;
    j["synthesis_num_predict"] = synthesis_num_predict;
    j["max_parallel_retrieval"] = max_parallel_retrieval;
    j["enable_retrieval_prefetch"] = enable_retrieval_prefetch;
    j["memory"] = {
        {"max_hot_messages", memory_max_hot_messages},
        {"max_hot_age_days", memory_max_hot_age_days},
        {"prune_batch_size", memory_prune_batch_size}
    };

    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << j.dump(4);
    return true;
}

bool Config::loadRetrievalConfig(const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx);
    std::ifstream file(path);
    if (!file.is_open()) return false;

    json j;
    try {
        file >> j;
    } catch (...) {
        return false;
    }

    if (j.contains("retrieval_weights")) {
        auto rw = j["retrieval_weights"];
        if (rw.contains("query")) wq = rw["query"];
        if (rw.contains("direction")) wd = rw["direction"];
        if (rw.contains("trajectory")) wt = rw["trajectory"];
        if (rw.contains("keyword")) keyword_weight = rw["keyword"];
    }

    return true;
}

bool Config::saveRetrievalConfig(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mtx);
    json j;
    j["retrieval_weights"] = {
        {"query", wq},
        {"direction", wd},
        {"trajectory", wt},
        {"keyword", keyword_weight}
    };

    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << j.dump(4);
    return true;
}

std::string Config::get(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mtx);
    if (key == "temperature") return std::to_string(temperature);
    if (key == "top_p") return std::to_string(top_p);
    if (key == "max_tokens") return std::to_string(max_tokens);
    if (key == "max_results") return std::to_string(max_results);
    if (key == "similarity_threshold") return std::to_string(similarity_threshold);
    if (key == "similarity_metric") return similarity_metric;
    if (key == "llm_model") return llm_model;
    if (key == "embedding_model") return embedding_model;
    if (key == "verbosity") return std::to_string(verbosity);
    if (key == "max_retries") return std::to_string(max_retries);
    if (key == "grag_directional") return grag_directional ? "true" : "false";
    if (key == "log_level") return log_level;
    if (key == "log_to_console") return log_to_console ? "true" : "false";
    if (key == "log_rotate_max_bytes") return std::to_string(log_rotate_max_bytes);
    if (key == "log_rotate_max_files") return std::to_string(log_rotate_max_files);
    if (key == "log_max_string_length") return std::to_string(log_max_string_length);
    if (key == "memory_limit_mb") return std::to_string(memory_limit_mb);
    if (key == "disk_quota_mb") return std::to_string(disk_quota_mb);
    if (key == "database_path") return database_path;
    if (key == "enable_tools") return enable_tools ? "true" : "false";
    if (key == "allow_network") return allow_network ? "true" : "false";
    if (key == "allow_shell_exec") return allow_shell_exec ? "true" : "false";
    if (key == "allow_web") return allow_web ? "true" : "false";
    if (key == "allow_file_io") return allow_file_io ? "true" : "false";
    if (key == "wq") return std::to_string(wq);
    if (key == "wd") return std::to_string(wd);
    if (key == "wt") return std::to_string(wt);
    if (key == "keyword_weight") return std::to_string(keyword_weight);
    return "<unknown>";
}

bool Config::set(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mtx);
    try {
        if (key == "temperature") temperature = std::stod(value);
        else if (key == "top_p") top_p = std::stod(value);
        else if (key == "max_tokens") max_tokens = std::stoi(value);
        else if (key == "max_results") max_results = std::stoi(value);
        else if (key == "similarity_threshold") similarity_threshold = std::stod(value);
        else if (key == "similarity_metric") similarity_metric = value;
        else if (key == "llm_model") llm_model = value;
        else if (key == "embedding_model") embedding_model = value;
        else if (key == "verbosity") verbosity = std::stoi(value);
        else if (key == "max_retries") max_retries = std::stoi(value);
        else if (key == "grag_directional") grag_directional = (value == "true");
        else if (key == "log_level") log_level = value;
        else if (key == "log_to_console") log_to_console = (value == "true");
        else if (key == "log_rotate_max_bytes") log_rotate_max_bytes = std::stoull(value);
        else if (key == "log_rotate_max_files") log_rotate_max_files = std::stoull(value);
        else if (key == "log_max_string_length") log_max_string_length = std::stoull(value);
        else if (key == "memory_limit_mb") memory_limit_mb = std::stoul(value);
        else if (key == "disk_quota_mb") disk_quota_mb = std::stoul(value);
        else if (key == "database_path") database_path = value;
        else if (key == "enable_tools") enable_tools = (value == "true");
        else if (key == "allow_network") allow_network = (value == "true");
        else if (key == "allow_shell_exec") allow_shell_exec = (value == "true");
        else if (key == "allow_web") allow_web = (value == "true");
        else if (key == "allow_file_io") allow_file_io = (value == "true");
        else if (key == "wq") wq = std::stof(value);
        else if (key == "wd") wd = std::stof(value);
        else if (key == "wt") wt = std::stof(value);
        else if (key == "keyword_weight") keyword_weight = std::stof(value);
        else return false;
    } catch (...) {
        return false;
    }
    return true;
}

void Config::printConfig() const {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "--- Agent Config ---\n";
    std::cout << "temperature         : " << temperature << "\n";
    std::cout << "top_p               : " << top_p << "\n";
    std::cout << "max_tokens          : " << max_tokens << "\n";
    std::cout << "max_results         : " << max_results << "\n";
    std::cout << "similarity_threshold: " << similarity_threshold << "\n";
    std::cout << "similarity_metric   : " << similarity_metric << "\n";
    std::cout << "verbosity           : " << verbosity << "\n";
    std::cout << "max_retries         : " << max_retries << "\n";
    std::cout << "grag_directional    : " << (grag_directional ? "true" : "false") << "\n";
    std::cout << "log_level           : " << log_level << "\n";
    std::cout << "log_to_console      : " << (log_to_console ? "true" : "false") << "\n";
    std::cout << "log_rotate_max_bytes: " << log_rotate_max_bytes << "\n";
    std::cout << "log_rotate_max_files: " << log_rotate_max_files << "\n";
    std::cout << "log_max_string_len  : " << log_max_string_length << "\n";
    std::cout << "memory_limit_mb      : " << memory_limit_mb << "\n";
    std::cout << "disk_quota_mb       : " << disk_quota_mb << "\n";
    std::cout << "database_path       : " << database_path << "\n";
    std::cout << "enable_tools        : " << (enable_tools ? "true" : "false") << "\n";
    std::cout << "allow_network       : " << (allow_network ? "true" : "false") << "\n";
    std::cout << "allow_shell_exec    : " << (allow_shell_exec ? "true" : "false") << "\n";
    std::cout << "allow_web           : " << (allow_web ? "true" : "false") << "\n";
    std::cout << "allow_file_io       : " << (allow_file_io ? "true" : "false") << "\n";
    std::cout << "wq                  : " << wq << "\n";
    std::cout << "wd                  : " << wd << "\n";
    std::cout << "wt                  : " << wt << "\n";
    std::cout << "keyword_weight      : " << keyword_weight << "\n";
}

