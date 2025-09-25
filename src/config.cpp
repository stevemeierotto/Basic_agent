#include "../include/config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <../include/json.hpp>

using json = nlohmann::json;

Config::Config()
    : temperature(0.7),
      top_p(1.0),
      max_tokens(512),
      max_results(5),            // default topK for RAG
      similarity_threshold(0.7)  // optional
{}

bool Config::loadFromJson(const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx);
    std::ifstream file(path);
    if (!file.is_open()) return false;

    json j;
    file >> j;

    if (j.contains("temperature")) temperature = j["temperature"];
    if (j.contains("top_p")) top_p = j["top_p"];
    if (j.contains("max_tokens")) max_tokens = j["max_tokens"];
    if (j.contains("verbosity")) verbosity = j["verbosity"];
    if (j.contains("max_retries")) max_retries = j["max_retries"];
    if (j.contains("memory_limit_mb")) memory_limit_mb = j["memory_limit_mb"];
    if (j.contains("disk_quota_mb")) disk_quota_mb = j["disk_quota_mb"];
    if (j.contains("allow_web")) allow_web = j["allow_web"];
    if (j.contains("allow_file_io")) allow_file_io = j["allow_file_io"];

    return true;
}

bool Config::saveToJson(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mtx);
    json j;

    j["temperature"] = temperature;
    j["top_p"] = top_p;
    j["max_tokens"] = max_tokens;
    j["verbosity"] = verbosity;
    j["max_retries"] = max_retries;
    j["memory_limit_mb"] = memory_limit_mb;
    j["disk_quota_mb"] = disk_quota_mb;
    j["allow_web"] = allow_web;
    j["allow_file_io"] = allow_file_io;

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
    if (key == "verbosity") return std::to_string(verbosity);
    if (key == "max_retries") return std::to_string(max_retries);
    if (key == "memory_limit_mb") return std::to_string(memory_limit_mb);
    if (key == "disk_quota_mb") return std::to_string(disk_quota_mb);
    if (key == "allow_web") return allow_web ? "true" : "false";
    if (key == "allow_file_io") return allow_file_io ? "true" : "false";
    return "<unknown>";
}

bool Config::set(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mtx);
    try {
        if (key == "temperature") temperature = std::stod(value);
        else if (key == "top_p") top_p = std::stod(value);
        else if (key == "max_tokens") max_tokens = std::stoi(value);
        else if (key == "verbosity") verbosity = std::stoi(value);
        else if (key == "max_retries") max_retries = std::stoi(value);
        else if (key == "memory_limit_mb") memory_limit_mb = std::stoul(value);
        else if (key == "disk_quota_mb") disk_quota_mb = std::stoul(value);
        else if (key == "allow_web") allow_web = (value == "true");
        else if (key == "allow_file_io") allow_file_io = (value == "true");
        else return false;
    } catch (...) {
        return false;
    }
    return true;
}

void Config::printConfig() const {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "--- Agent Config ---\n";
    std::cout << "temperature     : " << temperature << "\n";
    std::cout << "top_p           : " << top_p << "\n";
    std::cout << "max_tokens      : " << max_tokens << "\n";
    std::cout << "verbosity       : " << verbosity << "\n";
    std::cout << "max_retries     : " << max_retries << "\n";
    std::cout << "memory_limit_mb : " << memory_limit_mb << "\n";
    std::cout << "disk_quota_mb   : " << disk_quota_mb << "\n";
    std::cout << "allow_web       : " << (allow_web ? "true" : "false") << "\n";
    std::cout << "allow_file_io   : " << (allow_file_io ? "true" : "false") << "\n";
}

