#include "env_loader.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cctype>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace EnvLoader {

// Helper trim function
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

static std::string toUpper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

static bool isSensitiveKeyName(const std::string& key) {
    const std::string upper = toUpper(key);
    return upper.find("KEY") != std::string::npos ||
           upper.find("TOKEN") != std::string::npos ||
           upper.find("SECRET") != std::string::npos ||
           upper.find("PASSWORD") != std::string::npos ||
           upper.find("BEARER") != std::string::npos;
}

static std::string displayKeyName(const std::string& key) {
    return isSensitiveKeyName(key) ? "<redacted-key>" : key;
}

#ifndef _WIN32
static void validateEnvFileSecurity(const std::string& filename) {
    struct stat st {};
    if (stat(filename.c_str(), &st) != 0) return;

    const uid_t currentUid = geteuid();
    if (st.st_uid != currentUid) {
        std::cerr << "EnvLoader: Warning - .env owner differs from current user for: "
                  << filename << "\n";
    }

    const bool groupWritable = (st.st_mode & S_IWGRP) != 0;
    const bool worldWritable = (st.st_mode & S_IWOTH) != 0;
    if (groupWritable || worldWritable) {
        std::cerr << "EnvLoader: Warning - .env permissions are too open (group/world writable): "
                  << filename << "\n";
    }
}
#endif

bool loadEnvFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "EnvLoader: Warning - .env file not found: " << filename << "\n";
        return false; // .env file not found
    }

#ifndef _WIN32
    validateEnvFileSecurity(filename);
#endif

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            key = trim(key);
            value = trim(value);

            if (key.empty()) continue; // skip invalid lines

#ifdef _WIN32
            if (_putenv_s(key.c_str(), value.c_str()) != 0) {
                std::cerr << "EnvLoader: Failed to set environment variable: "
                          << displayKeyName(key) << "\n";
            }
#else
            if (setenv(key.c_str(), value.c_str(), 1) != 0) {
                std::cerr << "EnvLoader: Failed to set environment variable: "
                          << displayKeyName(key) << "\n";
            }
#endif
        }
    }

    return true;
}

} // namespace EnvLoader

