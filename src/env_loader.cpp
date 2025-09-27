#include "env_loader.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

namespace EnvLoader {

// Helper trim function
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

bool loadEnvFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "EnvLoader: Warning - .env file not found: " << filename << "\n";
        return false; // .env file not found
    }

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
                std::cerr << "EnvLoader: Failed to set environment variable: " << key << "\n";
            }
#else
            if (setenv(key.c_str(), value.c_str(), 1) != 0) {
                std::cerr << "EnvLoader: Failed to set environment variable: " << key << "\n";
            }
#endif
        }
    }

    return true;
}

} // namespace EnvLoader

