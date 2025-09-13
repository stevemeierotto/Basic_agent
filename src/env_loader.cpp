#include "../include/env_loader.h"
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace EnvLoader {

bool loadEnvFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false; // .env file not found
    }

    std::string line;
    while (std::getline(file, line)) {
        // Trim spaces
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines or comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            // Trim again
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);

            // Set environment variable
#ifdef _WIN32
            _putenv_s(key.c_str(), value.c_str());
#else
            setenv(key.c_str(), value.c_str(), 1);
#endif
        }
    }

    return true;
}

} // namespace EnvLoader

