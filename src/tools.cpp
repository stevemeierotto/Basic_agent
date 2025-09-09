#include "../include/tools.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace Tools {

bool replaceName(const std::string& filePath,
                 const std::string& oldName,
                 const std::string& newName) {
    std::ifstream in(filePath);
    if (!in.is_open()) return false;

    std::ostringstream buffer;
    std::string line;
    bool changed = false;

    while (std::getline(in, line)) {
        size_t pos = 0;
        while ((pos = line.find(oldName, pos)) != std::string::npos) {
            line.replace(pos, oldName.size(), newName);
            pos += newName.size();
            changed = true;
        }
        buffer << line << "\n";
    }
    in.close();

    if (changed) {
        std::ofstream out(filePath);
        out << buffer.str();
    }
    return changed;
}



} // namespace Tools

