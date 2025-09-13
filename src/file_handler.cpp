#include "file_handler.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::string FileHandler::getAgentWorkspacePath(const std::string& filename) const {
    // Get full path of the binary
    fs::path exePath = fs::canonical("/proc/self/exe");

    // Go up one level (from build/ to project root)
    fs::path projectRoot = exePath.parent_path().parent_path();

    // Always point into agent_workspace
    fs::path workspace = projectRoot / "agent_workspace";

    if (!filename.empty()) {
        workspace /= filename;
    }

    std::cerr << "[FileHandler] Resolved path: " << workspace << "\n";
    return workspace.string();
}

std::string FileHandler::getMemoryPath() const {
    return getAgentWorkspacePath("memory.json");
}

std::string FileHandler::getRagPath(const std::string& filename) const {
    fs::path exePath = fs::canonical("/proc/self/exe");
    fs::path projectRoot = exePath.parent_path().parent_path();
    fs::path ragFolder = projectRoot / "agent_workspace" / "rag";

    // Ensure folder exists
    std::filesystem::create_directories(ragFolder);

    if (!filename.empty()) {
        ragFolder /= filename;
    }

    std::cerr << "[FileHandler] Resolved RAG path: " << ragFolder << "\n";
    return ragFolder.string();
}

std::string FileHandler::getRagDirectory() const {
    fs::path base = getAgentWorkspacePath();   // always agent_workspace
    fs::path ragDir = base / "rag";

    if (!fs::exists(ragDir)) {
        fs::create_directories(ragDir);
    }

    return ragDir.string();
}

