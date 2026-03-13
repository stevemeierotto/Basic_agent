#include "file_handler.h"
#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <vector>

namespace fs = std::filesystem;

static std::string getEnvOrEmpty(const char* key) {
    const char* value = std::getenv(key);
    return (value && *value) ? std::string(value) : std::string();
}

std::string FileHandler::getProjectRoot() const {
    std::string explicitRoot = getEnvOrEmpty("THOTH_PROJECT_ROOT");
    if (!explicitRoot.empty()) {
        return fs::absolute(explicitRoot).lexically_normal().string();
    }

    try {
        fs::path current = fs::current_path();
        while (current.has_parent_path()) {
            if (fs::exists(current / "GEMINI.md") || fs::exists(current / ".git")) {
                return fs::absolute(current).lexically_normal().string();
            }
            current = current.parent_path();
        }

        fs::path exePath = fs::canonical("/proc/self/exe");
        return exePath.parent_path().parent_path().lexically_normal().string();
    } catch (...) {
        return fs::current_path().lexically_normal().string();
    }
}

std::string FileHandler::getAgentWorkspacePath(const std::string& filename) const {
    std::string workspaceOverride = getEnvOrEmpty("THOTH_WORKSPACE_PATH");
    fs::path workspace = workspaceOverride.empty()
        ? fs::path(getProjectRoot()) / "agent_workspace"
        : fs::path(workspaceOverride);

    fs::create_directories(workspace);

    if (!filename.empty()) {
        workspace /= filename;
    }

    return workspace.string();
}

std::string FileHandler::getRagPath(const std::string& filename) const {
    fs::path ragFolder = fs::path(getAgentWorkspacePath()) / "rag";

    // Ensure folder exists
    std::filesystem::create_directories(ragFolder);

    if (!filename.empty()) {
        ragFolder /= filename;
    }

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

std::string FileHandler::getConfigPath() const {
    std::string explicitPath = getEnvOrEmpty("THOTH_CONFIG_PATH");
    if (!explicitPath.empty()) {
        return fs::absolute(explicitPath).lexically_normal().string();
    }

    const fs::path cwd = fs::current_path();
    const fs::path root = fs::path(getProjectRoot());
    const std::vector<fs::path> candidates = {
        cwd / "config.json",
        root / "config.json",
        root / "external" / "basic_agent" / "config.json"
    };

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.lexically_normal().string();
        }
    }

    return (root / "config.json").lexically_normal().string();
}

std::string FileHandler::getEnvPath() const {
    std::string explicitPath = getEnvOrEmpty("THOTH_ENV_PATH");
    if (!explicitPath.empty()) {
        return fs::absolute(explicitPath).lexically_normal().string();
    }

    const fs::path cwd = fs::current_path();
    const fs::path root = fs::path(getProjectRoot());
    const std::vector<fs::path> candidates = {
        cwd / ".env",
        root / ".env",
        root / "external" / "basic_agent" / ".env"
    };

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.lexically_normal().string();
        }
    }

    return (root / ".env").lexically_normal().string();
}

