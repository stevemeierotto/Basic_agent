#pragma once
#include <string>

class FileHandler {
public:
    FileHandler() = default;

    std::string getProjectRoot() const;

    // Returns full path to agent_workspace with optional filename
    std::string getAgentWorkspacePath(const std::string& filename = "") const;

    // Returns logs root, or logs root + filename (same contract as getAgentWorkspacePath)
    std::string getLogsPath(const std::string& filename = "") const;

    // Returns full path to agent_workspace/rag with optional filename
    std::string getRagPath(const std::string& filename = "") const;

    // Returns just the rag directory path (no filename)
    std::string getRagDirectory() const;

    // Config and env resolution with overrides
    std::string getConfigPath() const;
    std::string getEnvPath() const;
};

