#pragma once
#include <string>

class FileHandler {
public:
    FileHandler() = default;

    // Returns full path to agent_workspace with optional filename
    std::string getAgentWorkspacePath(const std::string& filename = "") const;

    // Convenience for memory file
    std::string getMemoryPath() const;

    // Returns full path to agent_workspace/rag with optional filename
    std::string getRagPath(const std::string& filename = "") const;

    // Returns just the rag directory path (no filename)
    std::string getRagDirectory() const;
};

