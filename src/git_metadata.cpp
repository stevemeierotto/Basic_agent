/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 git metadata helper (Checkpoint B)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/git_metadata.h"

#include <array>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace Thoth {

namespace {

std::string shellSingleQuote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += '\'';
    return quoted;
}

std::string runGitShortSha(const std::string& repoPath) {
    if (repoPath.empty()) {
        return "unknown";
    }
    std::error_code ec;
    if (!fs::exists(repoPath, ec) || !fs::is_directory(repoPath, ec)) {
        return "unknown";
    }

    const std::string command =
        "git -C " + shellSingleQuote(repoPath) + " rev-parse --short HEAD 2>/dev/null";

    std::array<char, 128> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "unknown";
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output.append(buffer.data());
    }
    pclose(pipe);

    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output.empty() ? "unknown" : output;
}

} // namespace

GitRepoMetadata GitMetadata::readShortSha(const std::string& repoPath) {
    return GitRepoMetadata{runGitShortSha(repoPath)};
}

void GitMetadata::readThothProject(const std::string& projectRoot,
                                   std::string& thothSha,
                                   std::string& basicAgentSha) {
    thothSha = runGitShortSha(projectRoot);
    basicAgentSha = runGitShortSha((fs::path(projectRoot) / "external" / "basic_agent").string());
}

} // namespace Thoth
