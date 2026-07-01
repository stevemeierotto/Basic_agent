/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E1 git metadata helper (Checkpoint B)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_GIT_METADATA_H
#define THOTH_GIT_METADATA_H

#include <string>

namespace Thoth {

struct GitRepoMetadata {
    std::string sha;
};

class GitMetadata {
public:
    /** Short SHA for repo at repoPath; returns "unknown" on failure. */
    static GitRepoMetadata readShortSha(const std::string& repoPath);

    /** Thoth root + external/basic_agent submodule SHAs. */
    static void readThothProject(const std::string& projectRoot,
                                 std::string& thothSha,
                                 std::string& basicAgentSha);
};

} // namespace Thoth

#endif // THOTH_GIT_METADATA_H
