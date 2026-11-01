/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — one-shot EGAR-LX workspace directory index
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_WORKSPACE_INDEX_RUN_H
#define THOTH_WORKSPACE_INDEX_RUN_H

#include <cstddef>
#include <string>

class IndexManager;

namespace Thoth {

/** Result of a saved workspace index. added is the change in chunk count. */
struct WorkspaceIndexResult {
    std::size_t chunks = 0;
    long long added = 0;
    bool no_evidence_added = false;
    std::string index_path;
};

/**
 * Resolve PATH and reject anything outside the configured workspace.
 * Symlink targets are resolved. A directory symlink is not followed when it
 * appears inside the tree. This does not index or save.
 */
bool validateWorkspaceIndexDirectory(const std::string& path,
                                    std::string& error,
                                    std::string& resolved_directory);

/**
 * Validate PATH, run IndexManager::indexProject, then saveIndex().
 * The caller loads any existing index first. This does not execute a goal
 * and does not enable fixed-evidence containment.
 */
bool indexWorkspaceDirectory(IndexManager& index,
                             const std::string& path,
                             std::string& error,
                             WorkspaceIndexResult& result);

/** One stdout line. no_evidence_added is present only when added is 0. */
std::string formatWorkspaceIndexStatus(const WorkspaceIndexResult& result);

} // namespace Thoth

#endif
