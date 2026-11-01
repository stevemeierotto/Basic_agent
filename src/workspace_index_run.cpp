/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — one-shot EGAR-LX workspace directory index
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/workspace_index_run.h"

#include "../include/file_handler.h"
#include "../include/index_manager.h"

#include <filesystem>
#include <sstream>

namespace Thoth {
namespace {

namespace fs = std::filesystem;

bool sameOrInside(const fs::path& root, const fs::path& candidate) {
    const fs::path norm_root = root.lexically_normal();
    const fs::path norm = candidate.lexically_normal();
    auto root_it = norm_root.begin();
    auto path_it = norm.begin();
    for (; root_it != norm_root.end(); ++root_it, ++path_it) {
        if (path_it == norm.end() || *path_it != *root_it) {
            return false;
        }
    }
    return true;
}

fs::path resolvePath(const fs::path& path, std::error_code& ec) {
    return fs::weakly_canonical(path, ec);
}

} // namespace

bool validateWorkspaceIndexDirectory(const std::string& path,
                                    std::string& error,
                                    std::string& resolved_directory) {
    resolved_directory.clear();
    if (path.empty()) {
        error = "index directory path is empty";
        return false;
    }

    FileHandler file_handler;
    std::error_code ec;
    const fs::path root = resolvePath(file_handler.getAgentWorkspacePath(), ec);
    if (ec) {
        error = "workspace path could not be resolved";
        return false;
    }

    const fs::path requested(path);
    if (!fs::exists(requested, ec) || ec) {
        error = "index directory not found: " + path;
        return false;
    }
    const fs::path resolved = resolvePath(requested, ec);
    if (ec) {
        error = "index directory could not be resolved: " + path;
        return false;
    }
    if (!fs::is_directory(resolved, ec) || ec) {
        error = "index path is not a directory: " + path;
        return false;
    }
    if (!sameOrInside(root, resolved)) {
        error = "index directory is outside the workspace: " + resolved.string();
        return false;
    }

    try {
        fs::recursive_directory_iterator it(
            resolved, fs::directory_options::skip_permission_denied);
        const fs::recursive_directory_iterator end;
        for (; it != end; ++it) {
            const fs::path entry = it->path();
            std::error_code entry_ec;
            const bool is_link = it->is_symlink(entry_ec);
            if (entry_ec) {
                error = "index path could not be resolved: " + entry.string();
                return false;
            }
            const fs::path target = resolvePath(entry, entry_ec);
            if (entry_ec) {
                error = "index path could not be resolved: " + entry.string();
                return false;
            }
            if (!sameOrInside(root, target)) {
                error = "index path resolves outside the workspace: " + entry.string();
                return false;
            }
            if (is_link) {
                const bool target_dir = fs::is_directory(target, entry_ec);
                if (!entry_ec && target_dir) {
                    it.disable_recursion_pending();
                }
            }
        }
    } catch (const std::exception& ex) {
        error = std::string("index directory could not be read: ") + ex.what();
        return false;
    }

    resolved_directory = resolved.string();
    error.clear();
    return true;
}

bool indexWorkspaceDirectory(IndexManager& index,
                             const std::string& path,
                             std::string& error,
                             WorkspaceIndexResult& result) {
    result = WorkspaceIndexResult{};
    std::string resolved;
    if (!validateWorkspaceIndexDirectory(path, error, resolved)) {
        return false;
    }

    const std::size_t before = index.chunkCount();
    index.indexProject(resolved);
    if (!index.saveIndex()) {
        error = "index could not be saved";
        return false;
    }

    FileHandler file_handler;
    result.chunks = index.chunkCount();
    result.added = static_cast<long long>(result.chunks) - static_cast<long long>(before);
    result.no_evidence_added = result.added == 0;
    result.index_path = file_handler.getRagPath("rag_index.bin");
    error.clear();
    return true;
}

std::string formatWorkspaceIndexStatus(const WorkspaceIndexResult& result) {
    std::ostringstream out;
    out << "WORKSPACE_INDEX saved chunks=" << result.chunks
        << " added=" << result.added;
    if (result.no_evidence_added) {
        out << " no_evidence_added";
    }
    if (!result.index_path.empty()) {
        out << " path=" << result.index_path;
    }
    return out.str();
}

} // namespace Thoth
