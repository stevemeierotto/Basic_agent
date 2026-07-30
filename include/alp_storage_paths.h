/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP1 physical storage namespaces (Attachment Lifecycle Protocol)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_STORAGE_PATHS_H
#define THOTH_ALP_STORAGE_PATHS_H

#include "file_handler.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace Thoth {
namespace AlpStoragePaths {

inline constexpr const char* kSeedSubdir = "seed";
inline constexpr const char* kAttachmentsSubdir = "attachments";
inline constexpr const char* kRevisionsSubdir = "revisions";
inline constexpr const char* kMigrationArchiveSubdir = "migration_archive";

inline std::string ragRoot() {
    FileHandler fh;
    return fh.getRagDirectory();
}

inline std::string seedCorpusDir() {
    return (fs::path(ragRoot()) / kSeedSubdir).lexically_normal().string();
}

inline std::string operatorAttachmentsDir() {
    return (fs::path(ragRoot()) / kAttachmentsSubdir).lexically_normal().string();
}

inline std::string revisionStorageRoot() {
    return (fs::path(ragRoot()) / kRevisionsSubdir).lexically_normal().string();
}

inline std::string revisionDirFor(const std::string& document_id,
                                  const std::string& revision_id) {
    return (fs::path(revisionStorageRoot()) / document_id / revision_id)
        .lexically_normal()
        .string();
}

inline std::string operatorAttachmentPath(const std::string& canonical_name) {
    return (fs::path(operatorAttachmentsDir()) / canonical_name)
        .lexically_normal()
        .string();
}

/** Create ALP namespace directories if missing (empty is OK for ALP-A). */
inline void ensureNamespaces() {
    std::error_code ec;
    fs::create_directories(seedCorpusDir(), ec);
    fs::create_directories(operatorAttachmentsDir(), ec);
    fs::create_directories(revisionStorageRoot(), ec);
}

} // namespace AlpStoragePaths
} // namespace Thoth

#endif // THOTH_ALP_STORAGE_PATHS_H
