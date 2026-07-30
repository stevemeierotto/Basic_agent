/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-B revision artifact storage (Attachment Lifecycle Protocol)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "revision_storage.h"

#include "alp_storage_paths.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace Thoth {
namespace RevisionStorage {

bool ensureRevisionDir(const std::string& document_id, const std::string& revision_id) {
    if (document_id.empty() || revision_id.empty()) {
        return false;
    }
    std::error_code ec;
    fs::create_directories(AlpStoragePaths::revisionDirFor(document_id, revision_id), ec);
    return !ec;
}

bool writeManifest(const std::string& document_id,
                   const std::string& revision_id,
                   const nlohmann::json& manifest) {
    if (!ensureRevisionDir(document_id, revision_id)) {
        return false;
    }
    const std::string dir = AlpStoragePaths::revisionDirFor(document_id, revision_id);
    const std::string path = (fs::path(dir) / "manifest.json").string();
    const std::string temp_path = path + ".tmp." + std::to_string(getpid());
    try {
        {
            std::ofstream out(temp_path, std::ios::trunc);
            if (!out) {
                return false;
            }
            out << manifest.dump(2);
            if (!out) {
                std::error_code ec;
                fs::remove(temp_path, ec);
                return false;
            }
        }
        std::error_code ec;
        fs::rename(temp_path, path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
            return false;
        }
        return true;
    } catch (...) {
        std::error_code ec;
        fs::remove(temp_path, ec);
        return false;
    }
}

} // namespace RevisionStorage
} // namespace Thoth
