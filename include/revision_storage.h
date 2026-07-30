/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-B revision artifact storage (Attachment Lifecycle Protocol)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_REVISION_STORAGE_H
#define THOTH_REVISION_STORAGE_H

#include "json.hpp"

#include <string>

namespace Thoth {
namespace RevisionStorage {

/** Ensure rag/revisions/{document_id}/{revision_id}/ exists. */
bool ensureRevisionDir(const std::string& document_id, const std::string& revision_id);

/** Write manifest.json for a revision candidate or terminal state. */
bool writeManifest(const std::string& document_id,
                   const std::string& revision_id,
                   const nlohmann::json& manifest);

} // namespace RevisionStorage
} // namespace Thoth

#endif // THOTH_REVISION_STORAGE_H
