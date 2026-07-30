/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-D0 migration dry-run analyzer (read-only)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_MIGRATION_ANALYZER_H
#define THOTH_ALP_MIGRATION_ANALYZER_H

#include "json.hpp"

#include <string>

namespace Thoth {

class AlpMigrationAnalyzer {
public:
    static constexpr int kReportSchemaVersion = 1;

    explicit AlpMigrationAnalyzer(std::string workspace_root);

    /** Build full dry-run report JSON (includes volatile fields + report_hash). */
    nlohmann::json runDryRun();

    /** Canonical semantic payload for hashing (excludes volatile top-level keys). */
    static nlohmann::json canonicalPayload(const nlohmann::json& report);
    static std::string computeReportHash(const nlohmann::json& report);

private:
    std::string workspace_root_;
    std::string rag_root_;
    std::string index_path_;
    std::string registry_path_;
    std::string sessions_path_;
};

} // namespace Thoth

#endif // THOTH_ALP_MIGRATION_ANALYZER_H
