/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-D1 migration apply (brownfield gate)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ALP_MIGRATION_APPLY_H
#define THOTH_ALP_MIGRATION_APPLY_H

#include "json.hpp"

#include <string>
#include <vector>

namespace Thoth {

struct AlpMigrationApplyResult {
    bool success = false;
    std::string run_id;
    std::string error;
    std::vector<std::string> warnings;
    nlohmann::json migration_log = nlohmann::json::object();
};

class AlpMigrationApply {
public:
    explicit AlpMigrationApply(std::string workspace_root);

    /** Validate approved report + optional resolution file; no mutations. */
    bool validateReport(const nlohmann::json& approved_report,
                        const nlohmann::json* resolution_file,
                        std::string& error_out) const;

    /** Execute apply phases; rolls back M0 on failure at D1-2+. */
    AlpMigrationApplyResult apply(const nlohmann::json& approved_report,
                                  const std::string& run_id,
                                  const nlohmann::json* resolution_file);

    /** Restore workspace from M0 snapshot for run_id. */
    bool rollback(const std::string& run_id, std::string& error_out);

private:
    std::string workspace_root_;
    std::string rag_root_;
    std::string index_path_;
    std::string registry_path_;
    std::string sessions_path_;
    std::string document_registry_path_;
    std::string legacy_map_path_;
    std::string migration_state_path_;
    std::string snapshots_root_;

    bool writeMigrationState(const std::string& run_id,
                             const std::string& phase,
                             const std::string& status,
                             const std::string& report_hash) const;

    bool createM0Snapshot(const std::string& run_id,
                          const std::string& report_hash,
                          std::string& error_out) const;

    bool rollbackM0(const std::string& run_id, std::string& error_out) const;
};

} // namespace Thoth

#endif // THOTH_ALP_MIGRATION_APPLY_H
