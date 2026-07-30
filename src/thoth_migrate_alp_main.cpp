/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP migration CLI (D0 dry-run + D1 apply/rollback)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "alp_migration_analyzer.h"
#include "alp_migration_apply.h"
#include "file_handler.h"

#include <fstream>
#include <iostream>
#include <string>

namespace {

void printUsage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " --dry-run [--workspace PATH] [--output FILE.json]\n"
              << "  " << argv0
              << " --apply --report FILE.json [--workspace PATH] [--run-id ID]\n"
              << "       [--resolution-file FILE.json]\n"
              << "  " << argv0 << " --rollback --run-id ID [--workspace PATH]\n"
              << "\n"
              << "  --dry-run   Read-only legacy corpus analysis (ALP-D0).\n"
              << "  --apply     Execute approved D0 report (ALP-D1).\n"
              << "  --rollback  Restore workspace from M0 snapshot for run-id.\n";
}

nlohmann::json loadJsonFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open: " + path);
    }
    nlohmann::json body;
    in >> body;
    return body;
}

} // namespace

int main(int argc, char* argv[]) {
    bool dry_run = false;
    bool apply = false;
    bool rollback = false;
    std::string workspace;
    std::string output_path;
    std::string report_path;
    std::string resolution_path;
    std::string run_id;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--apply") {
            apply = true;
        } else if (arg == "--rollback") {
            rollback = true;
        } else if (arg == "--workspace" && i + 1 < argc) {
            workspace = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        } else if (arg == "--resolution-file" && i + 1 < argc) {
            resolution_path = argv[++i];
        } else if (arg == "--run-id" && i + 1 < argc) {
            run_id = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    const int mode_count = static_cast<int>(dry_run) + static_cast<int>(apply)
        + static_cast<int>(rollback);
    if (mode_count != 1) {
        printUsage(argv[0]);
        return 1;
    }

    if (workspace.empty()) {
        FileHandler fh;
        workspace = fh.getAgentWorkspacePath();
    }

    try {
        if (dry_run) {
            Thoth::AlpMigrationAnalyzer analyzer(workspace);
            const nlohmann::json report = analyzer.runDryRun();
            const std::string body = report.dump(2);
            if (output_path.empty()) {
                std::cout << body << '\n';
            } else {
                std::ofstream out(output_path, std::ios::trunc);
                if (!out) {
                    std::cerr << "Failed to write output: " << output_path << "\n";
                    return 1;
                }
                out << body;
            }
            return 0;
        }

        if (rollback) {
            if (run_id.empty()) {
                std::cerr << "--rollback requires --run-id\n";
                return 1;
            }
            Thoth::AlpMigrationApply apply_engine(workspace);
            std::string err;
            if (!apply_engine.rollback(run_id, err)) {
                std::cerr << "Rollback failed: " << err << "\n";
                return 1;
            }
            std::cout << "Rollback complete for run_id=" << run_id << "\n";
            return 0;
        }

        if (report_path.empty()) {
            std::cerr << "--apply requires --report\n";
            return 1;
        }

        const nlohmann::json report = loadJsonFile(report_path);
        const nlohmann::json* resolution = nullptr;
        nlohmann::json resolution_body;
        if (!resolution_path.empty()) {
            resolution_body = loadJsonFile(resolution_path);
            resolution = &resolution_body;
        }

        if (run_id.empty()) {
            if (report.contains("migration_run_id")
                && report["migration_run_id"].is_string()) {
                run_id = report["migration_run_id"].get<std::string>();
            } else if (report.contains("report_hash")
                       && report["report_hash"].is_string()) {
                run_id = "alp-d1-" + report["report_hash"].get<std::string>().substr(0, 12);
            } else {
                run_id = "alp-d1-manual";
            }
        }

        Thoth::AlpMigrationApply apply_engine(workspace);
        const Thoth::AlpMigrationApplyResult result =
            apply_engine.apply(report, run_id, resolution);
        if (!result.success) {
            std::cerr << "ALP-D1 apply failed: " << result.error << "\n";
            return 1;
        }

        std::cout << result.migration_log.dump(2) << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Migration command failed: " << ex.what() << "\n";
        return 1;
    }
}
