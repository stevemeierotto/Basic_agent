/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — code_modify tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/code_modify_tool.h"
#include "../include/file_handler.h"
#include "../include/config.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <array>

namespace fs = std::filesystem;

namespace {

constexpr const char* kShellExecDenied =
    "Shell execution is disabled. Set allow_shell_exec to true in config.json to enable this operation.";

} // namespace

CodeModifyTool::CodeModifyTool(Config* config) : config_(config) {}

nlohmann::json CodeModifyTool::input_schema() const {
    return {
        {"type", "object"},
        {"properties", {
            {"operation", {{"type", "string"}, {"enum", {"read", "apply_diff", "build", "revert"}}}},
            {"file_path", {{"type", "string"}, {"description", "Relative path to the file from project root. Required for read, apply_diff, and revert."}}},
            {"unified_diff", {{"type", "string"}, {"description", "Required if operation is apply_diff."}}},
            {"auto_build", {{"type", "boolean"}, {"description", "If true, automatically run build after apply_diff. Defaults to false."}}},
            {"confirmed", {{"type", "boolean"}, {"description", "Must be true to proceed with risky operations."}}}
        }},
        {"required", {"operation"}},
        {"additionalProperties", false}
    };
}

bool CodeModifyTool::requires_confirmation() const {
    return true;
}

bool CodeModifyTool::isPathSafe(const std::string& relative_path, const std::string& project_root) const {
    if (relative_path.empty()) return true; // Empty path OK for 'build' operation
    
    // Reject absolute paths
    if (fs::path(relative_path).is_absolute()) return false;

    // Reject path traversal
    if (relative_path.find("..") != std::string::npos) return false;

    fs::path root = fs::path(project_root).lexically_normal();
    fs::path full = (root / relative_path).lexically_normal();

    // Ensure it's still under root
    auto root_str = root.string();
    auto full_str = full.string();
    
    return full_str.substr(0, root_str.size()) == root_str;
}

std::string CodeModifyTool::applyDiff(const std::string& original, const std::string& diff, std::string& error) const {
    // For v1.0 prototype, we'll return an error. In Phase 3.3/3.4 we focus on harness.
    error = "Unified diff application not fully implemented in v1.0 prototype. Harness is ready.";
    return "";
}

nlohmann::json CodeModifyTool::execute(const nlohmann::json& input) const {
    FileHandler fh;
    std::string project_root = fh.getProjectRoot();
    std::string operation = input.at("operation");
    std::string file_path = input.value("file_path", "");

    if (!isPathSafe(file_path, project_root)) {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "Path is outside project root or invalid: " + file_path}
        };
    }

    fs::path full_path = fs::path(project_root) / file_path;

    if (operation == "read") {
        if (file_path.empty()) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "file_path is required for read operation."}};
        }
        if (!fs::exists(full_path)) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", "File does not exist: " + file_path}
            };
        }

        std::ifstream in(full_path);
        if (!in.is_open()) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", "Failed to open file: " + file_path}
            };
        }

        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return {
            {"status", "success"},
            {"data", {{"content", content}}},
            {"error_message", nullptr}
        };
    } else if (operation == "build") {
        if (!config_ || !config_->allow_shell_exec) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", kShellExecDenied}
            };
        }
        // Use the build preset from CMakePresets.json
        std::string build_cmd = "cmake --build " + project_root + "/build/debug --preset build-debug";
        std::string output;
        std::array<char, 128> buffer;
        FILE* pipe = popen(build_cmd.c_str(), "r");
        if (!pipe) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "Failed to run build command."}};
        }
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            output += buffer.data();
        }
        int exit_code = pclose(pipe);
        bool success = (exit_code == 0);
        return {
            {"status", success ? "success" : "error"},
            {"data", {{"build_output", output}, {"exit_code", exit_code}}},
            {"error_message", success ? nullptr : "Build failed."}
        };
    } else if (operation == "revert") {
        if (file_path.empty()) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "file_path is required for revert operation."}};
        }
        fs::path backup_path = full_path.string() + ".bak";
        if (!fs::exists(backup_path)) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "No backup found for file: " + file_path}};
        }
        try {
            fs::copy_file(backup_path, full_path, fs::copy_options::overwrite_existing);
            return {
                {"status", "success"},
                {"data", {{"message", "File restored from backup."}}},
                {"error_message", nullptr}
            };
        } catch (const std::exception& e) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "Failed to revert file: " + std::string(e.what())}};
        }
    } else if (operation == "apply_diff") {
        if (file_path.empty()) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "file_path is required for apply_diff operation."}};
        }
        if (!input.contains("unified_diff")) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", "unified_diff is required for apply_diff operation."}
            };
        }

        // 1. Take backup
        try {
            fs::copy_file(full_path, full_path.string() + ".bak", fs::copy_options::overwrite_existing);
        } catch (...) {
            return {{"status", "error"}, {"data", nlohmann::json::object()}, {"error_message", "Failed to create backup before applying diff."}};
        }

        // Real implementation would apply patch here.
        // For now, return error as defined in applyDiff stub.
        std::string error;
        std::string result = applyDiff("", input["unified_diff"], error);
        
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", error}
        };
    }

    return {
        {"status", "error"},
        {"data", nlohmann::json::object()},
        {"error_message", "Unknown operation: " + operation}
    };
}
