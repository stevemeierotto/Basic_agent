/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — project_analyze tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/project_analyze_tool.h"
#include "../include/file_handler.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <iostream>

namespace fs = std::filesystem;

nlohmann::json ProjectAnalyzeTool::input_schema() const {
    return {
        {"type", "object"},
        {"properties", {
            {"root_path", {{"type", "string"}, {"description", "The root directory to analyze (relative to project root). Defaults to ./ if not provided."}}}
        }},
        {"additionalProperties", false}
    };
}

bool ProjectAnalyzeTool::requires_confirmation() const {
    return false;
}

std::vector<std::string> ProjectAnalyzeTool::extractClasses(const std::string& content) const {
    std::vector<std::string> classes;
    std::regex class_regex(R"(class\s+([A-Za-z0-9_]+))");
    auto it = std::sregex_iterator(content.begin(), content.end(), class_regex);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        classes.push_back((*it)[1].str());
    }
    return classes;
}

std::vector<std::string> ProjectAnalyzeTool::extractIncludes(const std::string& content) const {
    std::vector<std::string> includes;
    std::regex include_regex(R"(#include\s+["<]([^">]+)[">])");
    auto it = std::sregex_iterator(content.begin(), content.end(), include_regex);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        includes.push_back((*it)[1].str());
    }
    return includes;
}

nlohmann::json ProjectAnalyzeTool::execute(const nlohmann::json& input) const {
    FileHandler fh;
    std::string project_root = fh.getProjectRoot();
    std::string relative_path = input.value("root_path", "./");
    
    fs::path target_path = fs::path(project_root) / relative_path;
    target_path = target_path.lexically_normal();

    if (!fs::exists(target_path) || !fs::is_directory(target_path)) {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "Root path does not exist or is not a directory: " + relative_path}
        };
    }

    nlohmann::json files_json = nlohmann::json::array();
    nlohmann::json classes_json = nlohmann::json::object();
    nlohmann::json dependencies_json = nlohmann::json::object();

    try {
        for (const auto& entry : fs::recursive_directory_iterator(target_path)) {
            if (!entry.is_regular_file()) continue;

            std::string path = fs::relative(entry.path(), target_path).string();
            std::string ext = entry.path().extension().string();
            size_t size = entry.file_size();

            files_json.push_back({
                {"path", path},
                {"size", size}
            });

            if (ext == ".h" || ext == ".hpp" || ext == ".cpp" || ext == ".cc") {
                std::ifstream in(entry.path());
                if (in.is_open()) {
                    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                    
                    if (ext == ".h" || ext == ".hpp") {
                        auto classes = extractClasses(content);
                        if (!classes.empty()) {
                            classes_json[path] = classes;
                        }
                    }

                    auto includes = extractIncludes(content);
                    if (!includes.empty()) {
                        dependencies_json[path] = includes;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "Exception during analysis: " + std::string(e.what())}
        };
    }

    return {
        {"status", "success"},
        {"data", {
            {"files", files_json},
            {"classes", classes_json},
            {"dependencies", dependencies_json}
        }},
        {"error_message", nullptr}
    };
}
