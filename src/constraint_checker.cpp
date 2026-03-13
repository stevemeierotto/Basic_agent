/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ConstraintChecker implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/constraint_checker.h"
#include <algorithm>

namespace Thoth {

ConstraintChecker::ConstraintChecker() {
    blocked_paths_ = {"/etc/", "/var/", "/usr/", "/bin/", "/sbin/", "/root/", "/home/steve/.ssh/"};
    blocked_domains_ = {"internal.google.com", "secret.com"};
}

ConstraintResult ConstraintChecker::check_action(const std::string& action_type, const nlohmann::json& payload) const {
    ConstraintResult result;

    if (action_type == "file_modify" || action_type == "file_read") {
        std::string path = payload.value("file_path", "");
        for (const auto& blocked : blocked_paths_) {
            if (path.find(blocked) == 0) {
                result.allowed = false;
                result.reason = "Path is restricted: " + path;
                return result;
            }
        }
    }

    if (action_type == "network_request") {
        std::string url = payload.value("url", "");
        for (const auto& blocked : blocked_domains_) {
            if (url.find(blocked) != std::string::npos) {
                result.allowed = false;
                result.reason = "Domain is restricted: " + url;
                return result;
            }
        }
    }

    if (action_type == "tool_call") {
        int call_count = payload.value("current_call_count", 0);
        if (call_count >= max_tool_calls_per_goal_) {
            result.allowed = false;
            result.reason = "Maximum tool call limit reached (" + std::to_string(max_tool_calls_per_goal_) + ")";
            return result;
        }
    }

    return result;
}

} // namespace Thoth
