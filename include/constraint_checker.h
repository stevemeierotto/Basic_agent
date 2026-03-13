/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ConstraintChecker for enforcing safety and policy
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CONSTRAINT_CHECKER_H
#define THOTH_CONSTRAINT_CHECKER_H

#include <string>
#include <vector>
#include <json.hpp>

namespace Thoth {

/**
 * @struct ConstraintResult
 * @brief The result of a constraint check.
 */
struct ConstraintResult {
    bool allowed = true;
    std::string reason;
};

/**
 * @class ConstraintChecker
 * @brief Enforces global safety and policy constraints on agent actions.
 */
class ConstraintChecker {
public:
    ConstraintChecker();

    /**
     * @brief Checks if a proposed action is allowed.
     * @param action_type The type of action (e.g., "tool_call", "file_modify", "network_request").
     * @param payload JSON containing action details.
     */
    ConstraintResult check_action(const std::string& action_type, const nlohmann::json& payload) const;

private:
    std::vector<std::string> blocked_paths_;
    std::vector<std::string> blocked_domains_;
    int max_tool_calls_per_goal_ = 20;
};

} // namespace Thoth

#endif // THOTH_CONSTRAINT_CHECKER_H
