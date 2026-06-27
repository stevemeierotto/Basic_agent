/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Goal text normalization for planner and UI display
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_GOAL_TEXT_UTILS_H
#define THOTH_GOAL_TEXT_UTILS_H

#include <string>
#include <utility>

namespace Thoth {

/** Splits user goal from ExecutiveController plan-reuse injection block. */
inline std::pair<std::string, std::string> splitPlanReuseInjection(const std::string& goal) {
    static const char* kMarker = "[RELEVANT PAST APPROACHES";
    const std::size_t pos = goal.find(kMarker);
    if (pos == std::string::npos) {
        return {goal, ""};
    }

    std::string userGoal = goal.substr(0, pos);
    while (!userGoal.empty() && (userGoal.back() == '\n' || userGoal.back() == '\r' || userGoal.back() == ' ')) {
        userGoal.pop_back();
    }
    return {userGoal, goal.substr(pos)};
}

} // namespace Thoth

#endif // THOTH_GOAL_TEXT_UTILS_H
