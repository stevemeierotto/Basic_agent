/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Goal text normalization for planner and UI display
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_GOAL_TEXT_UTILS_H
#define THOTH_GOAL_TEXT_UTILS_H

#include "json.hpp"
#include "plan.h"
#include <optional>
#include <sstream>
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

/** Strip nested plan-reuse markers for persistence (prevents recursive pollution). */
inline std::string cleanGoalForStorage(const std::string& goal) {
    auto [clean, _] = splitPlanReuseInjection(goal);
    return clean;
}

inline bool isAsciiSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/** A line whose trimmed text is exactly "Problem:", not a longer line that contains it. */
inline bool isStandaloneProblemLine(const std::string& line) {
    std::size_t begin = 0;
    std::size_t end = line.size();
    while (begin < end && (line[begin] == ' ' || line[begin] == '\t')) {
        ++begin;
    }
    while (end > begin && (line[end - 1] == ' ' || line[end - 1] == '\t' || line[end - 1] == '\r')) {
        --end;
    }
    static const char kMarker[] = "Problem:";
    constexpr std::size_t kLen = sizeof(kMarker) - 1;
    return (end - begin) == kLen && line.compare(begin, kLen, kMarker) == 0;
}

/**
 * Text after the first standalone Problem: line.
 * Empty or whitespace-only sections are not a case excerpt.
 */
inline std::optional<std::string> standaloneProblemSection(const std::string& text) {
    std::size_t lineStart = 0;
    while (lineStart <= text.size()) {
        const std::size_t newline = text.find('\n', lineStart);
        const std::size_t lineEnd = newline == std::string::npos ? text.size() : newline;
        if (isStandaloneProblemLine(text.substr(lineStart, lineEnd - lineStart))) {
            const std::size_t contentStart = newline == std::string::npos ? text.size() : newline + 1;
            const std::string rest = text.substr(contentStart);
            std::size_t begin = 0;
            std::size_t end = rest.size();
            while (begin < end && isAsciiSpace(rest[begin])) {
                ++begin;
            }
            while (end > begin && isAsciiSpace(rest[end - 1])) {
                --end;
            }
            if (begin >= end) {
                return std::nullopt;
            }
            return rest.substr(begin, end - begin);
        }
        if (newline == std::string::npos) {
            break;
        }
        lineStart = newline + 1;
    }
    return std::nullopt;
}

/**
 * Planner trajectory block.
 * A nonempty standalone Problem: section is placed after the header and stored
 * success score so the existing front truncation keeps the case text.
 * Every other goal keeps the historical Goal-then-Success layout.
 */
inline std::string formatTrajectoryInjection(const std::string& goal, float success_score) {
    const std::string cleaned = cleanGoalForStorage(goal);
    std::ostringstream oss;
    oss << "[PAST EXPERIENCE — RELEVANT TRAJECTORY]\n";
    const std::optional<std::string> problem = standaloneProblemSection(cleaned);
    if (problem.has_value()) {
        oss << "  Success: " << success_score << "\n";
        oss << "- Case: " << *problem << "\n";
    } else {
        oss << "- Goal: " << cleaned << "\n";
        oss << "  Success: " << success_score << "\n";
    }
    return oss.str();
}

/** Truncate injection text with an explicit marker (never silent cut). */
inline std::string capInjectionText(const std::string& text, std::size_t maxChars) {
    if (maxChars == 0) {
        return "";
    }
    if (text.empty()) {
        return text;
    }
    if (text.size() <= maxChars) {
        return text;
    }
    return text.substr(0, maxChars) + "\n...(truncated)...";
}

inline const char* stepTypeLabel(StepType type) {
    switch (type) {
        case StepType::RETRIEVAL: return "RETRIEVAL";
        case StepType::LLM: return "LLM";
        case StepType::TOOL: return "TOOL";
        case StepType::NODE: return "NODE";
    }
    return "UNKNOWN";
}

inline std::string stepTypeLabelFromJson(const nlohmann::json& step) {
    if (step.contains("step_type") && step["step_type"].is_string()) {
        return step["step_type"].get<std::string>();
    }
    if (step.contains("type")) {
        if (step["type"].is_string()) {
            return step["type"].get<std::string>();
        }
        if (step["type"].is_number_integer()) {
            return stepTypeLabel(static_cast<StepType>(step["type"].get<int>()));
        }
    }
    return "UNKNOWN";
}

/** Step-type summary only — no tool payloads or full JSON. */
inline std::string sanitizePlanOutline(const std::string& outline) {
    try {
        const auto j = nlohmann::json::parse(outline);
        const nlohmann::json* steps = nullptr;
        if (j.contains("steps") && j["steps"].is_array()) {
            steps = &j["steps"];
        } else if (j.contains("plan") && j["plan"].is_array()) {
            steps = &j["plan"];
        }
        if (!steps) {
            return "(no steps)";
        }

        std::ostringstream oss;
        bool first = true;
        for (const auto& step : *steps) {
            if (!first) {
                oss << " → ";
            }
            first = false;
            const std::string type = stepTypeLabelFromJson(step);
            const std::string id = step.value("step_id", "?");
            oss << type << "(" << id << ")";
        }
        return oss.str();
    } catch (...) {
        return "(unparseable outline)";
    }
}

} // namespace Thoth

#endif // THOTH_GOAL_TEXT_UTILS_H
