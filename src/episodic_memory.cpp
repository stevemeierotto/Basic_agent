/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EpisodicMemory implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/episodic_memory.h"
#include <algorithm>
#include <sstream>

namespace Thoth {

namespace {

void appendSection(std::ostringstream& out, const char* title, const std::vector<std::string>& items) {
    if (items.empty()) {
        return;
    }
    out << title << ":\n";
    for (const auto& item : items) {
        out << "- " << item << "\n";
    }
}

nlohmann::json stringArrayToJson(const std::vector<std::string>& values) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& v : values) {
        arr.push_back(v);
    }
    return arr;
}

std::vector<std::string> jsonToStringArray(const nlohmann::json& j, const char* key) {
    std::vector<std::string> out;
    if (!j.contains(key) || !j[key].is_array()) {
        return out;
    }
    for (const auto& item : j[key]) {
        if (item.is_string()) {
            out.push_back(item.get<std::string>());
        }
    }
    return out;
}

} // namespace

nlohmann::json EpisodicMemory::toJson() const {
    nlohmann::json j;
    j["goals"] = stringArrayToJson(goals);
    j["plans_attempted"] = stringArrayToJson(plans_attempted);
    j["decisions"] = stringArrayToJson(decisions);
    j["failures"] = stringArrayToJson(failures);
    j["tool_results"] = stringArrayToJson(tool_results);
    j["open_tasks"] = stringArrayToJson(open_tasks);
    j["facts_learned"] = stringArrayToJson(facts_learned);
    j["user_preferences"] = stringArrayToJson(user_preferences);
    j["outstanding_questions"] = stringArrayToJson(outstanding_questions);
    j["importance"] = importance;
    j["novelty"] = novelty;
    j["confidence"] = confidence;
    return j;
}

EpisodicMemory EpisodicMemory::fromJson(const nlohmann::json& j) {
    EpisodicMemory memory;
    memory.goals = jsonToStringArray(j, "goals");
    memory.plans_attempted = jsonToStringArray(j, "plans_attempted");
    memory.decisions = jsonToStringArray(j, "decisions");
    memory.failures = jsonToStringArray(j, "failures");
    memory.tool_results = jsonToStringArray(j, "tool_results");
    memory.open_tasks = jsonToStringArray(j, "open_tasks");
    memory.facts_learned = jsonToStringArray(j, "facts_learned");
    memory.user_preferences = jsonToStringArray(j, "user_preferences");
    memory.outstanding_questions = jsonToStringArray(j, "outstanding_questions");
    if (j.contains("importance")) {
        memory.importance = j["importance"].get<float>();
    }
    if (j.contains("novelty")) {
        memory.novelty = j["novelty"].get<float>();
    }
    if (j.contains("confidence")) {
        memory.confidence = j["confidence"].get<float>();
    }
    return memory;
}

std::string EpisodicMemory::serialize() const {
    return toJson().dump();
}

EpisodicMemory EpisodicMemory::deserialize(const std::string& jsonText) {
    if (jsonText.empty()) {
        return {};
    }
    return fromJson(nlohmann::json::parse(jsonText));
}

std::string EpisodicMemory::toCanonicalEmbedText() const {
    std::ostringstream out;
    appendSection(out, "Goals", goals);
    appendSection(out, "Plans Attempted", plans_attempted);
    appendSection(out, "Decisions", decisions);
    appendSection(out, "Failures", failures);
    appendSection(out, "Tool Results", tool_results);
    appendSection(out, "Open Tasks", open_tasks);
    appendSection(out, "Facts Learned", facts_learned);
    appendSection(out, "User Preferences", user_preferences);
    appendSection(out, "Outstanding Questions", outstanding_questions);
    return out.str();
}

std::string renderEpisodicMemory(const EpisodicMemory& memory) {
    std::ostringstream out;
    auto renderList = [&](const char* title, const std::vector<std::string>& items) {
        if (items.empty()) {
            return;
        }
        out << title << ":\n";
        for (const auto& item : items) {
            out << "  • " << item << "\n";
        }
        out << "\n";
    };

    renderList("Goals", memory.goals);
    renderList("Plans attempted", memory.plans_attempted);
    renderList("Decisions", memory.decisions);
    renderList("Failures", memory.failures);
    renderList("Tool results", memory.tool_results);
    renderList("Open tasks", memory.open_tasks);
    renderList("Facts learned", memory.facts_learned);
    renderList("User preferences", memory.user_preferences);
    renderList("Outstanding questions", memory.outstanding_questions);

    std::string text = out.str();
    if (text.empty()) {
        return "No episodic details extracted.";
    }
    return text;
}

float scoreEpisodicImportance(const EpisodicMemory& memory) {
    float score = 0.2f;
    if (!memory.decisions.empty()) {
        score += 0.3f;
    }
    if (!memory.tool_results.empty()) {
        score += 0.2f;
    }
    if (!memory.user_preferences.empty()) {
        score += 0.2f;
    }
    if (!memory.goals.empty() && memory.open_tasks.empty() && memory.failures.empty()) {
        score += 0.3f;
    }
    return std::min(1.0f, score);
}

} // namespace Thoth
