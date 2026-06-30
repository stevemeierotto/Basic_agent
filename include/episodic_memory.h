/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — structured episodic memory (warm tier source of truth)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODIC_MEMORY_H
#define THOTH_EPISODIC_MEMORY_H

#include <json.hpp>
#include <string>
#include <vector>

namespace Thoth {

struct EpisodicMemory {
    std::vector<std::string> goals;
    std::vector<std::string> plans_attempted;
    std::vector<std::string> decisions;
    std::vector<std::string> failures;
    std::vector<std::string> tool_results;
    std::vector<std::string> open_tasks;
    std::vector<std::string> facts_learned;
    std::vector<std::string> user_preferences;
    std::vector<std::string> outstanding_questions;

    float importance = 0.5f;
    float novelty = 0.5f;
    float confidence = 1.0f;

    /** Serialize for SQLite persistence (not the source of truth). */
    std::string serialize() const;
    static EpisodicMemory deserialize(const std::string& jsonText);

    /** Deterministic text for embedding — independent of prose renderer. */
    std::string toCanonicalEmbedText() const;

    nlohmann::json toJson() const;
    static EpisodicMemory fromJson(const nlohmann::json& j);
};

/** Prose for humans / prompt injection. */
std::string renderEpisodicMemory(const EpisodicMemory& memory);

/** Deterministic importance from structured fields (M1). */
float scoreEpisodicImportance(const EpisodicMemory& memory);

} // namespace Thoth

#endif
