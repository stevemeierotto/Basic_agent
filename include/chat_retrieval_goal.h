/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — CSG-A session goal resolution for chat retrieval
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_RETRIEVAL_GOAL_H
#define THOTH_CHAT_RETRIEVAL_GOAL_H

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class EmbeddingEngine;

namespace Thoth {

struct ChatRetrievalGoal {
    std::vector<float> embedding;
    /** "executive" | "session" | "none" */
    std::string source = "none";
    std::string error;
};

/** Normalize session goal text before hash/embed (aligns with GUI TrimGoalForDisplay + storage clean). */
std::string normalizeSessionGoalText(const std::string& goal);

/** Session-scoped embed cache — no cross-session reuse (CSG-A §4). */
class SessionGoalEmbedCache {
public:
    static constexpr std::size_t kMaxGoalsPerSession = 4;

    std::optional<std::vector<float>> lookup(const std::string& session_id,
                                             const std::string& normalized_goal) const;

    void store(const std::string& session_id,
               const std::string& normalized_goal,
               const std::vector<float>& embedding,
               std::int64_t cached_at_ms);

    std::size_t entryCountForSession(const std::string& session_id) const;

private:
    struct Entry {
        std::string goal_hash;
        std::vector<float> embedding;
        std::int64_t cached_at_ms = 0;
    };

    std::unordered_map<std::string, std::vector<Entry>> buckets_;
};

ChatRetrievalGoal resolveChatRetrievalGoal(
    class ExecutiveController* controller,
    const std::string& session_id,
    const std::optional<std::string>& active_goal,
    SessionGoalEmbedCache& cache,
    EmbeddingEngine* embed_engine);

} // namespace Thoth

#endif // THOTH_CHAT_RETRIEVAL_GOAL_H
