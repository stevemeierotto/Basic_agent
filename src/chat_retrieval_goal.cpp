/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — CSG-A session goal resolution for chat retrieval
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "chat_retrieval_goal.h"

#include "alp_sha256.h"
#include "embedding_engine.h"
#include "executive_controller.h"
#include "goal_text_utils.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <sstream>

namespace Thoth {

namespace {

std::string goalHashKey(const std::string& normalized_goal) {
    const std::string hex = sha256Hex(normalized_goal);
    return hex.size() > 16 ? hex.substr(0, 16) : hex;
}

std::string collapseInternalWhitespace(std::string text) {
    std::ostringstream out;
    bool in_space = false;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!in_space) {
                out << ' ';
                in_space = true;
            }
        } else {
            out << ch;
            in_space = false;
        }
    }
    return out.str();
}

std::vector<float> embedStructuredGoal(EmbeddingEngine* embed_engine, const std::string& goal_text) {
    nlohmann::json structured_goal;
    structured_goal["schema_version"] = 2;
    structured_goal["type"] = "goal";
    structured_goal["content"] = goal_text;
    return embed_engine->embed(structured_goal.dump());
}

} // namespace

std::string normalizeSessionGoalText(const std::string& goal) {
    std::string cleaned = cleanGoalForStorage(goal);
    const auto first = cleaned.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    cleaned = cleaned.substr(first);
    const auto last = cleaned.find_last_not_of(" \t\r\n");
    if (last != std::string::npos) {
        cleaned = cleaned.substr(0, last + 1);
    }
    cleaned = collapseInternalWhitespace(cleaned);

    if (cleaned.size() >= 6 && cleaned.compare(0, 6, "/goal ") == 0) {
        cleaned = cleaned.substr(6);
    } else if (cleaned.size() >= 5 && cleaned.compare(0, 5, "Goal:") == 0) {
        cleaned = cleaned.substr(5);
        const auto lead = cleaned.find_first_not_of(" \t");
        if (lead != std::string::npos) {
            cleaned = cleaned.substr(lead);
        }
    }
    return cleaned;
}

std::optional<std::vector<float>> SessionGoalEmbedCache::lookup(
    const std::string& session_id,
    const std::string& normalized_goal) const {
    const auto bucket_it = buckets_.find(session_id);
    if (bucket_it == buckets_.end()) {
        return std::nullopt;
    }
    const std::string hash = goalHashKey(normalized_goal);
    for (const Entry& entry : bucket_it->second) {
        if (entry.goal_hash == hash) {
            return entry.embedding;
        }
    }
    return std::nullopt;
}

void SessionGoalEmbedCache::store(const std::string& session_id,
                                 const std::string& normalized_goal,
                                 const std::vector<float>& embedding,
                                 const std::int64_t cached_at_ms) {
    const std::string hash = goalHashKey(normalized_goal);
    std::vector<Entry>& bucket = buckets_[session_id];

    for (Entry& entry : bucket) {
        if (entry.goal_hash == hash) {
            entry.embedding = embedding;
            entry.cached_at_ms = cached_at_ms;
            return;
        }
    }

    bucket.push_back(Entry{hash, embedding, cached_at_ms});
    if (bucket.size() > kMaxGoalsPerSession) {
        auto oldest = bucket.begin();
        for (auto it = bucket.begin() + 1; it != bucket.end(); ++it) {
            if (it->cached_at_ms < oldest->cached_at_ms) {
                oldest = it;
            }
        }
        bucket.erase(oldest);
    }
}

std::size_t SessionGoalEmbedCache::entryCountForSession(const std::string& session_id) const {
    const auto it = buckets_.find(session_id);
    return it == buckets_.end() ? 0 : it->second.size();
}

ChatRetrievalGoal resolveChatRetrievalGoal(
    ExecutiveController* controller,
    const std::string& session_id,
    const std::optional<std::string>& active_goal,
    SessionGoalEmbedCache& cache,
    EmbeddingEngine* embed_engine) {
    ChatRetrievalGoal result;

    if (controller) {
        const Plan current_plan = controller->get_current_plan();
        const std::vector<float> executive_embedding = controller->get_goal_embedding();
        if (!current_plan.plan_id.empty() && !executive_embedding.empty()) {
            result.embedding = executive_embedding;
            result.source = "executive";
            return result;
        }
    }

    if (!active_goal || active_goal->empty()) {
        result.source = "none";
        return result;
    }

    const std::string normalized = normalizeSessionGoalText(*active_goal);
    if (normalized.empty()) {
        result.source = "none";
        return result;
    }

    const std::string resolved_session = session_id.empty() ? "default" : session_id;
    result.source = "session";

    if (const auto cached = cache.lookup(resolved_session, normalized)) {
        result.embedding = *cached;
        return result;
    }

    if (!embed_engine) {
        result.error = "embedding engine unavailable";
        return result;
    }

    try {
        const std::vector<float> embedded = embedStructuredGoal(embed_engine, normalized);
        if (embedded.empty()) {
            result.error = "session goal embed returned empty vector";
            return result;
        }
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        cache.store(resolved_session, normalized, embedded, now_ms);
        result.embedding = embedded;
        return result;
    } catch (const std::exception& ex) {
        result.error = std::string("session goal embed failed: ") + ex.what();
        return result;
    } catch (...) {
        result.error = "session goal embed failed: unknown error";
        return result;
    }
}

} // namespace Thoth
