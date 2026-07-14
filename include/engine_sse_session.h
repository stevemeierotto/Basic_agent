/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — SSE client session queues (Plan G)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "engine_event.h"
#include "engine_runtime.h"

#include <chrono>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Thoth {

class SseSession {
public:
    static constexpr std::size_t kDefaultQueueCapacity = 256;

    bool enqueue(std::string framed_chunk);
    bool waitForChunk(std::string& out,
                      std::chrono::milliseconds timeout,
                      bool& closed_out);
    void close();
    bool isClosed() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    bool closed_{false};
};

class SseSessionManager {
public:
    explicit SseSessionManager(EngineRuntime& runtime);
    ~SseSessionManager();

    SseSessionManager(const SseSessionManager&) = delete;
    SseSessionManager& operator=(const SseSessionManager&) = delete;

    std::shared_ptr<SseSession> createSession();
    void removeSession(const std::shared_ptr<SseSession>& session);
    void publishToSessions(const EngineEvent& event);
    void closeAll();

    std::size_t sessionCountForTests() const;

private:
    EngineRuntime& runtime_;
    uint64_t event_subscription_id_{0};
    mutable std::mutex mutex_;
    std::vector<std::weak_ptr<SseSession>> sessions_;
};

} // namespace Thoth
