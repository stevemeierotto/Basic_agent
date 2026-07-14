/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — SSE client session queues (Plan G)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_sse_session.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace Thoth {

bool SseSession::enqueue(std::string framed_chunk) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return false;
    }
    if (queue_.size() >= kDefaultQueueCapacity) {
        closed_ = true;
        cv_.notify_all();
        return false;
    }
    queue_.push_back(std::move(framed_chunk));
    cv_.notify_one();
    return true;
}

bool SseSession::waitForChunk(std::string& out,
                              std::chrono::milliseconds timeout,
                              bool& closed_out) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); })) {
        closed_out = closed_;
        return false;
    }
    if (queue_.empty()) {
        closed_out = closed_;
        return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    closed_out = closed_;
    return true;
}

void SseSession::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    cv_.notify_all();
}

bool SseSession::isClosed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

SseSessionManager::SseSessionManager(EngineRuntime& runtime) : runtime_(runtime) {
    event_subscription_id_ = runtime_.subscribeEvents(
        [this](const EngineEvent& event) { publishToSessions(event); });
}

SseSessionManager::~SseSessionManager() {
    closeAll();
    if (event_subscription_id_ != 0) {
        runtime_.unsubscribeEvents(event_subscription_id_);
        event_subscription_id_ = 0;
    }
}

std::shared_ptr<SseSession> SseSessionManager::createSession() {
    auto session = std::make_shared<SseSession>();
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.push_back(session);
    return session;
}

void SseSessionManager::removeSession(const std::shared_ptr<SseSession>& session) {
    if (!session) {
        return;
    }
    session->close();
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.erase(
        std::remove_if(sessions_.begin(),
                       sessions_.end(),
                       [&session](const std::weak_ptr<SseSession>& weak) {
                           const auto locked = weak.lock();
                           return !locked || locked == session;
                       }),
        sessions_.end());
}

void SseSessionManager::publishToSessions(const EngineEvent& event) {
    const std::string framed = frameEngineEventSse(event);
    std::vector<std::shared_ptr<SseSession>> live_sessions;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& weak : sessions_) {
            if (auto session = weak.lock()) {
                live_sessions.push_back(std::move(session));
            }
        }
    }

    for (const auto& session : live_sessions) {
        if (!session->enqueue(framed)) {
            std::cerr << "[SseSessionManager] dropping slow SSE client (queue overflow)\n";
            removeSession(session);
        }
    }
}

void SseSessionManager::closeAll() {
    std::vector<std::shared_ptr<SseSession>> live_sessions;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& weak : sessions_) {
            if (auto session = weak.lock()) {
                live_sessions.push_back(std::move(session));
            }
        }
        sessions_.clear();
    }
    for (const auto& session : live_sessions) {
        session->close();
    }
}

std::size_t SseSessionManager::sessionCountForTests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t count = 0;
    for (const auto& weak : sessions_) {
        if (!weak.expired()) {
            ++count;
        }
    }
    return count;
}

} // namespace Thoth
