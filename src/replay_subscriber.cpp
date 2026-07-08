/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-D2 ReplaySubscriber
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "replay_subscriber.h"

#include <memory>

namespace Thoth {

namespace {

std::weak_ptr<ReplaySubscriber> g_last_replay_subscriber_for_tests;

} // namespace

void ReplaySubscriber::onEpisodeCompleted(const EpisodeCompleted& event) {
    captured_.push_back(event);
}

bool ReplaySubscriber::replayCaptured(std::size_t index) const {
    if (index >= captured_.size() || !replay_sink_) {
        return false;
    }
    replay_sink_(captured_[index]);
    return true;
}

std::size_t ReplaySubscriber::captureCount() const {
    return captured_.size();
}

const EpisodeCompleted* ReplaySubscriber::capturedAtForTests(std::size_t index) const {
    if (index >= captured_.size()) {
        return nullptr;
    }
    return &captured_[index];
}

void ReplaySubscriber::setReplaySinkForTests(std::function<void(const EpisodeCompleted&)> sink) {
    replay_sink_ = std::move(sink);
}

std::size_t ReplaySubscriber::captureCountForTests() {
    if (const auto locked = g_last_replay_subscriber_for_tests.lock()) {
        return locked->captureCount();
    }
    return 0;
}

void registerReplaySubscriber(IEpisodeEventChannel& channel) {
    auto subscriber = std::make_shared<ReplaySubscriber>();
    g_last_replay_subscriber_for_tests = subscriber;
    channel.subscribe(subscriber);
}

} // namespace Thoth
