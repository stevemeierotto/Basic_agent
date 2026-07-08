/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — E2-D3 TraceSubscriber (skeleton)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "trace_subscriber.h"

#include "episode_event_channel.h"

#include <memory>

namespace Thoth {

namespace {

std::weak_ptr<TraceSubscriber> g_last_trace_subscriber_for_tests;

} // namespace

void TraceSubscriber::onEpisodeCompleted(const EpisodeCompleted& event) {
    TraceSegmentRecord segment;
    segment.segment_type = "episode_completed";
    segment.timestamp_ms = event.completed_at_ms;
    segment.plan_id = event.plan_id;
    segment.run_id = event.run_id;
    segment.source = "channel";

    segments_.push_back(std::move(segment));
    if (segments_.size() > kRingCapacity) {
        segments_.erase(segments_.begin());
    }
}

std::size_t TraceSubscriber::segmentCount() const {
    return segments_.size();
}

const TraceSegmentRecord* TraceSubscriber::segmentAtForTests(std::size_t index) const {
    if (index >= segments_.size()) {
        return nullptr;
    }
    return &segments_[index];
}

std::size_t TraceSubscriber::segmentCountForTests() {
    if (const auto locked = g_last_trace_subscriber_for_tests.lock()) {
        return locked->segmentCount();
    }
    return 0;
}

bool TraceSubscriber::isRegisteredOnChannelForTests(
    const InProcessEpisodeEventChannel& channel) {
    const auto registered = g_last_trace_subscriber_for_tests.lock();
    if (!registered) {
        return false;
    }
    return channel.containsSubscriberForTests(registered.get());
}

void registerTraceSubscriber(IEpisodeEventChannel& channel) {
    auto subscriber = std::make_shared<TraceSubscriber>();
    g_last_trace_subscriber_for_tests = subscriber;
    channel.subscribe(subscriber);
}

} // namespace Thoth
