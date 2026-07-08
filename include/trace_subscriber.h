/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — E2-D3 TraceSubscriber (skeleton)
 *
 * Protocol: docs/D_PHASE_PROTOCOL.md § D3, docs/cursor_list.md § D.3.0
 *
 * Passive observer — correlation segments only; no statistics or scoring.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_TRACE_SUBSCRIBER_H
#define THOTH_TRACE_SUBSCRIBER_H

#include "episode_events.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Thoth {

class InProcessEpisodeEventChannel;

struct TraceSegmentRecord {
    std::string segment_type;
    std::int64_t timestamp_ms = 0;
    std::string plan_id;
    std::string run_id;
    std::string source;
};

/**
 * Append-only trace segments from immutable EpisodeCompleted (D3 Step 1 skeleton).
 */
class TraceSubscriber final : public IEpisodeEventSubscriber {
public:
    void onEpisodeCompleted(const EpisodeCompleted& event) override;

    std::size_t segmentCount() const;

    const TraceSegmentRecord* segmentAtForTests(std::size_t index) const;

    static std::size_t segmentCountForTests();

    /** Testing only — last registered instance is on this channel (identity, not just count). */
    static bool isRegisteredOnChannelForTests(const InProcessEpisodeEventChannel& channel);

private:
    static constexpr std::size_t kRingCapacity = 64;
    std::vector<TraceSegmentRecord> segments_;
};

void registerTraceSubscriber(IEpisodeEventChannel& channel);

} // namespace Thoth

#endif // THOTH_TRACE_SUBSCRIBER_H
