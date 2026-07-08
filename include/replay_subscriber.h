/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-D2 ReplaySubscriber
 *
 * Protocol: docs/D_PHASE_PROTOCOL.md § D2
 *
 * Replay is subscriber-internal only — never republishes to IEpisodeEventChannel.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_REPLAY_SUBSCRIBER_H
#define THOTH_REPLAY_SUBSCRIBER_H

#include "episode_events.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace Thoth {

/**
 * Captures immutable EpisodeCompleted snapshots (append-only FIFO) and replays
 * them to an internal sink via replayCaptured(index) — never via the event channel.
 */
class ReplaySubscriber final : public IEpisodeEventSubscriber {
public:
    void onEpisodeCompleted(const EpisodeCompleted& event) override;

    /** Deliver captured episode at index to the internal replay sink only. */
    bool replayCaptured(std::size_t index) const;

    std::size_t captureCount() const;

    /** Testing — read captured episode without replaying (not production API). */
    const EpisodeCompleted* capturedAtForTests(std::size_t index) const;

    /** Testing — register sink invoked by replayCaptured (not production API). */
    void setReplaySinkForTests(std::function<void(const EpisodeCompleted&)> sink);

    /** Testing — number of stored episodes (not production API). */
    static std::size_t captureCountForTests();

private:
    std::vector<EpisodeCompleted> captured_;
    std::function<void(const EpisodeCompleted&)> replay_sink_;
};

void registerReplaySubscriber(IEpisodeEventChannel& channel);

} // namespace Thoth

#endif // THOTH_REPLAY_SUBSCRIBER_H
