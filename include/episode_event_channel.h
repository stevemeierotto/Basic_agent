/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — in-process episode event channel (E2-C2, E2-D1)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODE_EVENT_CHANNEL_H
#define THOTH_EPISODE_EVENT_CHANNEL_H

#include "episode_events.h"

#include <cstddef>
#include <optional>

namespace Thoth {

/**
 * In-process fan-out channel.
 *
 * Delivery policy: FIFO by registration order. Each subscriber receives the same
 * immutable event snapshot. Per-subscriber failures are isolated (logged only).
 *
 * Channel imports no evaluation, diagnostic, or telemetry semantics.
 */
class InProcessEpisodeEventChannel final : public IEpisodeEventChannel {
public:
    void publish(const EpisodeCompleted& event) override;
    void subscribe(std::shared_ptr<IEpisodeEventSubscriber> subscriber) override;

    /** Testing only — subscriber registry size; not exposed to Executive. */
    std::size_t subscriberCountForTests() const;

    /** Testing only — last event passed to publish(); unset until first publish. */
    std::optional<EpisodeCompleted> lastPublishedEventForTests() const;

    /** Testing only — whether a subscriber instance is registered (identity, not just count). */
    bool containsSubscriberForTests(const IEpisodeEventSubscriber* subscriber) const;

private:
    std::vector<std::shared_ptr<IEpisodeEventSubscriber>> subscribers_;
    std::optional<EpisodeCompleted> last_published_for_tests_;
};

} // namespace Thoth

#endif // THOTH_EPISODE_EVENT_CHANNEL_H
