/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — in-process episode event channel (E2-C2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODE_EVENT_CHANNEL_H
#define THOTH_EPISODE_EVENT_CHANNEL_H

#include "episode_events.h"

namespace Thoth {

/** Fan-out delivery to registered subscribers. Failures isolated per subscriber. */
class InProcessEpisodeEventChannel final : public IEpisodeEventChannel {
public:
    void publish(const EpisodeCompleted& event) override;
    void subscribe(std::shared_ptr<IEpisodeEventSubscriber> subscriber) override;

private:
    std::vector<std::shared_ptr<IEpisodeEventSubscriber>> subscribers_;
};

} // namespace Thoth

#endif // THOTH_EPISODE_EVENT_CHANNEL_H
