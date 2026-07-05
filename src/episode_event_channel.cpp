/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — in-process episode event channel (E2-C2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "episode_event_channel.h"

#include "logger.h"

namespace Thoth {

void InProcessEpisodeEventChannel::publish(const EpisodeCompleted& event) {
    for (const auto& subscriber : subscribers_) {
        if (!subscriber) {
            continue;
        }
        try {
            subscriber->onEpisodeCompleted(event);
        } catch (const std::exception& e) {
            StructuredLogger::instance().log(
                LogLevel::Warn, "episode_channel", "subscriber_failure",
                std::string("isolated subscriber failure: ") + e.what());
        } catch (...) {
            StructuredLogger::instance().log(LogLevel::Warn, "episode_channel",
                                             "subscriber_failure",
                                             "isolated subscriber failure: unknown");
        }
    }
}

void InProcessEpisodeEventChannel::subscribe(std::shared_ptr<IEpisodeEventSubscriber> subscriber) {
    if (subscriber) {
        subscribers_.push_back(std::move(subscriber));
    }
}

} // namespace Thoth
