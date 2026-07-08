/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — execution-domain episode events (E2-C2)
 *
 * Protocol: docs/C_PHASE_PROTOCOL.md § E2-C2
 *
 * Domain event only — no evaluation semantics.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODE_EVENTS_H
#define THOTH_EPISODE_EVENTS_H

#include "json.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Thoth {

/**
 * Execution-domain event: a completed goal episode snapshot.
 * Immutable after publication. Reusable by evaluation, telemetry, replay, tracing.
 */
struct EpisodeCompleted {
    std::string plan_id;
    std::string goal;
    /** COMPLETED | FAILED */
    std::string terminal_state;
    float final_success_score = 0.0f;
    std::int64_t completed_at_ms = 0;
    std::string run_id;
    std::string env_hash;
    nlohmann::json plan_snapshot = nlohmann::json::object();
    nlohmann::json trajectory_snapshot = nlohmann::json::object();

    nlohmann::json toJson() const;
};

class IEpisodeEventSubscriber {
public:
    virtual ~IEpisodeEventSubscriber() = default;
    virtual void onEpisodeCompleted(const EpisodeCompleted& event) = 0;
};

class IEpisodeEventChannel {
public:
    virtual ~IEpisodeEventChannel() = default;
    /** Fire-and-forget fan-out. Delivery order is FIFO by registration order. */
    virtual void publish(const EpisodeCompleted& event) = 0;
    virtual void subscribe(std::shared_ptr<IEpisodeEventSubscriber> subscriber) = 0;
};

} // namespace Thoth

#endif // THOTH_EPISODE_EVENTS_H
