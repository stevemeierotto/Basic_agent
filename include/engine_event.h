/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineEvent wire envelope (Plan G)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "controller_event.h"
#include "json.hpp"

#include <cstdint>
#include <string>

namespace Thoth {

struct EngineEvent {
    uint64_t event_id{0};
    uint64_t sequence{0};
    std::string timestamp;
    std::string type;
    std::string session_id;
    std::string plan_id;
    std::string step_id;
    std::string controller_state_name;
    nlohmann::json metadata;

    nlohmann::json toJson() const;
};

const char* engineEventTypeName(EventType type);

/** Map ControllerEvent to EngineEvent; sequence assigned by dispatch thread. */
EngineEvent makeEngineEvent(const ControllerEvent& source,
                            uint64_t sequence,
                            int64_t dispatch_time_ms);

/** Format SSE wire chunk for one EngineEvent. */
std::string frameEngineEventSse(const EngineEvent& event);

} // namespace Thoth
