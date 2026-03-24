/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include <functional>
#include <cstdint>
#include "json.hpp"

enum class EventType {
    PLAN_CREATED,
    STEP_STARTED,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_RETRYING,
    PLAN_REVISED,
    PLAN_COMPLETED,
    PLAN_ABORTED,
    PLAN_FAILED,
    STATE_CHANGED,
    MODE_SWITCHED,
    EMBEDDING_FAILED,
    RETRIEVAL_DIAGNOSTICS,
    INDEXING_STARTED,
    INDEXING_COMPLETED
};

struct ControllerEvent {
    EventType type;
    std::string session_id;    // Correlate event with UI session
    std::string plan_id;
    std::string step_id;       // Empty string if not step-specific
    std::string controller_state_name;
    nlohmann::json metadata;   // Flexible payload for UI and logging
    int64_t timestamp_ms;
};

// Callback type — UI, CLI, and trace logger all subscribe via this
using EventCallback = std::function<void(const ControllerEvent&)>;
