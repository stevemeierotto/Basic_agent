/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineEvent wire envelope (Plan G)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_event.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace Thoth {

namespace {

std::string formatTimestampUtc(int64_t timestamp_ms) {
    const std::time_t seconds = static_cast<std::time_t>(timestamp_ms / 1000);
    std::tm tm {};
#if defined(_WIN32)
    gmtime_s(&tm, &seconds);
#else
    gmtime_r(&seconds, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

} // namespace

const char* engineEventTypeName(EventType type) {
    switch (type) {
    case EventType::PLAN_CREATED:
        return "PLAN_CREATED";
    case EventType::STEP_STARTED:
        return "STEP_STARTED";
    case EventType::STEP_COMPLETED:
        return "STEP_COMPLETED";
    case EventType::STEP_FAILED:
        return "STEP_FAILED";
    case EventType::STEP_RETRYING:
        return "STEP_RETRYING";
    case EventType::PLAN_REVISED:
        return "PLAN_REVISED";
    case EventType::PLAN_COMPLETED:
        return "PLAN_COMPLETED";
    case EventType::PLAN_ABORTED:
        return "PLAN_ABORTED";
    case EventType::PLAN_FAILED:
        return "PLAN_FAILED";
    case EventType::STATE_CHANGED:
        return "STATE_CHANGED";
    case EventType::MODE_SWITCHED:
        return "MODE_SWITCHED";
    case EventType::EMBEDDING_FAILED:
        return "EMBEDDING_FAILED";
    case EventType::RETRIEVAL_DIAGNOSTICS:
        return "RETRIEVAL_DIAGNOSTICS";
    case EventType::INDEXING_STARTED:
        return "INDEXING_STARTED";
    case EventType::INDEXING_COMPLETED:
        return "INDEXING_COMPLETED";
    case EventType::PLAN_REUSE_INJECTION:
        return "PLAN_REUSE_INJECTION";
    case EventType::REFLECTION_REPLAN:
        return "REFLECTION_REPLAN";
    case EventType::PLAN_HISTORY_STORED:
        return "PLAN_HISTORY_STORED";
    }
    return "UNKNOWN";
}

EngineEvent makeEngineEvent(const ControllerEvent& source,
                            uint64_t sequence,
                            int64_t dispatch_time_ms) {
    EngineEvent event;
    event.event_id = sequence;
    event.sequence = sequence;
    event.timestamp = formatTimestampUtc(source.timestamp_ms != 0 ? source.timestamp_ms
                                                                : dispatch_time_ms);
    event.type = engineEventTypeName(source.type);
    event.session_id = source.session_id;
    event.plan_id = source.plan_id;
    event.step_id = source.step_id;
    event.controller_state_name = source.controller_state_name;
    event.metadata = source.metadata;
    return event;
}

nlohmann::json EngineEvent::toJson() const {
    return nlohmann::json{{"event_id", event_id},
                          {"sequence", sequence},
                          {"timestamp", timestamp},
                          {"type", type},
                          {"session_id", session_id},
                          {"plan_id", plan_id},
                          {"step_id", step_id},
                          {"controller_state_name", controller_state_name},
                          {"metadata", metadata}};
}

std::string frameEngineEventSse(const EngineEvent& event) {
    std::ostringstream out;
    out << "event: engine\n";
    out << "id: " << event.sequence << "\n";
    out << "data: " << event.toJson().dump() << "\n\n";
    return out.str();
}

} // namespace Thoth
