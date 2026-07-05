/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C4 pipeline telemetry service (measurement only)
 *
 * Protocol: docs/C_PHASE_PROTOCOL.md § E2-C4
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_PIPELINE_TELEMETRY_SERVICE_H
#define THOTH_PIPELINE_TELEMETRY_SERVICE_H

#include "json.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Thoth {

/**
 * Run/episode attribution and clock anchors only — must NOT carry evaluation or
 * diagnostic results. See C_PHASE_PROTOCOL.md § E2-C4 C4.2 ownership rule.
 */
struct E2PipelineTelemetryContext {
    std::string run_id;
    std::string env_hash;
    std::string plan_id;
    std::string goal_id;
    std::int64_t episode_completed_at_ms = 0;
    std::int64_t subscriber_entry_ms = 0;
};

struct E2PipelineStageTiming {
    std::string stage;
    /** steady_clock delta — never derived from system_clock. */
    std::int64_t duration_ms = 0;
};

/**
 * Subscriber-recorded stage segments — no eval/diag payloads.
 * All duration_ms fields are steady_clock deltas.
 * publication_to_subscriber_ms (optional) is a system_clock anchor delta only.
 */
struct E2PipelineStageTimings {
    std::int64_t mapping_duration_ms = 0;
    std::int64_t evaluation_duration_ms = 0;
    std::int64_t diagnostic_duration_ms = 0;
    std::int64_t pipeline_duration_ms = 0;
    std::optional<std::int64_t> publication_to_subscriber_ms;
    int episodes_processed = 1;
};

struct E2PipelineTelemetryRecord {
    std::string run_id;
    std::string plan_id;
    std::string telemetry_tier = "ARCHITECTURE";
    std::string telemetry_schema_version = "1.0";
    std::int64_t pipeline_duration_ms = 0;
    std::vector<E2PipelineStageTiming> stages;
    std::optional<std::int64_t> publication_to_subscriber_ms;
    int episodes_processed = 1;
    std::optional<std::int64_t> queue_delay_ms;
};

/**
 * Phase C4 — stateless telemetry boundary.
 * Subscriber-owned measurement; service-owned record assembly and JSONL shape only.
 * Consumes timing snapshots and attribution only; never invokes evaluation or diagnostics.
 */
class IE2PipelineTelemetryService {
public:
    virtual ~IE2PipelineTelemetryService() = default;

    virtual E2PipelineTelemetryRecord recordPipelineRun(
        const E2PipelineStageTimings& timings,
        const E2PipelineTelemetryContext& context) const = 0;
};

const IE2PipelineTelemetryService& episodicPipelineTelemetryService();

nlohmann::json e2PipelineTelemetryToJson(const E2PipelineTelemetryRecord& record);

/** Testing only — inject throw from recordPipelineRun for E2-C4-02 non-blocking proof. */
void setE2PipelineTelemetryThrowForTests(bool enabled);

} // namespace Thoth

#endif // THOTH_PIPELINE_TELEMETRY_SERVICE_H
