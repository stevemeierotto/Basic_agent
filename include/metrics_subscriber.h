/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — E2-D3 MetricsSubscriber
 *
 * Protocol: docs/D_PHASE_PROTOCOL.md § D3, docs/cursor_list.md § D.3.0
 *
 * Passive observer — counters and raw observations only; no evaluation semantics.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_METRICS_SUBSCRIBER_H
#define THOTH_METRICS_SUBSCRIBER_H

#include "episode_events.h"
#include "json.hpp"
#include "pipeline_telemetry_service.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace Thoth {

class InProcessEpisodeEventChannel;

/** Subscriber-owned per-run aggregate (observational only). */
struct MetricsRunAggregate {
    std::size_t episode_completed_total = 0;
    float observed_final_success_score = 0.0f;
    std::size_t plan_step_count = 0;
    std::size_t histogram_score_samples = 0;
    std::int64_t last_pipeline_duration_ms = 0;
};

/**
 * Records episode and pipeline observations (E2-D3-01).
 * Frozen aggregation ops only; no eval/diag imports.
 */
class MetricsSubscriber final : public IEpisodeEventSubscriber {
public:
    void onEpisodeCompleted(const EpisodeCompleted& event) override;

    /** Pipeline-scoped metrics — C4 telemetry envelope only (no eval/diag). */
    void observePipelineTelemetry(const E2PipelineTelemetryRecord& record);

    std::size_t deliveryCount() const;

    const MetricsRunAggregate* runAggregateForRun(const std::string& run_id) const;

    nlohmann::json lastJsonlRecord() const;

    static std::size_t deliveryCountForTests();

    static const MetricsRunAggregate* runAggregateForTests(const std::string& run_id);

    static nlohmann::json lastJsonlRecordForTests();

    static void setJsonlSinkPathForTests(const std::string& path);

    static void resetJsonlSinkPathForTests();

    /** Testing only — last registered instance is on this channel (identity, not just count). */
    static bool isRegisteredOnChannelForTests(const InProcessEpisodeEventChannel& channel);

private:
    struct RunState {
        MetricsRunAggregate aggregate;
        std::vector<float> score_histogram_samples;
        std::int64_t last_episode_timestamp_ms = 0;
    };

    RunState& stateForRun(const std::string& run_id);

    void counterIncrement(std::size_t& field);
    void counterAdd(std::size_t& field, std::size_t delta);
    void gaugeSet(float& field, float value);
    void histogramObserve(std::vector<float>& samples, std::size_t& sample_count, float value);
    void durationObserveMs(std::int64_t& field, std::int64_t ms);

    nlohmann::json buildEpisodeJsonlRecord(const EpisodeCompleted& event,
                                           const MetricsRunAggregate& aggregate) const;
    nlohmann::json buildPipelineJsonlRecord(const E2PipelineTelemetryRecord& record,
                                            std::int64_t timestamp_ms) const;
    void appendJsonlRecord(const nlohmann::json& record);

    std::unordered_map<std::string, RunState> runs_;
    std::size_t delivery_count_ = 0;
    nlohmann::json last_jsonl_record_for_tests_ = nlohmann::json::object();
};

void registerMetricsSubscriber(IEpisodeEventChannel& channel);

} // namespace Thoth

#endif // THOTH_METRICS_SUBSCRIBER_H
