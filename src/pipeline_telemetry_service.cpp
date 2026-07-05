/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C4 E2PipelineTelemetryService
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "pipeline_telemetry_service.h"

#include <stdexcept>

namespace Thoth {

namespace {

bool g_throw_for_tests = false;

class E2PipelineTelemetryService final : public IE2PipelineTelemetryService {
public:
    E2PipelineTelemetryRecord recordPipelineRun(
        const E2PipelineStageTimings& timings,
        const E2PipelineTelemetryContext& context) const override {
        if (g_throw_for_tests) {
            throw std::runtime_error("E2-C4-02 telemetry throw injection");
        }
        E2PipelineTelemetryRecord record;
        record.run_id = context.run_id;
        record.plan_id = context.plan_id;
        record.telemetry_tier = "ARCHITECTURE";
        record.telemetry_schema_version = "1.0";
        record.pipeline_duration_ms = timings.pipeline_duration_ms;
        record.episodes_processed = timings.episodes_processed;
        record.publication_to_subscriber_ms = timings.publication_to_subscriber_ms;
        record.queue_delay_ms = std::nullopt;

        if (timings.publication_to_subscriber_ms.has_value()) {
            record.stages.push_back(
                {"publication_to_subscriber", *timings.publication_to_subscriber_ms});
        }
        record.stages.push_back({"mapping", timings.mapping_duration_ms});
        record.stages.push_back({"evaluation", timings.evaluation_duration_ms});
        record.stages.push_back({"diagnostics", timings.diagnostic_duration_ms});
        record.stages.push_back({"pipeline_total", timings.pipeline_duration_ms});
        return record;
    }
};

const E2PipelineTelemetryService& telemetryServiceInstance() {
    static const E2PipelineTelemetryService instance;
    return instance;
}

} // namespace

const IE2PipelineTelemetryService& episodicPipelineTelemetryService() {
    return telemetryServiceInstance();
}

nlohmann::json e2PipelineTelemetryToJson(const E2PipelineTelemetryRecord& record) {
    nlohmann::json stages = nlohmann::json::array();
    for (const auto& stage : record.stages) {
        stages.push_back({{"stage", stage.stage}, {"duration_ms", stage.duration_ms}});
    }

    nlohmann::json j = {{"event_type", "E2_EVAL_TELEMETRY_PIPELINE"},
                        {"run_id", record.run_id},
                        {"telemetry_tier", record.telemetry_tier},
                        {"telemetry_schema_version", record.telemetry_schema_version},
                        {"pipeline_duration_ms", record.pipeline_duration_ms},
                        {"stages", stages},
                        {"episodes_processed", record.episodes_processed}};
    if (!record.plan_id.empty()) {
        j["plan_id"] = record.plan_id;
    }
    if (record.publication_to_subscriber_ms.has_value()) {
        j["publication_to_subscriber_ms"] = *record.publication_to_subscriber_ms;
    }
    if (record.queue_delay_ms.has_value()) {
        j["queue_delay_ms"] = *record.queue_delay_ms;
    }
    return j;
}

void setE2PipelineTelemetryThrowForTests(bool enabled) {
    g_throw_for_tests = enabled;
}

} // namespace Thoth
