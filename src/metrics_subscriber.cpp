/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — E2-D3 MetricsSubscriber
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "metrics_subscriber.h"

#include "episode_event_channel.h"
#include "logger.h"

#include <fstream>
#include <memory>

namespace Thoth {

namespace {

constexpr const char* kMetricsSchemaVersion = "1.0";
constexpr const char* kDefaultJsonlSink = "logs/e2_metrics.jsonl";
constexpr std::size_t kHistogramMaxSamples = 16;

std::weak_ptr<MetricsSubscriber> g_last_metrics_subscriber_for_tests;
std::string g_jsonl_sink_path = kDefaultJsonlSink;

std::size_t countPlanSteps(const nlohmann::json& plan_snapshot) {
    if (!plan_snapshot.contains("steps") || !plan_snapshot["steps"].is_array()) {
        return 0;
    }
    return plan_snapshot["steps"].size();
}

} // namespace

MetricsSubscriber::RunState& MetricsSubscriber::stateForRun(const std::string& run_id) {
    const std::string key = run_id.empty() ? "unknown_run" : run_id;
    return runs_[key];
}

void MetricsSubscriber::counterIncrement(std::size_t& field) {
    ++field;
}

void MetricsSubscriber::counterAdd(std::size_t& field, const std::size_t delta) {
    field += delta;
}

void MetricsSubscriber::gaugeSet(float& field, const float value) {
    field = value;
}

void MetricsSubscriber::histogramObserve(std::vector<float>& samples,
                                         std::size_t& sample_count,
                                         const float value) {
    samples.push_back(value);
    if (samples.size() > kHistogramMaxSamples) {
        samples.erase(samples.begin());
    }
    ++sample_count;
}

void MetricsSubscriber::durationObserveMs(std::int64_t& field, const std::int64_t ms) {
    field = ms;
}

void MetricsSubscriber::onEpisodeCompleted(const EpisodeCompleted& event) {
    ++delivery_count_;

    RunState& state = stateForRun(event.run_id);
    auto& agg = state.aggregate;

    counterIncrement(agg.episode_completed_total);
    gaugeSet(agg.observed_final_success_score, event.final_success_score);
    histogramObserve(state.score_histogram_samples, agg.histogram_score_samples,
                     event.final_success_score);
    counterAdd(agg.plan_step_count, countPlanSteps(event.plan_snapshot));
    state.last_episode_timestamp_ms = event.completed_at_ms;

    const nlohmann::json record = buildEpisodeJsonlRecord(event, agg);
    appendJsonlRecord(record);
}

void MetricsSubscriber::observePipelineTelemetry(const E2PipelineTelemetryRecord& record) {
    RunState& state = stateForRun(record.run_id);
    durationObserveMs(state.aggregate.last_pipeline_duration_ms, record.pipeline_duration_ms);

    const nlohmann::json jsonl =
        buildPipelineJsonlRecord(record, state.last_episode_timestamp_ms);
    appendJsonlRecord(jsonl);
}

nlohmann::json MetricsSubscriber::buildEpisodeJsonlRecord(
    const EpisodeCompleted& event, const MetricsRunAggregate& aggregate) const {
    return {{"metrics_schema_version", kMetricsSchemaVersion},
            {"record_type", "episode_observation"},
            {"timestamp_ms", event.completed_at_ms},
            {"run_id", event.run_id},
            {"plan_id", event.plan_id},
            {"observations",
             {{"episode_completed_total", aggregate.episode_completed_total},
              {"observed_final_success_score", aggregate.observed_final_success_score},
              {"plan_step_count", aggregate.plan_step_count},
              {"terminal_state_label", event.terminal_state}}}};
}

nlohmann::json MetricsSubscriber::buildPipelineJsonlRecord(
    const E2PipelineTelemetryRecord& record, const std::int64_t timestamp_ms) const {
    nlohmann::json pipeline = {{"telemetry_schema_version", record.telemetry_schema_version},
                               {"telemetry_tier", record.telemetry_tier},
                               {"pipeline_duration_ms", record.pipeline_duration_ms},
                               {"episodes_processed", record.episodes_processed}};
    if (!record.plan_id.empty()) {
        pipeline["plan_id"] = record.plan_id;
    }
    if (record.publication_to_subscriber_ms.has_value()) {
        pipeline["publication_to_subscriber_ms"] = *record.publication_to_subscriber_ms;
    }
    if (record.queue_delay_ms.has_value()) {
        pipeline["queue_delay_ms"] = *record.queue_delay_ms;
    }

    nlohmann::json stages = nlohmann::json::array();
    for (const auto& stage : record.stages) {
        stages.push_back({{"stage", stage.stage}, {"duration_ms", stage.duration_ms}});
    }
    pipeline["stages"] = stages;

    return {{"metrics_schema_version", kMetricsSchemaVersion},
            {"record_type", "pipeline_observation"},
            {"timestamp_ms", timestamp_ms},
            {"run_id", record.run_id},
            {"plan_id", record.plan_id},
            {"pipeline", pipeline}};
}

void MetricsSubscriber::appendJsonlRecord(const nlohmann::json& record) {
    last_jsonl_record_for_tests_ = record;
    try {
        std::ofstream out(g_jsonl_sink_path, std::ios::app);
        if (!out.is_open()) {
            StructuredLogger::instance().log(LogLevel::Warn, "metrics_subscriber",
                                             "jsonl_sink_failure",
                                             "could not open metrics jsonl sink");
            return;
        }
        out << record.dump() << '\n';
    } catch (const std::exception& e) {
        StructuredLogger::instance().log(LogLevel::Warn, "metrics_subscriber", "jsonl_sink_failure",
                                         std::string("metrics jsonl append failed: ") + e.what());
    } catch (...) {
        StructuredLogger::instance().log(LogLevel::Warn, "metrics_subscriber", "jsonl_sink_failure",
                                         "metrics jsonl append failed: unknown");
    }
}

std::size_t MetricsSubscriber::deliveryCount() const {
    return delivery_count_;
}

const MetricsRunAggregate* MetricsSubscriber::runAggregateForRun(
    const std::string& run_id) const {
    const std::string key = run_id.empty() ? "unknown_run" : run_id;
    const auto it = runs_.find(key);
    if (it == runs_.end()) {
        return nullptr;
    }
    return &it->second.aggregate;
}

nlohmann::json MetricsSubscriber::lastJsonlRecord() const {
    return last_jsonl_record_for_tests_;
}

std::size_t MetricsSubscriber::deliveryCountForTests() {
    if (const auto locked = g_last_metrics_subscriber_for_tests.lock()) {
        return locked->deliveryCount();
    }
    return 0;
}

const MetricsRunAggregate* MetricsSubscriber::runAggregateForTests(const std::string& run_id) {
    if (const auto locked = g_last_metrics_subscriber_for_tests.lock()) {
        return locked->runAggregateForRun(run_id);
    }
    return nullptr;
}

nlohmann::json MetricsSubscriber::lastJsonlRecordForTests() {
    if (const auto locked = g_last_metrics_subscriber_for_tests.lock()) {
        return locked->lastJsonlRecord();
    }
    return nlohmann::json::object();
}

void MetricsSubscriber::setJsonlSinkPathForTests(const std::string& path) {
    g_jsonl_sink_path = path;
}

void MetricsSubscriber::resetJsonlSinkPathForTests() {
    g_jsonl_sink_path = kDefaultJsonlSink;
}

bool MetricsSubscriber::isRegisteredOnChannelForTests(
    const InProcessEpisodeEventChannel& channel) {
    const auto registered = g_last_metrics_subscriber_for_tests.lock();
    if (!registered) {
        return false;
    }
    return channel.containsSubscriberForTests(registered.get());
}

void registerMetricsSubscriber(IEpisodeEventChannel& channel) {
    auto subscriber = std::make_shared<MetricsSubscriber>();
    g_last_metrics_subscriber_for_tests = subscriber;
    channel.subscribe(subscriber);
}

} // namespace Thoth
