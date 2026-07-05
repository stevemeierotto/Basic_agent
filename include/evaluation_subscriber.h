/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2-C2 EvaluationSubscriber (first episode consumer)
 *
 * Protocol: docs/C_PHASE_PROTOCOL.md § E2-C2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EVALUATION_SUBSCRIBER_H
#define THOTH_EVALUATION_SUBSCRIBER_H

#include "episode_events.h"
#include "diagnostic_service.h"
#include "episodic_learning_eval.h"
#include "pipeline_telemetry_service.h"

#include <optional>

namespace Thoth {

/** Deterministic, side-effect-free mapping — subscriber-owned (E2-C2). */
struct ProductionEpisodeMapping {
    EpisodicLearningArmObservation cold;
    EpisodicLearningArmObservation warm;
    EpisodicLearningExpectations expectations;
};

ProductionEpisodeMapping mapEpisodeToProductionObservations(const EpisodeCompleted& event);

/** Consumes EpisodeCompleted; invokes evaluation service at INTEGRATION tier only. */
class EvaluationSubscriber final : public IEpisodeEventSubscriber {
public:
    void onEpisodeCompleted(const EpisodeCompleted& event) override;

    /** Testing observation interface — last summary produced by subscriber (not production API). */
    static const EpisodicLearningSummary* lastSummaryForTests();

    /** Testing observation interface — last run diagnostics (not production API). */
    static const EvaluationDiagnosticsSummary* lastRunDiagnosticsForTests();

    /** Testing observation interface — last captured stage timings (not production API). */
    static const E2PipelineStageTimings* lastStageTimingsForTests();

    /** Testing observation interface — last assembled telemetry record when telemetry path ran. */
    static const E2PipelineTelemetryRecord* lastTelemetryRecordForTests();
};

void setEvaluationSubscriberPipelineTelemetryEnabled(bool enabled);
bool evaluationSubscriberPipelineTelemetryEnabled();

/** Testing only — pin evaluation config for C5 path equivalence (unset = production INTEGRATION). */
void setEvaluationSubscriberEvalConfigForTests(std::optional<E2EvalConfig> config);

/** Flips `g_pipeline_telemetry_enabled` only — no eval/diag calls, no side effects. */

void registerEvaluationSubscriber(IEpisodeEventChannel& channel);

} // namespace Thoth

#endif // THOTH_EVALUATION_SUBSCRIBER_H
