/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — MemoryPruner / memory consolidation (M2 policy + M3 operational API)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/memory_pruner.h"
#include "../include/clock.h"
#include "../include/config.h"
#include "../include/memory_consolidation_config.h"
#include "../include/memory_consolidation_metrics.h"
#include "../include/decision_trace.h"
#include "../include/embedding_engine.h"
#include "../include/episodic_memory.h"
#include "../include/memory.h"
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace Thoth {

namespace {

constexpr int64_t kMsPerDay = 86'400'000LL;

int daysSince(int64_t nowMs, int64_t thenMs) {
    if (thenMs <= 0 || nowMs <= thenMs) {
        return 0;
    }
    return static_cast<int>((nowMs - thenMs) / kMsPerDay);
}

int64_t elapsedMs(int64_t startMs, int64_t endMs) {
    return endMs - startMs;
}

std::string batchDigest(const std::vector<MessageRecord>& batch) {
    std::ostringstream canonical;
    for (const auto& msg : batch) {
        canonical << msg.role << '\n' << msg.timestamp_ms << '\n' << msg.content << "\n---\n";
    }
    return Memory::calculateContentHash(canonical.str());
}

bool mockEmbedEmpty() {
    const char* env = std::getenv("THOTH_MOCK_EMBED_EMPTY");
    return env && (std::string(env) == "1" || std::string(env) == "true");
}

} // namespace

PruningPolicy PruningPolicy::fromConfig(const Config& config) {
    PruningPolicy policy;
    policy.max_hot_messages = config.memory_max_hot_messages;
    policy.max_hot_age_days = config.memory_max_hot_age_days;
    policy.prune_batch_size = config.memory_prune_batch_size;
    return policy;
}

MemoryPruner::MemoryPruner(MemoryRepository& repo,
                           const PruningPolicy& policy,
                           LLMInterface* llm,
                           EmbeddingEngine* embeddingEngine,
                           std::shared_ptr<Clock> clock)
    : repo_(repo),
      policy_(policy),
      summaryGenerator_(llm),
      embeddingEngine_(embeddingEngine),
      clock_(clock ? std::move(clock) : makeSystemClock()) {}

ConsolidationDecision MemoryPruner::evaluatePolicy(const std::string& sessionId) const {
    ConsolidationDecision decision;

    decision.hot_count = repo_.getHotMessageCount(sessionId);
    if (decision.hot_count <= 0) {
        return decision;
    }

    const int64_t now = clock_->nowMs();

    if (decision.hot_count > static_cast<int>(policy_.max_hot_messages)) {
        decision.reasons |= ConsolidationReason::HOT_COUNT;
    }

    if (policy_.max_hot_age_days > 0) {
        const int64_t age_limit_ms = static_cast<int64_t>(policy_.max_hot_age_days) * kMsPerDay;

        if (auto session = repo_.getSession(sessionId)) {
            decision.session_age_days = daysSince(now, session->updated_at_ms);
            if (now - session->updated_at_ms > age_limit_ms) {
                decision.reasons |= ConsolidationReason::SESSION_INACTIVE;
            }
        }

        if (auto oldest_ts = repo_.getOldestHotMessageTimestamp(sessionId)) {
            decision.oldest_message_age_days = daysSince(now, *oldest_ts);
            if (now - *oldest_ts > age_limit_ms) {
                decision.reasons |= ConsolidationReason::OLDEST_MESSAGE;
            }
        }
    }

    return decision;
}

ConsolidationStatus MemoryPruner::buildStatus(const std::string& sessionId,
                                              bool marked_stale,
                                              bool goal_active) const {
    ConsolidationStatus status;
    status.session_id = sessionId;
    status.decision = evaluatePolicy(sessionId);
    status.max_hot_messages = static_cast<int>(policy_.max_hot_messages);
    status.max_hot_age_days = policy_.max_hot_age_days;
    status.prune_batch_size = static_cast<int>(policy_.prune_batch_size);
    status.max_batches_per_invocation = static_cast<int>(policy_.max_batches_per_invocation);
    status.marked_stale = marked_stale;
    status.goal_active = goal_active;
    status.embed_ready = isEmbedReady();
    return status;
}

bool MemoryPruner::shouldEnterConsolidation(const ConsolidationDecision& decision,
                                            const ConsolidationRequest& request) const {
    if (decision.hot_count <= 0) {
        return false;
    }
    if (decision.shouldConsolidate()) {
        return true;
    }
    return request.ignore_thresholds;
}

void MemoryPruner::finalizeResultCompat(ConsolidationResult& result) const {
    result.total_archived = result.archived;
    result.batches_completed = result.batches;
    result.final_decision = result.decision;
}

BatchConsolidationOutcome MemoryPruner::consolidateOneBatchInternal(
    const std::string& sessionId,
    const ConsolidationDecision& decision,
    ConsolidationSource source,
    const std::string& requested_by) {
    BatchConsolidationOutcome outcome;
    const int64_t consolidationStart = clock_->nowMs();
    ConsolidationTiming timing;

    const int hot_count = repo_.getHotMessageCount(sessionId);
    if (hot_count <= 0) {
        return outcome;
    }

    int to_archive = static_cast<int>(policy_.prune_batch_size);
    if (to_archive > hot_count) {
        to_archive = hot_count;
    }

    auto batch = repo_.getOldestMessages(sessionId, to_archive);
    if (batch.empty()) {
        return outcome;
    }

    DecisionTraceLogger logger;
    DecisionTrace trace = logger.startTrace("memory_consolidation", hot_count);

    logger.addStage(trace, "consolidation_source", true, "Consolidation initiated", {
        {"session_id", sessionId},
        {"source", consolidationSourceToString(source)},
        {"requested_by", requested_by.empty() ? "SYSTEM" : requested_by},
        {"decision", consolidationDecisionToJson(decision)}
    });

    logger.addStage(trace, "policy_evaluated", true, "Consolidation policy evaluated", {
        {"session_id", sessionId},
        {"decision", consolidationDecisionToJson(decision)}
    });

    MemoryRepository::MemoryConsolidationRequest request;
    request.session_id = sessionId;
    request.messages_to_archive = batch;

    const std::string digest = batchDigest(batch);
    const int64_t ts_start = batch.front().timestamp_ms;
    const int64_t ts_end = batch.back().timestamp_ms;
    bool warm_will_commit = false;

    if (policy_.summarize_before_pruning) {
        const int64_t summaryStart = clock_->nowMs();
        const auto extraction = summaryGenerator_.extract(batch);
        timing.summary_ms = elapsedMs(summaryStart, clock_->nowMs());

        MemoryRepository::WarmMemoryRecord warm;
        warm.id = sessionId + "-warm-" + std::to_string(ts_start);
        warm.session_id = sessionId;
        warm.scope = MemoryScope::SESSION;
        warm.episodic_payload = extraction.memory.serialize();
        warm.rendered_summary = renderEpisodicMemory(extraction.memory);
        warm.importance = extraction.memory.importance;
        warm.novelty = extraction.memory.novelty;
        warm.confidence = extraction.memory.confidence;
        warm.covered_turn_start = 0;
        warm.covered_turn_end = static_cast<int>(batch.size()) - 1;
        warm.covered_ts_start = ts_start;
        warm.covered_ts_end = ts_end;
        warm.derived_from_hash = digest;
        warm.summary_version = MemoryConsolidation::kSummaryVersion;
        warm.prompt_version = extraction.prompt_version;
        warm.llm_model = extraction.llm_model;
        warm.summary_missing = !extraction.llm_success || !extraction.parse_success;
        warm.created_at_ms = clock_->nowMs();
        warm.embedding_version = MemoryConsolidation::kWarmMemoryEmbeddingVersion;

        if (warm.summary_missing) {
            warm.rendered_summary = "Episodic summary unavailable; raw turns archived.";
            warm.episodic_payload = EpisodicMemory{}.serialize();
        }

        if (!embeddingEngine_) {
            timing.consolidation_ms = elapsedMs(consolidationStart, clock_->nowMs());
            recordConsolidationTiming(timing);
            logger.addStage(trace, "consolidation_failed", false, "Embedding engine unavailable", {
                {"session_id", sessionId},
                {"summary_ms", timing.summary_ms},
                {"consolidation_ms", timing.consolidation_ms}
            });
            logger.finishTrace(trace, false, "Consolidation aborted — hot tier unchanged");
            logger.writeTrace(trace);
            return outcome;
        }

        const EpisodicMemory episodic = extraction.memory;
        const std::string embedText = warm.summary_missing
            ? std::string("summary_missing\n") + digest
            : episodic.toCanonicalEmbedText();

        const int64_t embedStart = clock_->nowMs();
        if (mockEmbedEmpty()) {
            warm.embedding.clear();
        } else {
            warm.embedding = embeddingEngine_->embed(embedText);
        }
        timing.embed_ms = elapsedMs(embedStart, clock_->nowMs());

        if (warm.embedding.empty()) {
            timing.consolidation_ms = elapsedMs(consolidationStart, clock_->nowMs());
            recordConsolidationTiming(timing);
            logger.addStage(trace, "consolidation_failed", false, "Embedding generation failed", {
                {"session_id", sessionId},
                {"summary_ms", timing.summary_ms},
                {"embed_ms", timing.embed_ms},
                {"consolidation_ms", timing.consolidation_ms}
            });
            logger.finishTrace(trace, false, "Consolidation aborted — hot tier unchanged");
            logger.writeTrace(trace);
            return outcome;
        }

        request.warm = warm;
        warm_will_commit = true;

        logger.addStage(trace, "episodic_extracted", true, "Episodic memory extracted", {
            {"session_id", sessionId},
            {"summary_missing", warm.summary_missing},
            {"importance", warm.importance},
            {"confidence", warm.confidence},
            {"summary_ms", timing.summary_ms},
            {"embed_ms", timing.embed_ms}
        });
    }

    const int64_t txnStart = clock_->nowMs();
    const bool success = repo_.consolidateSessionBatch(request);
    timing.transaction_ms = elapsedMs(txnStart, clock_->nowMs());
    timing.consolidation_ms = elapsedMs(consolidationStart, clock_->nowMs());
    recordConsolidationTiming(timing);

    if (success) {
        outcome.archived = static_cast<int>(batch.size());
        outcome.warm_created = warm_will_commit ? 1 : 0;
        logger.addStage(trace, "consolidation_committed", true,
                        "Consolidated " + std::to_string(batch.size()) + " turns", {
            {"session_id", sessionId},
            {"turns_consolidated", static_cast<int>(batch.size())},
            {"remaining_hot", hot_count - static_cast<int>(batch.size())},
            {"derived_from_hash", digest},
            {"decision", consolidationDecisionToJson(decision)},
            {"source", consolidationSourceToString(source)},
            {"requested_by", requested_by.empty() ? "SYSTEM" : requested_by},
            {"summary_ms", timing.summary_ms},
            {"embed_ms", timing.embed_ms},
            {"transaction_ms", timing.transaction_ms},
            {"consolidation_ms", timing.consolidation_ms}
        });
        logger.finishTrace(trace, true, "Memory consolidation completed");
    } else {
        logger.addStage(trace, "consolidation_failed", false, "Database consolidation failed", {
            {"session_id", sessionId},
            {"summary_ms", timing.summary_ms},
            {"embed_ms", timing.embed_ms},
            {"transaction_ms", timing.transaction_ms},
            {"consolidation_ms", timing.consolidation_ms}
        });
        logger.finishTrace(trace, false, "Consolidation rolled back — hot tier unchanged");
    }

    logger.writeTrace(trace);
    return outcome;
}

ConsolidationResult MemoryPruner::runConsolidation(const std::string& sessionId,
                                                   const ConsolidationRequest& request) {
    ConsolidationResult result;
    result.source = request.source;
    result.decision = evaluatePolicy(sessionId);

    if (!shouldEnterConsolidation(result.decision, request)) {
        result.remaining_hot = result.decision.hot_count;
        finalizeResultCompat(result);
        return result;
    }

    const std::string requested_by = request.requested_by.empty() ? "SYSTEM" : request.requested_by;

    if (request.single_batch) {
        const auto batch = consolidateOneBatchInternal(
            sessionId, result.decision, request.source, requested_by);
        result.archived = batch.archived;
        result.warm_created = batch.warm_created;
        result.batches = batch.archived > 0 ? 1 : 0;
        result.decision = evaluatePolicy(sessionId);
        result.remaining_hot = result.decision.hot_count;
        finalizeResultCompat(result);
        return result;
    }

    for (size_t batch_idx = 0; batch_idx < policy_.max_batches_per_invocation; ++batch_idx) {
        result.decision = evaluatePolicy(sessionId);
        if (!shouldEnterConsolidation(result.decision, request)) {
            break;
        }

        const auto batch = consolidateOneBatchInternal(
            sessionId, result.decision, request.source, requested_by);
        if (batch.archived <= 0) {
            break;
        }

        result.archived += batch.archived;
        result.warm_created += batch.warm_created;
        result.batches++;
    }

    result.decision = evaluatePolicy(sessionId);
    result.remaining_hot = result.decision.hot_count;

    if (shouldEnterConsolidation(result.decision, request)) {
        result.deferred = true;
        DecisionTraceLogger logger;
        DecisionTrace trace = logger.startTrace("memory_consolidation", result.decision.hot_count);
        logger.addStage(trace, "consolidation_deferred", true,
                        "Consolidation paused — batch cap reached. "
                        "Remaining stale messages will be consolidated on next access.", {
            {"session_id", sessionId},
            {"batches_completed", result.batches},
            {"total_archived", result.archived},
            {"remaining_hot", result.remaining_hot},
            {"decision", consolidationDecisionToJson(result.decision)},
            {"source", consolidationSourceToString(request.source)},
            {"requested_by", requested_by}
        });
        logger.finishTrace(trace, true, "Consolidation deferred");
        logger.writeTrace(trace);
    }

    finalizeResultCompat(result);
    return result;
}

int MemoryPruner::consolidateOneBatch(const std::string& sessionId) {
    const auto decision = evaluatePolicy(sessionId);
    if (!decision.shouldConsolidate()) {
        return 0;
    }
    return consolidateOneBatchInternal(
        sessionId, decision, ConsolidationSource::AUTOMATIC, "SYSTEM").archived;
}

ConsolidationResult MemoryPruner::consolidateIfNeeded(const std::string& sessionId) {
    ConsolidationRequest request;
    request.source = ConsolidationSource::AUTOMATIC;
    request.requested_by = "SYSTEM";
    return runConsolidation(sessionId, request);
}

int MemoryPruner::prune(const std::string& sessionId) {
    return consolidateIfNeeded(sessionId).archived;
}

std::vector<MemoryRepository::ArchivedTurnRecord> MemoryPruner::restore(const std::string& sessionId) {
    return repo_.getArchivedMessages(sessionId);
}

} // namespace Thoth
