/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineRuntime service layer (Plan F / Plan G)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "controller_event.h"
#include "engine_event.h"
#include "corpus_create.h"
#include "json.hpp"

#include <chrono>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thoth {

/**
 * Owns BasicAgentPlugin, the worker thread, command queue, and session state.
 * Transports (CLI, HTTP) call into EngineRuntime — not the plugin directly.
 */
class EngineRuntime {
public:
    static std::unique_ptr<EngineRuntime> create();
    ~EngineRuntime();

    EngineRuntime(const EngineRuntime&) = delete;
    EngineRuntime& operator=(const EngineRuntime&) = delete;

    /** Stop accepting work; drain queue (bounded), then tear down plugin and worker. */
    void shutdown(std::chrono::milliseconds drain_timeout = std::chrono::seconds(5));

    /** Marks not-ready for new commands (e.g. HTTP SIGTERM before drain). */
    void beginShutdown();

    std::future<std::string> submitChat(const std::string& session_id,
                                        const std::string& text,
                                        const std::optional<std::string>& active_goal = std::nullopt);
    std::future<std::string> submitGoal(const std::string& session_id, const std::string& goal);
    void pause();
    void resume();
    void abort();

    uint64_t subscribeEvents(std::function<void(const EngineEvent&)> handler);
    void unsubscribeEvents(uint64_t subscription_id);

    /** Unit tests only — enqueue a synthetic ControllerEvent through the ingress path. */
    void publishControllerEventForTests(const ControllerEvent& event);

    std::size_t eventSubscriberCountForTests() const;
    uint64_t lastEventSequenceForTests() const;
    std::size_t droppedEventCountForTests() const;

    std::string workspacePath() const;
    bool isReady() const;
    std::vector<std::string> capabilities() const;

    /**
     * Phase 4 — structured decision summary for Explain Plan.
     * Assembled from Engine workspace traces (storage is an implementation detail).
     */
    nlohmann::json getLatestDecisionSummary() const;

    /**
     * Phase 8 — Engine-owned corpus document list.
     * Storage format is an implementation detail — never part of the GUI API.
     */
    nlohmann::json listCorpusDocuments() const;

    /**
     * Phase 9 — create corpus document (acceptance JSON).
     * Indexing progress is emitted via INDEXING_* events after acceptance.
     */
    nlohmann::json createCorpusDocument(const std::string& suggested_name,
                                        const std::string& content,
                                        const std::string& owner_context_id = "");

    /** ALP-C — extended create request. */
    nlohmann::json createCorpusDocument(const CorpusCreate::CreateDocumentRequest& request);

    /** ALP amend — remove session↔document link. */
    nlohmann::json unlinkSessionDocument(const std::string& document_id,
                                         const std::string& session_id);

    nlohmann::json createConversationSession();
    nlohmann::json appendUserTurn(const std::string& session_id,
                                  const std::string& content,
                                  const std::optional<std::string>& active_goal = std::nullopt);
    nlohmann::json getConversationForSession(const std::string& session_id) const;
    nlohmann::json getConversationSummaryForSession(const std::string& session_id) const;

    nlohmann::json listStrategies() const;
    nlohmann::json listTrajectories() const;
    nlohmann::json listEpisodes() const;

    nlohmann::json getGraphStatisticsResource() const;

private:
    EngineRuntime();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Thoth
