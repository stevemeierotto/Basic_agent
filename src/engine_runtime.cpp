/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — EngineRuntime service layer (Plan F / Plan G)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_runtime.h"

#include "../include/basic_agent_plugin.h"
#include "../include/chat_turn_timing.h"
#include "../include/decision_summary.h"
#include "../include/conversation_authority.h"
#include "../include/research_resources.h"
#include "../include/graph_statistics.h"
#include "../include/corpus_documents.h"
#include "../include/corpus_create.h"
#include "../include/engine_error.h"
#include "../include/file_handler.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Thoth {

namespace {

std::future<std::string> rejectWithError(const EngineError& error) {
    auto promise = std::make_shared<std::promise<std::string>>();
    promise->set_exception(std::make_exception_ptr(EngineException(error)));
    return promise->get_future();
}

int64_t nowUtcMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace

struct EngineRuntime::Impl {
    static constexpr std::size_t kEventIngressCapacity = 4096;

    std::unique_ptr<BasicAgentPlugin> plugin;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<std::function<void()>> tasks;
    bool shutting_down{false};
    bool ready{false};
    std::string active_session_id{"default"};
    std::unordered_set<std::string> known_sessions{"default"};

    std::deque<ControllerEvent> event_ingress;
    std::mutex event_ingress_mutex;
    std::condition_variable event_ingress_cv;
    std::thread dispatch_thread;
    bool dispatch_running{false};

    std::mutex subscribers_mutex;
    std::unordered_map<uint64_t, std::function<void(const EngineEvent&)>> subscribers;
    std::atomic<uint64_t> next_subscription_id{1};
    std::atomic<uint64_t> next_sequence{0};
    std::atomic<uint64_t> last_sequence_for_tests{0};
    std::atomic<std::size_t> dropped_event_count{0};

    void startWorker() {
        worker = std::thread([this] { workerLoop(); });
    }

    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return shutting_down || !tasks.empty(); });
                if (shutting_down && tasks.empty()) {
                    return;
                }
                task = std::move(tasks.front());
                tasks.pop();
            }
            if (task) {
                task();
            }
        }
    }

    void enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (shutting_down) {
                throw EngineException(EngineError::engineBusy("Engine is shutting down."));
            }
            tasks.push(std::move(task));
        }
        cv.notify_one();
    }

    void ensureSessionOnWorker(const std::string& session_id) {
        known_sessions.insert(session_id);
        if (active_session_id != session_id) {
            active_session_id = session_id;
            if (plugin) {
                plugin->setSessionId(session_id);
            }
        }
    }

    void enqueueControllerEvent(const ControllerEvent& event) {
        {
            std::lock_guard<std::mutex> lock(event_ingress_mutex);
            if (event_ingress.size() >= kEventIngressCapacity) {
                dropped_event_count.fetch_add(1, std::memory_order_relaxed);
                std::cerr << "[EngineRuntime] event ingress full; dropping event\n";
                return;
            }
            event_ingress.push_back(event);
        }
        event_ingress_cv.notify_one();
    }

    void startDispatchThread() {
        dispatch_running = true;
        dispatch_thread = std::thread([this] { dispatchLoop(); });
    }

    void dispatchLoop() {
        while (true) {
            ControllerEvent raw;
            {
                std::unique_lock<std::mutex> lock(event_ingress_mutex);
                event_ingress_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return !event_ingress.empty() || !dispatch_running;
                });
                if (event_ingress.empty()) {
                    if (!dispatch_running) {
                        return;
                    }
                    continue;
                }
                raw = std::move(event_ingress.front());
                event_ingress.pop_front();
            }

            const uint64_t sequence = next_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
            last_sequence_for_tests.store(sequence, std::memory_order_relaxed);
            const EngineEvent envelope = makeEngineEvent(raw, sequence, nowUtcMs());

            std::vector<std::function<void(const EngineEvent&)>> handlers;
            {
                std::lock_guard<std::mutex> lock(subscribers_mutex);
                handlers.reserve(subscribers.size());
                for (const auto& entry : subscribers) {
                    handlers.push_back(entry.second);
                }
            }

            for (const auto& handler : handlers) {
                if (!handler) {
                    continue;
                }
                try {
                    handler(envelope);
                } catch (const std::exception& ex) {
                    std::cerr << "[EngineRuntime] event subscriber failure: " << ex.what() << '\n';
                } catch (...) {
                    std::cerr << "[EngineRuntime] event subscriber failure: unknown\n";
                }
            }
        }
    }

    uint64_t subscribeEvents(std::function<void(const EngineEvent&)> handler) {
        const uint64_t id = next_subscription_id.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(subscribers_mutex);
        subscribers[id] = std::move(handler);
        return id;
    }

    void unsubscribeEvents(uint64_t subscription_id) {
        std::lock_guard<std::mutex> lock(subscribers_mutex);
        subscribers.erase(subscription_id);
    }

    void drainWorker(std::chrono::milliseconds drain_timeout) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutting_down = true;
            ready = false;
        }
        cv.notify_all();

        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!cv.wait_for(lock, drain_timeout, [this] { return tasks.empty(); })) {
                std::cerr << "[EngineRuntime] shutdown: drain timeout with " << tasks.size()
                          << " queued task(s); waiting for worker\n";
            }
        }

        if (worker.joinable()) {
            worker.join();
        }
    }

    void drainEventIngress(std::chrono::milliseconds drain_timeout) {
        const auto deadline = std::chrono::steady_clock::now() + drain_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(event_ingress_mutex);
                if (event_ingress.empty()) {
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void stopDispatchThread(std::chrono::milliseconds drain_timeout) {
        drainEventIngress(drain_timeout);
        dispatch_running = false;
        event_ingress_cv.notify_all();
        if (dispatch_thread.joinable()) {
            dispatch_thread.join();
        }
        std::lock_guard<std::mutex> lock(subscribers_mutex);
        subscribers.clear();
    }
};

EngineRuntime::EngineRuntime() : impl_(std::make_unique<Impl>()) {}

EngineRuntime::~EngineRuntime() {
    shutdown();
}

std::unique_ptr<EngineRuntime> EngineRuntime::create() {
    auto runtime = std::unique_ptr<EngineRuntime>(new EngineRuntime());
    runtime->impl_->startDispatchThread();
    runtime->impl_->plugin = std::make_unique<BasicAgentPlugin>();
    runtime->impl_->plugin->onEvent = [impl = runtime->impl_.get()](const ControllerEvent& event) {
        impl->enqueueControllerEvent(event);
    };
    runtime->impl_->ready = true;
    runtime->impl_->startWorker();
    return runtime;
}

void EngineRuntime::beginShutdown() {
    if (!impl_) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->shutting_down = true;
    impl_->ready = false;
}

void EngineRuntime::shutdown(std::chrono::milliseconds drain_timeout) {
    if (!impl_) {
        return;
    }

    beginShutdown();

    if (impl_->worker.joinable()) {
        impl_->drainWorker(drain_timeout);
    } else {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->shutting_down = true;
        impl_->ready = false;
    }

    impl_->plugin.reset();
    impl_->stopDispatchThread(std::chrono::seconds(2));
}

std::future<std::string> EngineRuntime::submitChat(const std::string& session_id,
                                                   const std::string& text,
                                                   const std::optional<std::string>& active_goal) {
    if (!isReady()) {
        return rejectWithError(EngineError::engineBusy("Engine is not ready."));
    }
    if (text.empty()) {
        return rejectWithError(EngineError::invalidRequest("Text cannot be empty."));
    }

    const std::string resolved_session = normalizeEngineSessionId(session_id);
    auto promise = std::make_shared<std::promise<std::string>>();
    std::future<std::string> future = promise->get_future();

    try {
        const std::int64_t enqueued_at_ms = nowUtcMs();
        impl_->enqueue([this, resolved_session, text, active_goal, promise, enqueued_at_ms]() {
            try {
                ChatTurnTiming::setActiveWorkerContext(
                    {enqueued_at_ms, nowUtcMs()});
                impl_->ensureSessionOnWorker(resolved_session);
                promise->set_value(impl_->plugin->processInput(text, active_goal));
            } catch (const std::exception& e) {
                promise->set_exception(
                    std::make_exception_ptr(EngineException(EngineError::internalError(e.what()))));
            } catch (...) {
                promise->set_exception(std::make_exception_ptr(
                    EngineException(EngineError::internalError("Unknown engine error."))));
            }
        });
    } catch (const EngineException& e) {
        return rejectWithError(e.error());
    }

    return future;
}

std::future<std::string> EngineRuntime::submitGoal(const std::string& session_id,
                                                   const std::string& goal) {
    if (!isReady()) {
        return rejectWithError(EngineError::engineBusy("Engine is not ready."));
    }
    if (goal.empty()) {
        return rejectWithError(EngineError::invalidRequest("Goal cannot be empty."));
    }

    const std::string resolved_session = normalizeEngineSessionId(session_id);
    auto promise = std::make_shared<std::promise<std::string>>();
    std::future<std::string> future = promise->get_future();

    try {
        impl_->enqueue([this, resolved_session, goal, promise]() {
            try {
                impl_->ensureSessionOnWorker(resolved_session);
                impl_->plugin->executeGoal(goal);
                promise->set_value("GOAL ACCEPTED: " + goal);
            } catch (const std::exception& e) {
                promise->set_exception(
                    std::make_exception_ptr(EngineException(EngineError::internalError(e.what()))));
            } catch (...) {
                promise->set_exception(std::make_exception_ptr(
                    EngineException(EngineError::internalError("Unknown engine error."))));
            }
        });
    } catch (const EngineException& e) {
        return rejectWithError(e.error());
    }

    return future;
}

void EngineRuntime::pause() {
    if (!isReady()) {
        return;
    }
    try {
        impl_->enqueue([this]() {
            if (impl_->plugin) {
                impl_->plugin->pause();
            }
        });
    } catch (const EngineException&) {
    }
}

void EngineRuntime::resume() {
    if (!isReady()) {
        return;
    }
    try {
        impl_->enqueue([this]() {
            if (impl_->plugin) {
                impl_->plugin->resume();
            }
        });
    } catch (const EngineException&) {
    }
}

void EngineRuntime::abort() {
    if (!isReady()) {
        return;
    }
    try {
        impl_->enqueue([this]() {
            if (impl_->plugin) {
                impl_->plugin->abort();
            }
        });
    } catch (const EngineException&) {
    }
}

uint64_t EngineRuntime::subscribeEvents(std::function<void(const EngineEvent&)> handler) {
    return impl_->subscribeEvents(std::move(handler));
}

void EngineRuntime::unsubscribeEvents(uint64_t subscription_id) {
    impl_->unsubscribeEvents(subscription_id);
}

void EngineRuntime::publishControllerEventForTests(const ControllerEvent& event) {
    impl_->enqueueControllerEvent(event);
}

std::size_t EngineRuntime::eventSubscriberCountForTests() const {
    std::lock_guard<std::mutex> lock(impl_->subscribers_mutex);
    return impl_->subscribers.size();
}

uint64_t EngineRuntime::lastEventSequenceForTests() const {
    return impl_->last_sequence_for_tests.load(std::memory_order_relaxed);
}

std::size_t EngineRuntime::droppedEventCountForTests() const {
    return impl_->dropped_event_count.load(std::memory_order_relaxed);
}

std::string EngineRuntime::workspacePath() const {
    FileHandler fh;
    return fh.getAgentWorkspacePath();
}

bool EngineRuntime::isReady() const {
    return impl_ && impl_->ready && !impl_->shutting_down;
}

std::vector<std::string> EngineRuntime::capabilities() const {
    return {"chat", "goals", "control", "events", "diagnostics", "corpus", "ingest",
            "conversation", "strategies", "trajectories", "episodes", "graph_stats"};
}

nlohmann::json EngineRuntime::getLatestDecisionSummary() const {
    FileHandler fh;
    const std::string path = fh.getAgentWorkspacePath("decision_trace.jsonl");
    return DecisionSummary::loadLatestFromDecisionTraceFile(path);
}

nlohmann::json EngineRuntime::listCorpusDocuments() const {
    if (!impl_ || !impl_->plugin) {
        return CorpusDocuments::emptyV1List();
    }
    return impl_->plugin->listCorpusDocuments();
}

nlohmann::json EngineRuntime::createCorpusDocument(const std::string& suggested_name,
                                                   const std::string& content,
                                                   const std::string& owner_context_id) {
    CorpusCreate::CreateDocumentRequest request;
    request.suggested_name = suggested_name;
    request.content = content;
    request.owner_context_id = owner_context_id;
    return createCorpusDocument(request);
}

nlohmann::json EngineRuntime::createCorpusDocument(
    const CorpusCreate::CreateDocumentRequest& request) {
    if (!isReady()) {
        throw EngineException(EngineError::engineBusy("Engine is not ready."));
    }
    if (!impl_ || !impl_->plugin) {
        throw EngineException(EngineError::engineBusy("Engine plugin not initialized."));
    }
    return impl_->plugin->createCorpusDocument(request);
}

nlohmann::json EngineRuntime::unlinkSessionDocument(const std::string& document_id,
                                                    const std::string& session_id) {
    if (!isReady()) {
        throw EngineException(EngineError::engineBusy("Engine is not ready."));
    }
    if (!impl_ || !impl_->plugin) {
        throw EngineException(EngineError::engineBusy("Engine plugin not initialized."));
    }
    return impl_->plugin->unlinkSessionDocument(document_id, session_id);
}

nlohmann::json EngineRuntime::createConversationSession() {
    if (!isReady()) {
        throw EngineException(EngineError::engineBusy("Engine is not ready."));
    }
    if (!impl_ || !impl_->plugin) {
        throw EngineException(EngineError::engineBusy("Engine plugin not initialized."));
    }
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    std::future<nlohmann::json> future = promise->get_future();
    impl_->enqueue([this, promise]() {
        try {
            nlohmann::json body = impl_->plugin->createConversationSession();
            impl_->known_sessions.insert(body["session_id"].get<std::string>());
            promise->set_value(std::move(body));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    try {
        return future.get();
    } catch (const EngineException&) {
        throw;
    } catch (const std::exception& ex) {
        throw EngineException(EngineError::internalError(ex.what()));
    }
}

nlohmann::json EngineRuntime::appendUserTurn(const std::string& session_id,
                                             const std::string& content,
                                             const std::optional<std::string>& active_goal) {
    if (!isReady()) {
        throw EngineException(EngineError::engineBusy("Engine is not ready."));
    }
    if (!impl_ || !impl_->plugin) {
        throw EngineException(EngineError::engineBusy("Engine plugin not initialized."));
    }
    const std::string resolved = normalizeEngineSessionId(session_id);
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    std::future<nlohmann::json> future = promise->get_future();
    const std::int64_t enqueued_at_ms = nowUtcMs();
    impl_->enqueue([this, promise, resolved, content, active_goal, enqueued_at_ms]() {
        try {
            ChatTurnTiming::setActiveWorkerContext({enqueued_at_ms, nowUtcMs()});
            impl_->ensureSessionOnWorker(resolved);
            promise->set_value(impl_->plugin->appendUserTurn(resolved, content, active_goal));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    try {
        return future.get();
    } catch (const EngineException&) {
        throw;
    } catch (const std::exception& ex) {
        throw EngineException(EngineError::internalError(ex.what()));
    }
}

nlohmann::json EngineRuntime::getConversationForSession(const std::string& session_id) const {
    if (!impl_ || !impl_->plugin) {
        return ConversationAuthority::emptyConversation(session_id);
    }
    const std::string resolved = normalizeEngineSessionId(session_id);
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    std::future<nlohmann::json> future = promise->get_future();
    const_cast<EngineRuntime*>(this)->impl_->enqueue([this, promise, resolved]() {
        try {
            promise->set_value(impl_->plugin->getConversationForSession(resolved));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    try {
        return future.get();
    } catch (const EngineException&) {
        throw;
    } catch (const std::exception& ex) {
        throw EngineException(EngineError::internalError(ex.what()));
    }
}

nlohmann::json EngineRuntime::getConversationSummaryForSession(
    const std::string& session_id) const {
    if (!impl_ || !impl_->plugin) {
        return ConversationAuthority::makeSummaryResponse(session_id, "");
    }
    const std::string resolved = normalizeEngineSessionId(session_id);
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    std::future<nlohmann::json> future = promise->get_future();
    const_cast<EngineRuntime*>(this)->impl_->enqueue([this, promise, resolved]() {
        try {
            promise->set_value(impl_->plugin->getConversationSummaryForSession(resolved));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    try {
        return future.get();
    } catch (const EngineException&) {
        throw;
    } catch (const std::exception& ex) {
        throw EngineException(EngineError::internalError(ex.what()));
    }
}

nlohmann::json EngineRuntime::listStrategies() const {
    if (!impl_ || !impl_->plugin) {
        return ResearchResources::emptyCollection();
    }
    return impl_->plugin->listStrategies();
}

nlohmann::json EngineRuntime::listTrajectories() const {
    if (!impl_ || !impl_->plugin) {
        return ResearchResources::emptyCollection();
    }
    return impl_->plugin->listTrajectories();
}

nlohmann::json EngineRuntime::listEpisodes() const {
    if (!impl_ || !impl_->plugin) {
        return ResearchResources::emptyCollection();
    }
    return impl_->plugin->listEpisodes();
}

nlohmann::json EngineRuntime::getGraphStatisticsResource() const {
    if (!impl_ || !impl_->plugin) {
        return GraphStatistics::emptyResponse(0);
    }
    return impl_->plugin->getGraphStatisticsResource();
}

} // namespace Thoth
