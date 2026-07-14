/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — HTTP transport adapter for EngineRuntime (Plan F)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_http_transport.h"

#include "../include/engine_error.h"
#include "../include/engine_runtime.h"
#include "../include/engine_sse_session.h"

#include <httplib.h>
#include <json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>

#ifndef THOTH_ENGINE_VERSION
#define THOTH_ENGINE_VERSION "0.2"
#endif

#ifndef THOTH_GIT_SHA
#define THOTH_GIT_SHA "unknown"
#endif

namespace Thoth {

namespace {

nlohmann::json capabilitiesJson(const EngineRuntime& runtime) {
    nlohmann::json caps = nlohmann::json::array();
    for (const std::string& capability : runtime.capabilities()) {
        caps.push_back(capability);
    }
    return caps;
}

void setJsonResponse(httplib::Response& res, int status, const nlohmann::json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void setErrorResponse(httplib::Response& res, const EngineError& error) {
    res.status = engineErrorHttpStatus(error.code);
    res.set_content(error.toJson(), "application/json");
}

void handleEngineFuture(std::future<std::string> future,
                        httplib::Response& res,
                        const std::function<void(const std::string&)>& onSuccess) {
    try {
        onSuccess(future.get());
    } catch (const EngineException& ex) {
        setErrorResponse(res, ex.error());
    } catch (const std::exception& ex) {
        setErrorResponse(res, EngineError::internalError(ex.what()));
    } catch (...) {
        setErrorResponse(res, EngineError::internalError("Unknown engine error."));
    }
}

bool parseJsonBody(const std::string& raw,
                   nlohmann::json& body,
                   httplib::Response& res) {
    try {
        body = nlohmann::json::parse(raw.empty() ? "{}" : raw);
        return true;
    } catch (...) {
        setErrorResponse(res,
                         EngineError::invalidRequest("Request body must be valid JSON."));
        return false;
    }
}

bool parseSessionIdField(const nlohmann::json& body,
                         std::string& session_id,
                         httplib::Response& res) {
    if (!body.contains("session_id")) {
        return true;
    }
    if (!body["session_id"].is_string()) {
        setErrorResponse(res,
                         EngineError::invalidRequest("Field \"session_id\" must be a string."));
        return false;
    }
    session_id = body["session_id"].get<std::string>();
    return true;
}

void handleControlPost(EngineRuntime& runtime,
                        httplib::Response& res,
                        const std::function<bool()>& reject,
                        const std::function<void()>& action) {
    if (reject()) {
        return;
    }
    if (!runtime.isReady()) {
        setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
        return;
    }
    action();
    setJsonResponse(res, 200, nlohmann::json{{"status", "ok"}});
}

} // namespace

EngineHttpConfig engineHttpConfigFromEnvironment() {
    EngineHttpConfig config;
    if (const char* bind = std::getenv("THOTH_ENGINE_BIND")) {
        if (bind[0] != '\0') {
            config.bind_host = bind;
        }
    }
    if (const char* port = std::getenv("THOTH_ENGINE_PORT")) {
        if (port[0] != '\0') {
            try {
                config.port = std::stoi(port);
            } catch (...) {
            }
        }
    }
    return config;
}

struct EngineHttpTransport::Impl {
    EngineRuntime& runtime;
    EngineHttpConfig config;
    std::atomic<bool> shutting_down{false};
    std::mutex server_mutex;
    httplib::Server* active_server{nullptr};
    SseSessionManager sse_manager;

    explicit Impl(EngineRuntime& runtime_in, EngineHttpConfig config_in)
        : runtime(runtime_in), config(std::move(config_in)), sse_manager(runtime_in) {}

    bool rejectIfShuttingDown(httplib::Response& res) const {
        if (shutting_down.load() || !runtime.isReady()) {
            setErrorResponse(res, EngineError::engineBusy("Engine is shutting down."));
            return true;
        }
        return false;
    }

    void registerRoutes(httplib::Server& server) {
        server.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            if (rejectIfShuttingDown(res)) {
                return;
            }
            setJsonResponse(res, 200, nlohmann::json{{"status", "ok"}});
        });

        server.Get("/ready", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json body;
            body["workspace"] = runtime.workspacePath();
            body["capabilities"] = capabilitiesJson(runtime);
            if (runtime.isReady()) {
                body["status"] = "ready";
                setJsonResponse(res, 200, body);
            } else {
                body["status"] = "not_ready";
                setJsonResponse(res, 503, body);
            }
        });

        server.Get("/version", [](const httplib::Request&, httplib::Response& res) {
            setJsonResponse(res,
                            200,
                            nlohmann::json{{"engine", THOTH_ENGINE_VERSION},
                                           {"git", THOTH_GIT_SHA},
                                           {"protocol", "v1"}});
        });

        server.Post("/v1/chat", [this](const httplib::Request& req, httplib::Response& res) {
            if (rejectIfShuttingDown(res)) {
                return;
            }

            nlohmann::json body;
            try {
                body = nlohmann::json::parse(req.body);
            } catch (...) {
                setErrorResponse(res,
                                 EngineError::invalidRequest("Request body must be valid JSON."));
                return;
            }

            if (!body.contains("text") || !body["text"].is_string()) {
                setErrorResponse(res, EngineError::invalidRequest("Field \"text\" is required."));
                return;
            }

            std::string session_id;
            if (!parseSessionIdField(body, session_id, res)) {
                return;
            }

            const std::string text = body["text"].get<std::string>();
            const std::string resolved_session = normalizeEngineSessionId(session_id);

            handleEngineFuture(runtime.submitChat(resolved_session, text),
                               res,
                               [&](const std::string& response) {
                                   setJsonResponse(res,
                                                   200,
                                                   nlohmann::json{{"response", response},
                                                                  {"session_id", resolved_session}});
                               });
        });

        server.Post("/v1/goals", [this](const httplib::Request& req, httplib::Response& res) {
            if (rejectIfShuttingDown(res)) {
                return;
            }

            nlohmann::json body;
            if (!parseJsonBody(req.body, body, res)) {
                return;
            }

            if (!body.contains("goal") || !body["goal"].is_string()) {
                setErrorResponse(res, EngineError::invalidRequest("Field \"goal\" is required."));
                return;
            }

            std::string session_id;
            if (!parseSessionIdField(body, session_id, res)) {
                return;
            }

            const std::string goal = body["goal"].get<std::string>();
            const std::string resolved_session = normalizeEngineSessionId(session_id);

            handleEngineFuture(runtime.submitGoal(resolved_session, goal),
                               res,
                               [&](const std::string& message) {
                                   setJsonResponse(res,
                                                   200,
                                                   nlohmann::json{{"status", "accepted"},
                                                                  {"message", message}});
                               });
        });

        server.Post("/v1/control/pause",
                    [this](const httplib::Request&, httplib::Response& res) {
                        handleControlPost(
                            runtime,
                            res,
                            [this, &res]() { return rejectIfShuttingDown(res); },
                            [this]() { runtime.pause(); });
                    });

        server.Post("/v1/control/resume",
                    [this](const httplib::Request&, httplib::Response& res) {
                        handleControlPost(
                            runtime,
                            res,
                            [this, &res]() { return rejectIfShuttingDown(res); },
                            [this]() { runtime.resume(); });
                    });

        server.Post("/v1/control/abort",
                    [this](const httplib::Request&, httplib::Response& res) {
                        handleControlPost(
                            runtime,
                            res,
                            [this, &res]() { return rejectIfShuttingDown(res); },
                            [this]() { runtime.abort(); });
                    });

        server.Get("/v1/events", [this](const httplib::Request& req, httplib::Response& res) {
            if (rejectIfShuttingDown(res)) {
                return;
            }

            (void)req.get_header_value("Last-Event-ID");

            auto session = sse_manager.createSession();
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_chunked_content_provider(
                "text/event-stream",
                [session](size_t, httplib::DataSink& sink) {
                    if (sink.is_writable && !sink.is_writable()) {
                        session->close();
                        return false;
                    }

                    std::string chunk;
                    bool closed = false;
                    if (session->waitForChunk(chunk,
                                               std::chrono::milliseconds(500),
                                               closed)) {
                        if (!sink.write(chunk.data(), chunk.size())) {
                            session->close();
                            return false;
                        }
                        return true;
                    }

                    if (closed) {
                        return false;
                    }

                    static constexpr char kKeepAlive[] = ": keepalive\n\n";
                    if (!sink.write(kKeepAlive, sizeof(kKeepAlive) - 1)) {
                        session->close();
                        return false;
                    }
                    return true;
                },
                [this, session](bool) { sse_manager.removeSession(session); });
        });

        server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
            if (res.status != 404) {
                return;
            }
            setErrorResponse(res,
                             EngineError::notFound("Route not found: " + req.path));
        });
    }

    void stopServer() {
        std::lock_guard<std::mutex> lock(server_mutex);
        if (active_server) {
            active_server->stop();
        }
    }
};

EngineHttpTransport::EngineHttpTransport(EngineRuntime& runtime, EngineHttpConfig config)
    : impl_(new Impl{runtime, std::move(config)}) {}

EngineHttpTransport::~EngineHttpTransport() {
    requestStop();
    delete impl_;
}

bool EngineHttpTransport::run() {
    httplib::Server server;
    impl_->registerRoutes(server);

    {
        std::lock_guard<std::mutex> lock(impl_->server_mutex);
        impl_->active_server = &server;
    }

    const bool listened =
        server.listen(impl_->config.bind_host.c_str(), impl_->config.port);

    {
        std::lock_guard<std::mutex> lock(impl_->server_mutex);
        impl_->active_server = nullptr;
    }

    return listened;
}

void EngineHttpTransport::requestStop() {
    impl_->shutting_down = true;
    impl_->sse_manager.closeAll();
    impl_->runtime.beginShutdown();
    impl_->stopServer();
}

std::size_t EngineHttpTransport::sseSessionCountForTests() const {
    return impl_->sse_manager.sessionCountForTests();
}

} // namespace Thoth
