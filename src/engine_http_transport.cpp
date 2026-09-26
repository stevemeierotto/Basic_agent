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
#include "../include/conversation_authority.h"
#include "../include/research_resources.h"
#include "../include/graph_statistics.h"
#include "../include/corpus_create.h"
#include "../include/corpus_documents.h"
#include "../include/runtime_bootstrap.h"

#include <httplib.h>
#include <json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <optional>
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
            body["embedding"] = embeddingProbeJson(getLastEmbeddingProbeSnapshot());
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
            std::optional<std::string> active_goal;
            if (body.contains("active_goal") && body["active_goal"].is_string()) {
                active_goal = body["active_goal"].get<std::string>();
            }

            handleEngineFuture(runtime.submitChat(resolved_session, text, active_goal),
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

        server.Get("/v1/diagnostics/latest-decision",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       setJsonResponse(res, 200, runtime.getLatestDecisionSummary());
                   });

        server.Get("/v1/rag/corpus",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       setJsonResponse(res, 200, runtime.listCorpusDocuments());
                   });

        server.Post("/v1/rag/documents",
                    [this](const httplib::Request& req, httplib::Response& res) {
                        if (rejectIfShuttingDown(res)) {
                            return;
                        }
                        if (!runtime.isReady()) {
                            setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                            return;
                        }

                        nlohmann::json body;
                        if (!parseJsonBody(req.body, body, res)) {
                            return;
                        }
                        if (!body.contains("content") || !body["content"].is_string()) {
                            setErrorResponse(res,
                                             EngineError::invalidRequest(
                                                 "Field \"content\" is required."));
                            return;
                        }
                        std::string suggested_name;
                        if (body.contains("name")) {
                            if (!body["name"].is_string()) {
                                setErrorResponse(res,
                                                 EngineError::invalidRequest(
                                                     "Field \"name\" must be a string."));
                                return;
                            }
                            suggested_name = body["name"].get<std::string>();
                        }
                        const std::string content = body["content"].get<std::string>();

                        std::string owner_context_id;
                        if (body.contains("session_id")) {
                            if (!body["session_id"].is_string()) {
                                setErrorResponse(res,
                                                 EngineError::invalidRequest(
                                                     "Field \"session_id\" must be a string."));
                                return;
                            }
                            owner_context_id = body["session_id"].get<std::string>();
                            const auto trim_bounds = [&owner_context_id]() {
                                const auto start = owner_context_id.find_first_not_of(" \t\r\n");
                                if (start == std::string::npos) {
                                    owner_context_id.clear();
                                    return;
                                }
                                const auto end = owner_context_id.find_last_not_of(" \t\r\n");
                                owner_context_id =
                                    owner_context_id.substr(start, end - start + 1);
                            };
                            trim_bounds();
                        }

                        Thoth::CorpusCreate::CreateDocumentRequest request;
                        request.suggested_name = suggested_name;
                        request.content = content;
                        request.owner_context_id = owner_context_id;

                        if (body.contains("content_hash") && body["content_hash"].is_string()) {
                            request.content_hash = body["content_hash"].get<std::string>();
                        }
                        if (body.contains("local_source_mtime")) {
                            if (body["local_source_mtime"].is_number_integer()) {
                                request.local_source_mtime_sec =
                                    body["local_source_mtime"].get<std::int64_t>();
                            } else if (body["local_source_mtime"].is_number_unsigned()) {
                                request.local_source_mtime_sec = static_cast<std::int64_t>(
                                    body["local_source_mtime"].get<std::uint64_t>());
                            }
                        }
                        if (body.contains("local_source_path")
                            && body["local_source_path"].is_string()) {
                            request.local_source_path =
                                body["local_source_path"].get<std::string>();
                        }
                        if (body.contains("force_replace") && body["force_replace"].is_boolean()) {
                            request.force_replace = body["force_replace"].get<bool>();
                        }
                        if (body.contains("dry_run") && body["dry_run"].is_boolean()) {
                            request.dry_run = body["dry_run"].get<bool>();
                        }

                        try {
                            setJsonResponse(res,
                                            200,
                                            runtime.createCorpusDocument(request));
                        } catch (const EngineException& ex) {
                            setErrorResponse(res, ex.error());
                        } catch (const std::exception& ex) {
                            setErrorResponse(res, EngineError::internalError(ex.what()));
                        } catch (...) {
                            setErrorResponse(res,
                                             EngineError::internalError("Unknown engine error."));
                        }
                    });

        server.Post(Thoth::CorpusCreate::kHttpPathSessionLinkRemove,
                    [this](const httplib::Request& req, httplib::Response& res) {
                        if (rejectIfShuttingDown(res)) {
                            return;
                        }
                        if (!runtime.isReady()) {
                            setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                            return;
                        }
                        nlohmann::json body;
                        if (!parseJsonBody(req.body, body, res)) {
                            return;
                        }
                        if (!body.contains("document_id") || !body["document_id"].is_string()
                            || !body.contains("session_id") || !body["session_id"].is_string()) {
                            setErrorResponse(res,
                                             EngineError::invalidRequest(
                                                 "Fields \"document_id\" and \"session_id\" "
                                                 "are required."));
                            return;
                        }
                        const std::string document_id = body["document_id"].get<std::string>();
                        const std::string session_id = body["session_id"].get<std::string>();
                        try {
                            setJsonResponse(
                                res, 200, runtime.unlinkSessionDocument(document_id, session_id));
                        } catch (const EngineException& ex) {
                            setErrorResponse(res, ex.error());
                        } catch (const std::exception& ex) {
                            setErrorResponse(res, EngineError::internalError(ex.what()));
                        } catch (...) {
                            setErrorResponse(res,
                                             EngineError::internalError("Unknown engine error."));
                        }
                    });

        server.Post("/v1/conversation/sessions",
                    [this](const httplib::Request&, httplib::Response& res) {
                        if (rejectIfShuttingDown(res)) {
                            return;
                        }
                        if (!runtime.isReady()) {
                            setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                            return;
                        }
                        try {
                            setJsonResponse(res, 200, runtime.createConversationSession());
                        } catch (const EngineException& ex) {
                            setErrorResponse(res, ex.error());
                        } catch (const std::exception& ex) {
                            setErrorResponse(res, EngineError::internalError(ex.what()));
                        } catch (...) {
                            setErrorResponse(res,
                                             EngineError::internalError("Unknown engine error."));
                        }
                    });

        server.Post("/v1/conversation/turns",
                    [this](const httplib::Request& req, httplib::Response& res) {
                        if (rejectIfShuttingDown(res)) {
                            return;
                        }
                        if (!runtime.isReady()) {
                            setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                            return;
                        }
                        nlohmann::json body;
                        if (!parseJsonBody(req.body, body, res)) {
                            return;
                        }
                        if (!body.contains("session_id") || !body["session_id"].is_string()) {
                            setErrorResponse(res,
                                             EngineError::invalidRequest(
                                                 "Field \"session_id\" is required."));
                            return;
                        }
                        if (!body.contains("content") || !body["content"].is_string()) {
                            setErrorResponse(res,
                                             EngineError::invalidRequest(
                                                 "Field \"content\" is required."));
                            return;
                        }
                        const std::string session_id = body["session_id"].get<std::string>();
                        const std::string content = body["content"].get<std::string>();
                        std::optional<std::string> active_goal;
                        if (body.contains("active_goal") && body["active_goal"].is_string()) {
                            active_goal = body["active_goal"].get<std::string>();
                        }
                        try {
                            setJsonResponse(res, 200,
                                            runtime.appendUserTurn(session_id, content, active_goal));
                        } catch (const EngineException& ex) {
                            setErrorResponse(res, ex.error());
                        } catch (const std::exception& ex) {
                            setErrorResponse(res, EngineError::internalError(ex.what()));
                        } catch (...) {
                            setErrorResponse(res,
                                             EngineError::internalError("Unknown engine error."));
                        }
                    });

        server.Get(R"(/v1/conversation/sessions/([A-Za-z0-9._-]+)/summary)",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       const std::string session_id = req.matches[1];
                       try {
                           setJsonResponse(res, 200,
                                           runtime.getConversationSummaryForSession(session_id));
                       } catch (const EngineException& ex) {
                           setErrorResponse(res, ex.error());
                       } catch (const std::exception& ex) {
                           setErrorResponse(res, EngineError::internalError(ex.what()));
                       }
                   });

        server.Get(R"(/v1/conversation/sessions/([A-Za-z0-9._-]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       const std::string session_id = req.matches[1];
                       try {
                           setJsonResponse(res, 200,
                                           runtime.getConversationForSession(session_id));
                       } catch (const EngineException& ex) {
                           setErrorResponse(res, ex.error());
                       } catch (const std::exception& ex) {
                           setErrorResponse(res, EngineError::internalError(ex.what()));
                       }
                   });

        server.Get("/v1/research/strategies",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       setJsonResponse(res, 200, runtime.listStrategies());
                   });

        server.Get("/v1/research/trajectories",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       setJsonResponse(res, 200, runtime.listTrajectories());
                   });

        server.Get("/v1/research/episodes",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       setJsonResponse(res, 200, runtime.listEpisodes());
                   });

        server.Get("/v1/graph/stats",
                   [this](const httplib::Request&, httplib::Response& res) {
                       if (rejectIfShuttingDown(res)) {
                           return;
                       }
                       if (!runtime.isReady()) {
                           setErrorResponse(res, EngineError::engineBusy("Engine is not ready."));
                           return;
                       }
                       setJsonResponse(res, 200, runtime.getGraphStatisticsResource());
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
