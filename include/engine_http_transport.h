/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — HTTP transport adapter for EngineRuntime (Plan F)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <cstddef>
#include <string>

namespace Thoth {

class EngineRuntime;

struct EngineHttpConfig {
    std::string bind_host{"127.0.0.1"};
    int port{8090};
};

/** Resolves bind/port using env defaults (CLI applied separately in main). */
EngineHttpConfig engineHttpConfigFromEnvironment();

class EngineHttpTransport {
public:
    EngineHttpTransport(EngineRuntime& runtime, EngineHttpConfig config);
    ~EngineHttpTransport();

    EngineHttpTransport(const EngineHttpTransport&) = delete;
    EngineHttpTransport& operator=(const EngineHttpTransport&) = delete;

    /** Blocks until the server stops (signal, requestStop, or listen failure). */
    bool run();

    void requestStop();

    /** Unit tests only — active SSE session count. */
    std::size_t sseSessionCountForTests() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace Thoth
