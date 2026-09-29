/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — diagnostic llama.cpp /slots observer for MTCP v1.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "inference_http.h"

#include <chrono>
#include <functional>
#include <string>

namespace Thoth {

class GenerationProgressIds {
public:
    static void set(const std::string& generation_id);
    static std::string current();
    static void clear();
};

struct ProgressObserverTestHooks {
    std::function<void(std::chrono::milliseconds)> wait_for;
    std::function<InferenceHttpResponse()> fetch;
};

void setProgressObserverTestHooks(ProgressObserverTestHooks hooks);
void clearProgressObserverTestHooks();

/** Joins on destruction. Does not change the completion request. */
class GenerationProgressSession {
public:
    GenerationProgressSession(std::string generation_id, std::string slots_url);
    ~GenerationProgressSession();

    GenerationProgressSession(const GenerationProgressSession&) = delete;
    GenerationProgressSession& operator=(const GenerationProgressSession&) = delete;

private:
    struct State;
    State* state_ = nullptr;
};

} // namespace Thoth
