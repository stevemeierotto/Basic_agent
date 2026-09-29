/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — diagnostic llama.cpp /slots observer for MTCP v1.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/generation_progress_observer.h"
#include "../include/generation_call_log.h"
#include "../include/inference_endpoint.h"

#include <../include/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

using json = nlohmann::json;

namespace Thoth {
namespace {

std::mutex id_mutex;
std::string current_generation_id;
ProgressObserverTestHooks test_hooks;
std::mutex hooks_mutex;
std::mutex inflight_mutex;
bool fetch_inflight = false;

bool containsTextKey(const json& value) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            const std::string& key = it.key();
            if (key == "prompt" || key == "generated" || key == "completion" || key == "text"
                || key == "raw_completion" || key == "chain_of_thought") {
                return true;
            }
            if (containsTextKey(it.value())) {
                return true;
            }
        }
    } else if (value.is_array()) {
        for (const auto& item : value) {
            if (containsTextKey(item)) {
                return true;
            }
        }
    }
    return false;
}

json slimSlot(const json& slot) {
    json out = json::object();
    if (slot.contains("is_processing") && slot["is_processing"].is_boolean()) {
        out["is_processing"] = slot["is_processing"];
    }
    for (const char* key : {"n_prompt_tokens", "n_prompt_tokens_processed", "n_prompt_tokens_cache"}) {
        if (slot.contains(key) && slot[key].is_number_integer()) {
            out[key] = slot[key];
        }
    }
    if (slot.contains("next_token") && slot["next_token"].is_object()) {
        json next = json::object();
        const auto& src = slot["next_token"];
        if (src.contains("n_decoded") && src["n_decoded"].is_number_integer()) {
            next["n_decoded"] = src["n_decoded"];
        }
        if (src.contains("n_remain") && src["n_remain"].is_number_integer()) {
            next["n_remain"] = src["n_remain"];
        }
        if (!next.empty()) {
            out["next_token"] = std::move(next);
        }
    }
    return out;
}

std::string progressLogPath() {
    if (const char* override_path = std::getenv("THOTH_GENERATION_PROGRESS_LOG")) {
        if (*override_path) {
            return override_path;
        }
    }
    std::string calls = GenerationCallLog::logFilePath();
    const auto slash = calls.find_last_of('/');
    if (slash == std::string::npos) {
        return "generation_progress.jsonl";
    }
    return calls.substr(0, slash + 1) + "generation_progress.jsonl";
}

void appendProgress(const json& line) {
    static std::mutex write_mutex;
    std::lock_guard<std::mutex> lock(write_mutex);
    std::ofstream out(progressLogPath(), std::ios::app);
    if (!out.is_open()) {
        return;
    }
    out << line.dump() << '\n';
}

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

json gap(const std::string& generation_id, const std::string& reason) {
    return {
        {"event", "GENERATION_PROGRESS_GAP"},
        {"generation_id", generation_id},
        {"sample_time_ms", nowMs()},
        {"reason", reason},
    };
}

void recordFetch(const std::string& generation_id, const InferenceHttpResponse& http) {
    if (!http.ok) {
        appendProgress(gap(generation_id, "transport"));
        return;
    }
    json body;
    try {
        body = json::parse(http.body);
    } catch (...) {
        appendProgress(gap(generation_id, "parse_error"));
        return;
    }
    if (containsTextKey(body)) {
        appendProgress(gap(generation_id, "discarded_text"));
        return;
    }
    json slots = json::array();
    if (body.is_array()) {
        for (const auto& slot : body) {
            if (slot.is_object()) {
                slots.push_back(slimSlot(slot));
            }
        }
    }
    appendProgress({
        {"event", "GENERATION_PROGRESS"},
        {"generation_id", generation_id},
        {"sample_time_ms", nowMs()},
        {"slots", std::move(slots)},
    });
}

struct InflightGuard {
    bool held = false;
    InflightGuard() {
        std::lock_guard<std::mutex> lock(inflight_mutex);
        if (!fetch_inflight) {
            fetch_inflight = true;
            held = true;
        }
    }
    ~InflightGuard() {
        if (held) {
            std::lock_guard<std::mutex> lock(inflight_mutex);
            fetch_inflight = false;
        }
    }
};

} // namespace

void GenerationProgressIds::set(const std::string& generation_id) {
    std::lock_guard<std::mutex> lock(id_mutex);
    current_generation_id = generation_id;
}

std::string GenerationProgressIds::current() {
    std::lock_guard<std::mutex> lock(id_mutex);
    return current_generation_id;
}

void GenerationProgressIds::clear() {
    std::lock_guard<std::mutex> lock(id_mutex);
    current_generation_id.clear();
}

void setProgressObserverTestHooks(ProgressObserverTestHooks hooks) {
    std::lock_guard<std::mutex> lock(hooks_mutex);
    test_hooks = std::move(hooks);
}

void clearProgressObserverTestHooks() {
    std::lock_guard<std::mutex> lock(hooks_mutex);
    test_hooks = {};
}

struct GenerationProgressSession::State {
    std::atomic<bool> stop{false};
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    std::thread worker;
};

GenerationProgressSession::GenerationProgressSession(std::string generation_id, std::string slots_url) {
    if (generation_id.empty()) {
        return;
    }
    state_ = new State();
    state_->worker = std::thread([this, generation_id = std::move(generation_id), slots_url = std::move(slots_url)]() {
        constexpr auto kInterval = std::chrono::seconds(60);
        while (!state_->stop.load()) {
            {
                std::unique_lock<std::mutex> lock(state_->wait_mutex);
                ProgressObserverTestHooks hooks;
                {
                    std::lock_guard<std::mutex> hook_lock(hooks_mutex);
                    hooks = test_hooks;
                }
                if (hooks.wait_for) {
                    lock.unlock();
                    hooks.wait_for(kInterval);
                } else if (state_->wait_cv.wait_for(lock, kInterval, [this] { return state_->stop.load(); })) {
                    break;
                }
            }
            if (state_->stop.load()) {
                break;
            }
            InflightGuard guard;
            if (!guard.held) {
                appendProgress(gap(generation_id, "in_flight"));
                continue;
            }
            InferenceHttpResponse http;
            ProgressObserverTestHooks hooks;
            {
                std::lock_guard<std::mutex> hook_lock(hooks_mutex);
                hooks = test_hooks;
            }
            if (hooks.fetch) {
                http = hooks.fetch();
            } else if (slots_url.find("fail_on_no_slot") != std::string::npos) {
                appendProgress(gap(generation_id, "transport"));
                continue;
            } else {
                http = inferenceHttpGet(slots_url, 10);
            }
            recordFetch(generation_id, http);
        }
    });
}

GenerationProgressSession::~GenerationProgressSession() {
    if (state_ == nullptr) {
        return;
    }
    state_->stop.store(true);
    state_->wait_cv.notify_all();
    if (state_->worker.joinable()) {
        state_->worker.join();
    }
    delete state_;
    state_ = nullptr;
}

} // namespace Thoth
