/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — call-scoped generation metadata for MTCP telemetry
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/generation_call.h"

#include <mutex>
#include <unordered_map>

namespace Thoth {

std::vector<GenerationCallContext>& GenerationCallScope::stack() {
    thread_local std::vector<GenerationCallContext> value;
    return value;
}

GenerationCallScope::GenerationCallScope(GenerationCallContext context) {
    stack().push_back(std::move(context));
}

GenerationCallScope::~GenerationCallScope() {
    if (!stack().empty()) {
        stack().pop_back();
    }
}

GenerationCallContext GenerationCallScope::current() {
    if (stack().empty()) {
        return {};
    }
    return stack().back();
}

namespace {

std::mutex& rawCaptureMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, std::string>& rawCaptureValues() {
    static std::unordered_map<std::string, std::string> values;
    return values;
}

} // namespace

void RawProviderCapture::store(const std::string& capture_id, const std::string& raw_text) {
    std::lock_guard<std::mutex> lock(rawCaptureMutex());
    rawCaptureValues()[capture_id] = raw_text;
}

std::optional<std::string> RawProviderCapture::take(const std::string& capture_id) {
    std::lock_guard<std::mutex> lock(rawCaptureMutex());
    const auto found = rawCaptureValues().find(capture_id);
    if (found == rawCaptureValues().end()) {
        return std::nullopt;
    }
    std::string text = std::move(found->second);
    rawCaptureValues().erase(found);
    return text;
}

} // namespace Thoth
