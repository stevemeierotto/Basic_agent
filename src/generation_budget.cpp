/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — MTCP shared generation ceiling and optional inference seed
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "../include/generation_budget.h"

#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace Thoth {

namespace {

struct BudgetState {
    bool loaded = false;
    bool ceiling_set = false;
    int ceiling = 0;
    bool seed_set = false;
    int seed = 0;
    std::string error;
};

BudgetState& state() {
    static BudgetState value;
    return value;
}

std::mutex& stateMutex() {
    static std::mutex value;
    return value;
}

bool isCanonical(const std::string& text) {
    if (text.empty() || text.size() > 10) {
        return false;
    }
    if (text[0] == '0') {
        return false;
    }
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
    }
    return true;
}

} // namespace

std::optional<std::string> GenerationBudget::parseCanonicalInteger(const std::string& text, int& out) {
    if (!isCanonical(text)) {
        return std::string("value must be a canonical base-10 integer from 1 to 2147483647");
    }
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(text, &consumed, 10);
        if (consumed != text.size() || parsed < kMinValue || parsed > kMaxValue) {
            return std::string("value must be a canonical base-10 integer from 1 to 2147483647");
        }
        out = static_cast<int>(parsed);
        return std::nullopt;
    } catch (...) {
        return std::string("value must be a canonical base-10 integer from 1 to 2147483647");
    }
}

void GenerationBudget::loadFromEnvironment() {
    std::lock_guard<std::mutex> lock(stateMutex());
    auto& current = state();
    current = BudgetState{};
    current.loaded = true;

    auto apply = [&](const char* name, bool& flag, int& slot) {
        const char* raw = std::getenv(name);
        if (raw == nullptr || std::string(raw) == "__UNSET__") {
            return;
        }
        int parsed = 0;
        if (const auto error = parseCanonicalInteger(raw, parsed)) {
            current.error += name;
            current.error += ": ";
            current.error += *error;
            current.error += "; ";
            return;
        }
        flag = true;
        slot = parsed;
    };

    apply("THOTH_GENERATION_MAX_TOKENS", current.ceiling_set, current.ceiling);
    apply("THOTH_INFERENCE_SEED", current.seed_set, current.seed);
}

void GenerationBudget::enforceOrThrow() {
    loadFromEnvironment();
    std::lock_guard<std::mutex> lock(stateMutex());
    if (!state().error.empty()) {
        throw std::invalid_argument(state().error);
    }
}

bool GenerationBudget::hasCeiling() {
    std::lock_guard<std::mutex> lock(stateMutex());
    return state().ceiling_set;
}

int GenerationBudget::ceiling() {
    std::lock_guard<std::mutex> lock(stateMutex());
    return state().ceiling;
}

int GenerationBudget::resolvedOr(int fallback) {
    std::lock_guard<std::mutex> lock(stateMutex());
    if (state().ceiling_set) {
        return state().ceiling;
    }
    return fallback;
}

bool GenerationBudget::hasSeed() {
    std::lock_guard<std::mutex> lock(stateMutex());
    return state().seed_set;
}

int GenerationBudget::seed() {
    std::lock_guard<std::mutex> lock(stateMutex());
    return state().seed;
}

void GenerationBudget::resetForTests() {
    std::lock_guard<std::mutex> lock(stateMutex());
    state() = BudgetState{};
    state().loaded = true;
}

void GenerationBudget::setForTests(std::optional<int> ceilingValue, std::optional<int> seedValue) {
    std::lock_guard<std::mutex> lock(stateMutex());
    state() = BudgetState{};
    state().loaded = true;
    if (ceilingValue) {
        state().ceiling_set = true;
        state().ceiling = *ceilingValue;
    }
    if (seedValue) {
        state().seed_set = true;
        state().seed = *seedValue;
    }
}

} // namespace Thoth
