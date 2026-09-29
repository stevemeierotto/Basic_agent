/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — MTCP shared generation ceiling and optional inference seed
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace Thoth {

/**
 * Process startup reading of THOTH_GENERATION_MAX_TOKENS and THOTH_INFERENCE_SEED.
 * Unset leaves each path on its existing ceiling and omits seed from requests.
 * A set but invalid value is an error. It does not fall back to 512.
 */
class GenerationBudget {
public:
    static constexpr int kMinValue = 1;
    static constexpr int kMaxValue = 2147483647;

    /** Empty string means the variable is unset. */
    static std::optional<std::string> parseCanonicalInteger(const std::string& text, int& out);

    static void loadFromEnvironment();
    /** Throws std::invalid_argument when a set variable is invalid. */
    static void enforceOrThrow();

    static bool hasCeiling();
    static int ceiling();
    /** When a ceiling is set, that value; otherwise fallback. */
    static int resolvedOr(int fallback);

    static bool hasSeed();
    static int seed();

    /** Test isolation. Does not read the process environment. */
    static void resetForTests();
    static void setForTests(std::optional<int> ceiling, std::optional<int> seed);
};

} // namespace Thoth
