/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — append-only non-chat generation telemetry
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include "generation_call.h"

namespace Thoth {

class GenerationCallLog {
public:
    /** Writes one metadata line. Returns false for chat or any record that would store text. */
    static bool append(const GenerationOutcome& outcome, const GenerationRecordFields& fields);

    static std::string logFilePath();
};

} // namespace Thoth
