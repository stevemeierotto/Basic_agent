/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — Conservative synthesis repetition guard
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_SYNTHESIS_OUTPUT_GUARD_H
#define THOTH_SYNTHESIS_OUTPUT_GUARD_H

#include <string>

namespace Thoth {

/**
 * @brief Detects synthesis text dominated by mechanical repetition.
 *
 * Phrase- and domain-agnostic. Intended to reject instruction-loop /
 * copy-paste loops without failing ordinary long or truncated answers.
 * Empty input returns false (empty handling stays elsewhere).
 */
bool synthesisOutputLooksUnusableRepetition(const std::string& text);

} // namespace Thoth

#endif
