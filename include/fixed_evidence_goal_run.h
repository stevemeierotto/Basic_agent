/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — one-shot EGAR-LX fixed-evidence goal file
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_FIXED_EVIDENCE_GOAL_RUN_H
#define THOTH_FIXED_EVIDENCE_GOAL_RUN_H

#include <string>

namespace Thoth {

class ExecutiveController;

/**
 * Read one local goal file. The returned text is the file contents.
 * A missing path, a non-file, an unreadable file, a NUL byte, or a
 * whitespace-only file sets error and returns false. This does not
 * execute a goal and does not consume the one-goal slot.
 */
bool readFixedEvidenceGoalFile(const std::string& path,
                               std::string& goal,
                               std::string& error);

/**
 * Run one goal through ExecutiveController::execute_fixed_evidence_goal
 * and wait until that goal reaches a terminal state.
 *
 * The first call in the process consumes the slot. A later call returns
 * false and does not start another goal. Empty text does not consume the
 * slot. An absent or invalid evidence directory does not consume the slot
 * and does not enable containment. Production Run, /goal, chat, --serve,
 * and --execute do not call this.
 */
bool runFixedEvidenceGoal(ExecutiveController& controller,
                          const std::string& goal,
                          const std::string& evidence_directory,
                          std::string& error,
                          std::string& terminal_state);

/** Read path, then runFixedEvidenceGoal. */
bool runFixedEvidenceGoalFile(ExecutiveController& controller,
                              const std::string& path,
                              const std::string& evidence_directory,
                              std::string& error,
                              std::string& terminal_state);

} // namespace Thoth

#endif
