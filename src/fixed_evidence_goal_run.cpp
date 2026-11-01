/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — one-shot EGAR-LX fixed-evidence goal file
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/fixed_evidence_goal_run.h"

#include "../include/agent_context_retrieval.h"
#include "../include/executive_controller.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace Thoth {
namespace {

std::atomic<bool> g_fixed_evidence_goal_claimed{false};

bool goalTextUsable(const std::string& goal) {
    for (unsigned char c : goal) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            return true;
        }
    }
    return false;
}

const char* terminalStateName(ControllerState state) {
    switch (state) {
        case ControllerState::COMPLETED: return "COMPLETED";
        case ControllerState::FAILED: return "FAILED";
        case ControllerState::ABORTED: return "ABORTED";
        default: return nullptr;
    }
}

bool claimFixedEvidenceGoal(std::string& error) {
    bool expected = false;
    if (!g_fixed_evidence_goal_claimed.compare_exchange_strong(expected, true)) {
        error = "one fixed-evidence goal per process";
        return false;
    }
    return true;
}

} // namespace

bool readFixedEvidenceGoalFile(const std::string& path,
                               std::string& goal,
                               std::string& error) {
    goal.clear();
    if (path.empty()) {
        error = "goal file path is empty";
        return false;
    }

    const std::filesystem::path file(path);
    std::error_code ec;
    const bool exists = std::filesystem::exists(file, ec);
    if (ec || !exists) {
        error = "goal file not found: " + path;
        return false;
    }
    const bool regular = std::filesystem::is_regular_file(file, ec);
    if (ec || !regular) {
        error = "goal path is not a file: " + path;
        return false;
    }

    std::ifstream in(file, std::ios::binary);
    if (!in) {
        error = "goal file could not be read: " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (in.bad()) {
        error = "goal file could not be read: " + path;
        return false;
    }

    const std::string text = buffer.str();
    if (text.find('\0') != std::string::npos) {
        error = "goal file is not valid text: " + path;
        return false;
    }
    if (!goalTextUsable(text)) {
        error = "goal file is empty: " + path;
        return false;
    }

    goal = text;
    error.clear();
    return true;
}

bool runFixedEvidenceGoal(ExecutiveController& controller,
                          const std::string& goal,
                          const std::string& evidence_directory,
                          std::string& error,
                          std::string& terminal_state) {
    terminal_state.clear();
    if (!goalTextUsable(goal)) {
        error = "goal text is empty";
        return false;
    }
    std::string resolved_directory;
    if (!resolveAuthorizedEvidenceDirectory(evidence_directory, resolved_directory, error)) {
        return false;
    }
    if (!claimFixedEvidenceGoal(error)) {
        return false;
    }

    try {
        const std::string accepted =
            controller.execute_fixed_evidence_goal(goal, resolved_directory);
        if (accepted.rfind("GOAL ACCEPTED:", 0) != 0) {
            error = accepted.empty() ? "fixed-evidence goal failed to start" : accepted;
            return false;
        }
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    } catch (...) {
        error = "fixed-evidence goal failed to start";
        return false;
    }

    // Wait until the goal loop leaves is_running(). Destroying the controller
    // earlier sets stop_requested and cuts the goal off.
    while (controller.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    const char* name = terminalStateName(controller.get_state());
    if (name == nullptr) {
        error = "fixed-evidence goal did not reach a terminal state";
        return false;
    }
    terminal_state = name;
    error.clear();
    return true;
}

bool runFixedEvidenceGoalFile(ExecutiveController& controller,
                              const std::string& path,
                              const std::string& evidence_directory,
                              std::string& error,
                              std::string& terminal_state) {
    std::string goal;
    if (!readFixedEvidenceGoalFile(path, goal, error)) {
        return false;
    }
    return runFixedEvidenceGoal(controller, goal, evidence_directory, error, terminal_state);
}

} // namespace Thoth
