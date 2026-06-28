/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C5 robustness & failure scenario harness
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/robustness_cases.h"
#include "../include/robustness_mock_responses.h"
#include "file_handler.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string logPath() {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "robustness_suite.jsonl").string();
}

void appendJsonLine(const std::string& path, const nlohmann::json& event) {
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << event.dump() << '\n';
    }
}

nlohmann::json outcomeToJson(const Thoth::RobustnessCaseOutcome& out, std::int64_t runTs) {
    return {
        {"event", "ROBUSTNESS_CASE"},
        {"timestamp_ms", runTs},
        {"case_id", out.case_id},
        {"category", out.category},
        {"scenario", out.scenario},
        {"terminal_state", out.terminal_state},
        {"failure_reason", out.failure_reason},
        {"reflection_cycles", out.reflection_cycles},
        {"fallback_used", out.fallback_used},
        {"planner_calls", out.planner_calls},
        {"structurally_valid_plan", out.structurally_valid_plan},
        {"valid_dependencies", out.valid_dependencies},
        {"synthesis_prompt_ok", out.synthesis_prompt_ok},
        {"duration_ms", out.duration_ms},
        {"pass", out.pass},
        {"pass_reason", out.pass_reason},
        {"details", out.details},
    };
}

} // namespace

int main() {
    std::cout << "C5 — Robustness Suite (mock-only, no Ollama)\n";

    const auto cases = Thoth::getRobustnessCases();
    const std::string logFile = logPath();
    const std::int64_t runTs = nowMs();

    {
        std::ofstream truncateLog(logFile, std::ios::trunc);
    }

    int passed = 0;
    for (const auto& spec : cases) {
        Thoth::RobustnessMockResponses::reset();
        const Thoth::RobustnessCaseOutcome out = Thoth::runRobustnessCase(spec);

        if (out.pass) {
            ++passed;
        }

        std::cout << "\n" << out.case_id << " [" << out.category << "] — " << out.scenario << '\n';
        std::cout << "  terminal_state=" << out.terminal_state;
        if (!out.failure_reason.empty()) {
            std::cout << " failure_reason=" << out.failure_reason;
        }
        if (out.reflection_cycles > 0) {
            std::cout << " reflection_cycles=" << out.reflection_cycles;
        }
        if (out.planner_calls > 0) {
            std::cout << " planner_calls=" << out.planner_calls;
        }
        if (out.fallback_used) {
            std::cout << " fallback_used=true";
        }
        if (out.valid_dependencies) {
            std::cout << " valid_dependencies=true";
        }
        if (out.synthesis_prompt_ok) {
            std::cout << " synthesis_prompt_ok=true";
        }
        std::cout << " duration_ms=" << out.duration_ms << '\n';
        std::cout << "  pass: " << (out.pass ? "YES" : "NO") << " — " << out.pass_reason << '\n';

        appendJsonLine(logFile, outcomeToJson(out, runTs));
    }

    appendJsonLine(logFile, {
        {"event", "ROBUSTNESS_SUMMARY"},
        {"timestamp_ms", runTs},
        {"case_count", cases.size()},
        {"cases_passed", passed},
    });

    std::cout << "\nSummary\n";
    std::cout << "  cases passed: " << passed << '/' << cases.size() << '\n';
    std::cout << "  log: " << logFile << '\n';

    return passed == static_cast<int>(cases.size()) ? 0 : 2;
}
