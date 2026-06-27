/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C3 reflection A/B measurement harness
 *
 * Compares goal outcomes with max_reflections=0 vs 2 on deterministic mock cases.
 * No Ollama required (THOTH_MOCK_LLM).
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/config.h"
#include "../include/embedding_engine.h"
#include "../include/executive_controller.h"
#include "../include/index_manager.h"
#include "../include/memory.h"
#include "../include/rag.h"
#include "../include/reflection_ab_cases.h"
#include "../include/tools.h"
#include "file_handler.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "../include/json.hpp"

namespace fs = std::filesystem;

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string benchmarkLogPath() {
    FileHandler fh;
    fs::path logsDir = fs::path(fh.getProjectRoot()) / "logs";
    fs::create_directories(logsDir);
    return (logsDir / "reflection_ab_benchmark.jsonl").string();
}

void appendJsonLine(const std::string& path, const nlohmann::json& event) {
    std::ofstream out(path, std::ios::app);
    if (out.is_open()) {
        out << event.dump() << '\n';
    }
}

std::string stateName(Thoth::ControllerState state) {
    switch (state) {
        case Thoth::ControllerState::COMPLETED: return "COMPLETED";
        case Thoth::ControllerState::FAILED: return "FAILED";
        case Thoth::ControllerState::ABORTED: return "ABORTED";
        default: return "INCOMPLETE";
    }
}

struct ArmResult {
    int max_reflections = 0;
    std::string terminal_state;
    int planner_calls = 0;
    int reflection_count = 0;
    std::string reflection_skip_reason;
    bool reflection_replan_seen = false;
    std::int64_t wall_clock_ms = 0;
};

ArmResult runCaseArm(const Thoth::ReflectionAbCase& spec, int maxReflections) {
    unsetenv("THOTH_MOCK_STEP_TIMEOUT");
    if (spec.fixture == Thoth::ReflectionAbFixture::TimeoutStepFailure) {
        setenv("THOTH_MOCK_STEP_TIMEOUT", "1", 1);
    }
    setenv("THOTH_MOCK_LLM", "true", 1);

    Config cfg;
    cfg.max_reflections = maxReflections;
    cfg.database_path = (fs::temp_directory_path() / ("thoth_reflection_ab_" + spec.id + "_" +
                                                      std::to_string(maxReflections) + ".db"))
                            .string();

    if (fs::exists(cfg.database_path)) {
        fs::remove(cfg.database_path);
    }

    auto memory = std::make_shared<Memory>(cfg);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    auto idx = new IndexManager(engine.get());
    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx);
    auto planner = std::make_shared<Thoth::ReflectionAbMockPlanner>(spec.fixture);
    auto registry = std::make_shared<ToolRegistry>();

    Thoth::ExecutiveController controller(planner, registry, rag, memory);
    controller.set_max_reflections(maxReflections);

    std::atomic<bool> terminal{false};
    std::atomic<bool> reflectionReplan{false};
    controller.set_event_callback([&](const ControllerEvent& ev) {
        if (ev.type == EventType::PLAN_COMPLETED || ev.type == EventType::PLAN_FAILED ||
            ev.type == EventType::PLAN_ABORTED) {
            terminal.store(true);
        }
        if (ev.type == EventType::REFLECTION_REPLAN) {
            reflectionReplan.store(true);
        }
    });

    const auto start = nowMs();
    controller.execute_goal(spec.goal);

    int timeout = 150;
    while (!terminal.load() && timeout > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        --timeout;
    }

    ArmResult result;
    result.max_reflections = maxReflections;
    result.terminal_state = stateName(controller.get_state());
    result.planner_calls = planner->call_count();
    result.reflection_count = controller.get_reflection_count();
    result.reflection_replan_seen = reflectionReplan.load();
    result.wall_clock_ms = nowMs() - start;

    if (result.terminal_state == "INCOMPLETE") {
        result.reflection_skip_reason = "harness_timeout";
    }

    fs::remove(cfg.database_path);
    return result;
}

bool armMatchesExpectation(const ArmResult& result, const std::string& expectedState, int expectedPlannerCalls) {
    return result.terminal_state == expectedState && result.planner_calls == expectedPlannerCalls;
}

} // namespace

int main() {
    std::cout << "C3 — Reflection A/B Benchmark (mock, no Ollama)\n";

    const auto cases = Thoth::getReflectionAbCases();
    const std::string logPath = benchmarkLogPath();
    const std::int64_t ts = nowMs();

    int passedCases = 0;
    float reflectionLift = 0.0f;

    for (const auto& spec : cases) {
        std::cout << "\n" << spec.id << " — " << spec.description << '\n';

        const ArmResult offArm = runCaseArm(spec, 0);
        const ArmResult onArm = runCaseArm(spec, 2);

        const bool offOk = armMatchesExpectation(offArm, spec.expected_outcome_off, spec.expected_planner_calls_off);
        const bool onOk = armMatchesExpectation(onArm, spec.expected_outcome_on, spec.expected_planner_calls_on);
        const bool casePass = offOk && onOk;

        if (casePass) {
            ++passedCases;
        }

        const bool reflectionHelped =
            offArm.terminal_state != "COMPLETED" && onArm.terminal_state == "COMPLETED";
        if (reflectionHelped) {
            reflectionLift += 1.0f;
        }

        std::cout << "  arm off (max=0): state=" << offArm.terminal_state
                  << " planner_calls=" << offArm.planner_calls
                  << " reflection=" << offArm.reflection_count << '\n';
        std::cout << "  arm on  (max=2): state=" << onArm.terminal_state
                  << " planner_calls=" << onArm.planner_calls
                  << " reflection=" << onArm.reflection_count
                  << " replan_event=" << (onArm.reflection_replan_seen ? "yes" : "no") << '\n';
        std::cout << "  case pass: " << (casePass ? "YES" : "NO") << '\n';

        appendJsonLine(logPath, {
            {"event", "REFLECTION_AB_CASE"},
            {"timestamp_ms", ts},
            {"case_id", spec.id},
            {"description", spec.description},
            {"arm_off", {
                 {"max_reflections", offArm.max_reflections},
                 {"terminal_state", offArm.terminal_state},
                 {"planner_calls", offArm.planner_calls},
                 {"reflection_count", offArm.reflection_count},
                 {"reflection_replan_seen", offArm.reflection_replan_seen},
                 {"wall_clock_ms", offArm.wall_clock_ms},
                 {"expected_state", spec.expected_outcome_off},
                 {"pass", offOk},
             }},
            {"arm_on", {
                 {"max_reflections", onArm.max_reflections},
                 {"terminal_state", onArm.terminal_state},
                 {"planner_calls", onArm.planner_calls},
                 {"reflection_count", onArm.reflection_count},
                 {"reflection_replan_seen", onArm.reflection_replan_seen},
                 {"wall_clock_ms", onArm.wall_clock_ms},
                 {"expected_state", spec.expected_outcome_on},
                 {"pass", onOk},
             }},
            {"reflection_helped", reflectionHelped},
            {"case_pass", casePass},
        });
    }

    reflectionLift /= static_cast<float>(cases.empty() ? 1 : cases.size());

    appendJsonLine(logPath, {
        {"event", "REFLECTION_AB_SUMMARY"},
        {"timestamp_ms", ts},
        {"case_count", cases.size()},
        {"cases_passed", passedCases},
        {"mean_reflection_lift", reflectionLift},
    });

    std::cout << "\nSummary\n";
    std::cout << "  cases passed: " << passedCases << '/' << cases.size() << '\n';
    std::cout << "  mean reflection lift: " << reflectionLift << '\n';
    std::cout << "  log: " << logPath << '\n';

    return passedCases == static_cast<int>(cases.size()) ? 0 : 2;
}
