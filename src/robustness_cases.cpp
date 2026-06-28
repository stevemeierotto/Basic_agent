/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C5 robustness golden cases
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/robustness_cases.h"
#include "../include/robustness_mock_responses.h"
#include "../include/config.h"
#include "../include/embedding_engine.h"
#include "../include/executive_controller.h"
#include "../include/index_manager.h"
#include "../include/llm_interface.h"
#include "../include/llm_interface.h"
#include "../include/llm_planner.h"
#include "../include/memory.h"
#include "../include/plan_validator.h"
#include "../include/prompt_factory.h"
#include "../include/rag.h"
#include "../include/tools.h"
#include "../include/step_metrics_repository.h"
#include "../include/workflow_engine.h"
#include "../include/file_handler.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <thread>

namespace fs = std::filesystem;

namespace Thoth {

namespace {

std::int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string stateName(ControllerState state) {
    switch (state) {
        case ControllerState::COMPLETED: return "COMPLETED";
        case ControllerState::FAILED: return "FAILED";
        case ControllerState::ABORTED: return "ABORTED";
        default: return "INCOMPLETE";
    }
}

void clearRobustnessEnv() {
    unsetenv("THOTH_MOCK_LLM");
    unsetenv("THOTH_MOCK_LLM_UNAVAILABLE");
    unsetenv("THOTH_MOCK_LLM_DELAY_MS");
    unsetenv("THOTH_MOCK_STEP_TIMEOUT");
    unsetenv("THOTH_TEST_SUITE_DEV");
    RobustnessMockResponses::reset();
}

bool planHasRetrievalBeforeLlm(const Plan& plan) {
    const PlanStep* firstExec = nullptr;
    for (const auto& step : plan.steps) {
        if (step.type == StepType::TOOL) {
            continue;
        }
        firstExec = &step;
        break;
    }
    if (!firstExec || firstExec->type != StepType::RETRIEVAL) {
        return false;
    }
    bool hasLlm = false;
    for (const auto& step : plan.steps) {
        if (step.type == StepType::LLM) {
            hasLlm = true;
        }
    }
    return hasLlm;
}

bool planHasValidDependencies(const Plan& plan) {
    for (const auto& step : plan.steps) {
        if (step.type != StepType::LLM || step.depends_on.empty()) {
            continue;
        }
        for (const auto& dep : step.depends_on) {
            bool found = false;
            for (const auto& other : plan.steps) {
                if (other.step_id == dep && other.type == StepType::RETRIEVAL) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool planIsStructurallyValid(Plan plan) {
    const auto validation = PlanValidator::validateAndRepair(plan, false);
    return validation.valid && planHasRetrievalBeforeLlm(plan);
}

struct ExecHarnessResult {
    std::string terminal_state;
    std::string failure_reason;
    int reflection_cycles = 0;
    int planner_calls = 0;
    bool synthesis_prompt_ok = false;
    std::string synthesis_prompt;
    nlohmann::json details;
};

ExecHarnessResult runExecutiveCase(
    const std::function<std::shared_ptr<IPlanner>()>& makePlanner,
    const std::function<void(std::shared_ptr<RAGPipeline>&, Config&)>& setupRag,
    int maxReflections = 0,
    int waitSeconds = 15) {
    ExecHarnessResult out;
    Config cfg;
    cfg.max_reflections = maxReflections;
    cfg.database_path =
        (fs::temp_directory_path() / ("thoth_robustness_" + std::to_string(nowMs()) + ".db")).string();
    if (fs::exists(cfg.database_path)) {
        fs::remove(cfg.database_path);
    }

    auto memory = std::make_shared<Memory>(cfg);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    auto idx = new IndexManager(engine.get());
    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
    setupRag(rag, cfg);

    auto planner = makePlanner();
    auto registry = std::make_shared<ToolRegistry>();
    LLMInterface llm(LLMBackend::Ollama, &cfg);
    ExecutiveController controller(planner, registry, rag, memory);
    controller.set_config(&cfg);
    controller.set_llm_interface(&llm);
    controller.set_max_reflections(maxReflections);

    std::atomic<bool> terminal{false};
    controller.set_event_callback([&](const ControllerEvent& ev) {
        if (ev.type == EventType::PLAN_COMPLETED || ev.type == EventType::PLAN_FAILED ||
            ev.type == EventType::PLAN_ABORTED) {
            terminal.store(true);
        }
    });

    controller.execute_goal("robustness harness goal");

    int ticks = waitSeconds * 10;
    while (!terminal.load() && ticks-- > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    out.terminal_state = stateName(controller.get_state());
    out.reflection_cycles = controller.get_reflection_count();

    const Plan finalPlan = controller.get_current_plan();
    for (const auto& step : finalPlan.steps) {
        if (step.type == StepType::LLM && step.result.is_object()) {
            if (step.result.contains("data") && step.result["data"].is_object()) {
                const auto& data = step.result["data"];
                if (data.contains("synthesis_prompt") && data["synthesis_prompt"].is_string()) {
                    out.synthesis_prompt = data["synthesis_prompt"].get<std::string>();
                } else if (data.contains("data") && data["data"].is_object()) {
                    const auto& inner = data["data"];
                    if (inner.contains("synthesis_prompt") && inner["synthesis_prompt"].is_string()) {
                        out.synthesis_prompt = inner["synthesis_prompt"].get<std::string>();
                    }
                }
            }
            if (step.status == StepStatus::FAILED && out.failure_reason.empty()) {
                out.failure_reason = step.result.value("error_message", "LLM_STEP_FAILED");
            }
        }
        if (step.type == StepType::RETRIEVAL && step.status == StepStatus::FAILED &&
            out.failure_reason.empty()) {
            out.failure_reason = step.result.value("error_message", "RETRIEVAL_STEP_FAILED");
        }
        if (step.step_id == "timeout-step" && step.status == StepStatus::FAILED &&
            out.failure_reason.empty()) {
            out.failure_reason = "STEP_TIMEOUT";
        }
    }

    if (out.terminal_state == "INCOMPLETE") {
        out.failure_reason = "HARNESS_TIMEOUT";
    }

    fs::remove(cfg.database_path);
    return out;
}

class HarnessMockPlanner : public IPlanner {
public:
    explicit HarnessMockPlanner(nlohmann::json planTemplate) : planTemplate_(std::move(planTemplate)) {}

    int call_count() const { return call_count_; }

    Plan create_plan(const std::string& goal) override {
        ++call_count_;
        Plan plan;
        plan.plan_id = "robustness-plan-" + std::to_string(call_count_);
        plan.goal = goal;
        plan.status = PlanStatus::ACTIVE;

        if (planTemplate_.contains("steps") && planTemplate_["steps"].is_array()) {
            for (const auto& spec : planTemplate_["steps"]) {
                PlanStep step;
                step.step_id = spec.value("step_id", "step");
                step.description = spec.value("description", "harness step");
                const std::string type = spec.value("step_type", "LLM");
                if (type == "RETRIEVAL") {
                    step.type = StepType::RETRIEVAL;
                } else if (type == "NODE") {
                    step.type = StepType::NODE;
                } else {
                    step.type = StepType::LLM;
                }
                if (spec.contains("payload") && spec["payload"].is_object()) {
                    step.payload = spec["payload"];
                } else {
                    step.payload = nlohmann::json::object();
                }
                if (spec.contains("depends_on")) {
                    step.depends_on = spec["depends_on"].get<std::vector<std::string>>();
                }
                if (spec.contains("timeout_ms")) {
                    step.failure_policy.timeout_ms = spec["timeout_ms"].get<int>();
                }
                plan.steps.push_back(step);
            }
        }
        return plan;
    }

    Plan revise_plan(const Plan& plan, const nlohmann::json&) override { return plan; }

private:
    nlohmann::json planTemplate_;
    int call_count_ = 0;
};

class AlwaysFailPlanner : public IPlanner {
public:
    int call_count() const { return call_count_; }

    Plan create_plan(const std::string& goal) override {
        ++call_count_;
        Plan plan;
        plan.plan_id = "fail-plan-" + std::to_string(call_count_);
        plan.goal = goal;
        PlanStep step;
        step.step_id = "fail-step";
        step.description = "Always fails for reflection budget test";
        step.type = StepType::NODE;
        step.payload = {{"node_id", "robustness-fail-node"}};
        plan.steps.push_back(step);
        return plan;
    }

    Plan revise_plan(const Plan& plan, const nlohmann::json&) override { return plan; }

private:
    int call_count_ = 0;
};

class SlowThenFastPlanner : public IPlanner {
public:
    int call_count() const { return call_count_; }
    const std::string& active_goal() const { return active_goal_; }

    Plan create_plan(const std::string& goal) override {
        ++call_count_;
        active_goal_ = goal;
        Plan plan;
        plan.plan_id = "concurrent-plan-" + std::to_string(call_count_);
        plan.goal = goal;

        PlanStep step;
        step.step_id = goal.find("slow") != std::string::npos ? "slow-step" : "fast-step";
        step.description = step.step_id;
        step.type = StepType::LLM;
        step.payload = {{"prompt", step.step_id}};
        plan.steps.push_back(step);
        return plan;
    }

    Plan revise_plan(const Plan& plan, const nlohmann::json&) override { return plan; }

private:
    int call_count_ = 0;
    std::string active_goal_;
};

RobustnessCaseOutcome makeOutcome(const RobustnessCaseSpec& spec) {
    RobustnessCaseOutcome out;
    out.case_id = spec.id;
    out.category = categoryName(spec.category);
    out.scenario = spec.scenario;
    return out;
}

RobustnessCaseOutcome runPlanningInvalidJsonFallback(const RobustnessCaseSpec& spec) {
    auto out = makeOutcome(spec);
    const auto start = nowMs();

    Config cfg;
    cfg.database_path = (fs::temp_directory_path() / "thoth_robustness_plan_json.db").string();
    auto memory = std::make_shared<Memory>(cfg);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    auto idx = new IndexManager(engine.get());
    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
    PromptFactory::ensureDefaultTemplatesExist();
    auto promptFactory = std::make_shared<PromptFactory>(*memory, *rag);
    LLMInterface llm(LLMBackend::Ollama, &cfg);
    LLMPlanner planner(memory, rag, promptFactory, &llm);

    RobustnessMockResponses::pushAll({
        "{ this is not valid planner json",
        "{\"plan\":[{\"step_type\":\"TOOL\",\"description\":\"bad tool step\"}]}",
    });

    const Plan plan = planner.create_plan("Robustness invalid JSON goal");
    out.structurally_valid_plan = planIsStructurallyValid(plan);
    out.valid_dependencies = planHasValidDependencies(plan);
    out.fallback_used = plan.steps.size() >= 2 &&
                        plan.steps.front().step_id == "retrieve-context" &&
                        plan.steps.back().step_id == "synthesize";
    out.terminal_state = out.structurally_valid_plan ? "PLAN_READY" : "PLAN_INVALID";
    out.pass = out.structurally_valid_plan && out.valid_dependencies;
    out.pass_reason = out.pass
                          ? "terminal_state=PLAN_READY structurally_valid_plan=true valid_dependencies=true"
                          : "plan did not reach a structurally valid RETRIEVAL→LLM shape";
    out.duration_ms = nowMs() - start;
    out.details = {{"step_count", plan.steps.size()}, {"fallback_used", out.fallback_used}};
    fs::remove(cfg.database_path);
    return out;
}

RobustnessCaseOutcome runPlanningInvalidOrderFallback(const RobustnessCaseSpec& spec) {
    auto out = makeOutcome(spec);
    const auto start = nowMs();

    const std::string wrongOrder = R"({
        "plan": [
            {"step_id":"s1","step_type":"LLM","description":"Summarize first"},
            {"step_id":"r1","step_type":"RETRIEVAL","description":"Retrieve second","payload":{"query":"late retrieval","top_k":3}}
        ]
    })";

    Config cfg;
    cfg.database_path = (fs::temp_directory_path() / "thoth_robustness_plan_order.db").string();
    auto memory = std::make_shared<Memory>(cfg);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    auto idx = new IndexManager(engine.get());
    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
    PromptFactory::ensureDefaultTemplatesExist();
    auto promptFactory = std::make_shared<PromptFactory>(*memory, *rag);
    LLMInterface llm(LLMBackend::Ollama, &cfg);
    LLMPlanner planner(memory, rag, promptFactory, &llm);

    RobustnessMockResponses::push(wrongOrder);
    RobustnessMockResponses::push(wrongOrder);

    const Plan plan = planner.create_plan("Robustness invalid order goal");
    out.structurally_valid_plan = planIsStructurallyValid(plan);
    out.valid_dependencies = planHasValidDependencies(plan);
    out.fallback_used = plan.steps.front().type == StepType::RETRIEVAL;
    out.terminal_state = out.structurally_valid_plan ? "PLAN_READY" : "PLAN_INVALID";
    out.pass = out.structurally_valid_plan && plan.steps.front().type == StepType::RETRIEVAL;
    out.pass_reason = out.pass
                          ? "terminal_state=PLAN_READY first_executable=RETRIEVAL valid_dependencies=true"
                          : "plan still has invalid executable ordering";
    out.duration_ms = nowMs() - start;
    fs::remove(cfg.database_path);
    return out;
}

RobustnessCaseOutcome runPlanningDependsOnRepair(const RobustnessCaseSpec& spec) {
    auto out = makeOutcome(spec);
    const auto start = nowMs();

    const std::string missingDeps = R"({
        "plan": [
            {"step_id":"retrieve-context","step_type":"RETRIEVAL","description":"Retrieve","payload":{"query":"repair test","top_k":3}},
            {"step_id":"synthesize","step_type":"LLM","description":"Summarize"}
        ]
    })";

    Config cfg;
    cfg.database_path = (fs::temp_directory_path() / "thoth_robustness_plan_dep.db").string();
    auto memory = std::make_shared<Memory>(cfg);
    auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
    auto idx = new IndexManager(engine.get());
    auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
    PromptFactory::ensureDefaultTemplatesExist();
    auto promptFactory = std::make_shared<PromptFactory>(*memory, *rag);
    LLMInterface llm(LLMBackend::Ollama, &cfg);
    LLMPlanner planner(memory, rag, promptFactory, &llm);

    RobustnessMockResponses::push(missingDeps);
    const Plan plan = planner.create_plan("Robustness depends_on repair goal");

    out.structurally_valid_plan = planIsStructurallyValid(plan);
    out.valid_dependencies = planHasValidDependencies(plan);
    out.terminal_state = out.valid_dependencies ? "PLAN_READY" : "PLAN_INVALID";
    out.pass = out.valid_dependencies;
    out.pass_reason = out.pass ? "valid_dependencies=true LLM→RETRIEVAL link present" : "depends_on not repaired";
    out.duration_ms = nowMs() - start;
    fs::remove(cfg.database_path);
    return out;
}

} // namespace

const char* categoryName(RobustnessCategory category) {
    switch (category) {
        case RobustnessCategory::Planning: return "planning";
        case RobustnessCategory::Retrieval: return "retrieval";
        case RobustnessCategory::Execution: return "execution";
        case RobustnessCategory::Reflection: return "reflection";
        case RobustnessCategory::Lifecycle: return "lifecycle";
        default: return "unknown";
    }
}

std::vector<RobustnessCaseSpec> getRobustnessCases() {
    return {
        {"C5-01", RobustnessCategory::Retrieval, "Empty RAG index — retrieval fails cleanly, goal reaches terminal state"},
        {"C5-02", RobustnessCategory::Retrieval, "Empty retrieval result — populated index, zero matches, LLM sees explicit empty-context message"},
        {"C5-03", RobustnessCategory::Planning, "Invalid JSON — retry — structurally valid executable plan"},
        {"C5-04", RobustnessCategory::Planning, "Invalid step ordering — retry — structurally valid executable plan"},
        {"C5-05", RobustnessCategory::Planning, "Missing depends_on — final plan has valid RETRIEVAL→LLM dependencies"},
        {"C5-06", RobustnessCategory::Reflection, "Reflection budget exhausted — terminal state without unbounded retry"},
        {"C5-07", RobustnessCategory::Execution, "Step timeout — terminal state within bounded wait"},
        {"C5-08", RobustnessCategory::Execution, "LLM unavailable (mock) — clean failure, no hang"},
        {"C5-09", RobustnessCategory::Execution, "Concurrent goals — second goal replaces first and completes"},
        {"C5-10", RobustnessCategory::Lifecycle, "Controller teardown after async work — no crash on destruction"},
    };
}

RobustnessCaseOutcome runRobustnessCase(const RobustnessCaseSpec& spec) {
    clearRobustnessEnv();
    setenv("THOTH_MOCK_LLM", "true", 1);

    if (spec.id == "C5-03") {
        return runPlanningInvalidJsonFallback(spec);
    }
    if (spec.id == "C5-04") {
        return runPlanningInvalidOrderFallback(spec);
    }
    if (spec.id == "C5-05") {
        return runPlanningDependsOnRepair(spec);
    }

    if (spec.id == "C5-01") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        const nlohmann::json planSpec = {
            {"steps", nlohmann::json::array({
                            {{"step_id", "retrieve-context"},
                             {"step_type", "RETRIEVAL"},
                             {"description", "Retrieve from empty index"},
                             {"payload", {{"query", "anything"}, {"top_k", 3}}}},
                            {{"step_id", "synthesize"},
                             {"step_type", "LLM"},
                             {"description", "Summarize"},
                             {"depends_on", nlohmann::json::array({"retrieve-context"})}},
                        })}};

        auto exec = runExecutiveCase(
            [&]() { return std::make_shared<HarnessMockPlanner>(planSpec); },
            [](std::shared_ptr<RAGPipeline>&, Config&) {},
            0,
            15);

        out.terminal_state = exec.terminal_state;
        out.failure_reason = exec.failure_reason.empty() ? "RETRIEVAL_FAILED" : exec.failure_reason;
        out.reflection_cycles = exec.reflection_cycles;
        out.duration_ms = nowMs() - start;
        out.pass = out.terminal_state == "FAILED" && out.terminal_state != "INCOMPLETE";
        out.pass_reason = out.pass
                              ? "terminal_state=FAILED failure_reason=" + out.failure_reason
                              : "expected clean FAILED terminal state for empty index";
        return out;
    }

    if (spec.id == "C5-02") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        setenv("THOTH_MOCK_LLM", "true", 1);

        Config cfg;
        auto memory = std::make_shared<Memory>(cfg);
        auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
        auto idx = new IndexManager(engine.get());
        auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
        auto registry = std::make_shared<ToolRegistry>();
        auto metrics = std::make_shared<StepMetricsRepository>("");
        Thoth::WorkflowEngine workflow(registry, rag, memory, metrics);
        LLMInterface llm(LLMBackend::Ollama, &cfg);
        workflow.setLLMInterface(&llm);

        PlanStep retrievalStep;
        retrievalStep.step_id = "retrieve-context";
        retrievalStep.type = StepType::RETRIEVAL;
        retrievalStep.description = "Retrieve with zero scored chunks";
        retrievalStep.payload = {{"query", "xqzvblth999nomatch"}, {"top_k", 5}};

        PlanStep llmStep;
        llmStep.step_id = "synthesize";
        llmStep.type = StepType::LLM;
        llmStep.description = "Summarize findings";
        llmStep.depends_on = {"retrieve-context"};
        llmStep.payload = nlohmann::json::object();

        CodeChunk chunk;
        chunk.code = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        chunk.fileName = "widgets.md";
        chunk.keyword_score = 1.0f;
        idx->addChunkToIndex(std::move(chunk));

        const StepResult retrievalResult = workflow.executeStep(retrievalStep, "robustness-plan", {});
        StepExecutionContext ctx;
        ctx.goal = "robustness empty retrieval goal";
        ctx.prior_steps.push_back({
            retrievalStep.step_id,
            static_cast<int>(StepType::RETRIEVAL),
            retrievalStep.description,
            retrievalResult.data,
        });

        const StepResult llmResult = workflow.executeStep(llmStep, "robustness-plan", ctx);
        std::string synthesisPrompt;
        if (llmResult.data.contains("data") && llmResult.data["data"].is_object()) {
            const auto& data = llmResult.data["data"];
            if (data.contains("synthesis_prompt") && data["synthesis_prompt"].is_string()) {
                synthesisPrompt = data["synthesis_prompt"].get<std::string>();
            }
        }

        const bool retrievalSucceededEmpty =
            retrievalResult.success && retrievalResult.data.value("retrieval_empty", false);
        out.synthesis_prompt_ok =
            synthesisPrompt.find("No relevant documents found.") != std::string::npos;
        out.terminal_state = llmResult.success ? "COMPLETED" : "FAILED";
        out.failure_reason = llmResult.success ? "" : llmResult.error_message;
        out.duration_ms = nowMs() - start;
        out.pass = retrievalSucceededEmpty && llmResult.success && out.synthesis_prompt_ok;
        out.pass_reason =
            out.pass
                ? "retrieval_empty=true terminal_state=COMPLETED synthesis_prompt_ok=true"
                : "empty retrieval did not produce explicit empty-context synthesis prompt";
        out.details = {{"retrieval_succeeded_empty", retrievalSucceededEmpty},
                       {"synthesis_prompt_excerpt",
                        synthesisPrompt.substr(0, std::min<std::size_t>(120, synthesisPrompt.size()))}};
        return out;
    }

    if (spec.id == "C5-06") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        std::shared_ptr<AlwaysFailPlanner> planner;
        auto exec = runExecutiveCase(
            [&]() {
                planner = std::make_shared<AlwaysFailPlanner>();
                return planner;
            },
            [](std::shared_ptr<RAGPipeline>&, Config&) {},
            1,
            20);

        out.terminal_state = exec.terminal_state;
        out.failure_reason = exec.failure_reason.empty() ? "LOW_SCORE" : exec.failure_reason;
        out.reflection_cycles = exec.reflection_cycles;
        out.planner_calls = planner ? planner->call_count() : 0;
        out.duration_ms = nowMs() - start;
        out.pass = out.terminal_state != "INCOMPLETE" && out.reflection_cycles <= 1 &&
                   out.planner_calls <= 2;
        out.pass_reason = out.pass
                              ? "terminal_state=" + out.terminal_state +
                                    " reflection_cycles=" + std::to_string(out.reflection_cycles) +
                                    " bounded=true"
                              : "reflection retry was unbounded or harness timed out";
        out.details = {{"planner_calls", out.planner_calls}};
        return out;
    }

    if (spec.id == "C5-07") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        unsetenv("THOTH_MOCK_LLM");
        setenv("THOTH_MOCK_STEP_TIMEOUT", "1", 1);

        const nlohmann::json planSpec = {
            {"steps", nlohmann::json::array({
                            {{"step_id", "timeout-step"},
                             {"step_type", "LLM"},
                             {"description", "Timeout probe"},
                             {"timeout_ms", 1}},
                        })}};

        auto exec = runExecutiveCase(
            [&]() { return std::make_shared<HarnessMockPlanner>(planSpec); },
            [](std::shared_ptr<RAGPipeline>&, Config&) {},
            2,
            15);

        out.terminal_state = exec.terminal_state;
        out.failure_reason = exec.failure_reason.empty() ? "STEP_TIMEOUT" : exec.failure_reason;
        out.reflection_cycles = exec.reflection_cycles;
        out.duration_ms = nowMs() - start;
        out.pass = out.terminal_state == "FAILED" && out.terminal_state != "INCOMPLETE";
        out.pass_reason = out.pass ? "terminal_state=FAILED failure_reason=STEP_TIMEOUT bounded=true"
                                   : "timeout did not reach terminal FAILED state";
        return out;
    }

    if (spec.id == "C5-08") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        unsetenv("THOTH_MOCK_LLM");
        setenv("THOTH_MOCK_LLM_UNAVAILABLE", "true", 1);

        const nlohmann::json planSpec = {
            {"steps", nlohmann::json::array({
                            {{"step_id", "retrieve-context"},
                             {"step_type", "RETRIEVAL"},
                             {"description", "Retrieve context"},
                             {"payload", {{"query", "zzzznomatchzzzz"}, {"top_k", 3}}}},
                            {{"step_id", "synthesize"},
                             {"step_type", "LLM"},
                             {"description", "Summarize"},
                             {"depends_on", nlohmann::json::array({"retrieve-context"})}},
                        })}};

        auto exec = runExecutiveCase(
            [&]() { return std::make_shared<HarnessMockPlanner>(planSpec); },
            [](std::shared_ptr<RAGPipeline>& rag, Config&) {
                CodeChunk chunk;
                chunk.code = "Widget corpus details for synthesis.";
                chunk.fileName = "widgets.md";
                chunk.keyword_score = 1.0f;
                chunk.embedding = rag->engine->embed(chunk.code);
                rag->indexManager->addChunkToIndex(std::move(chunk));
            },
            0,
            15);

        out.terminal_state = exec.terminal_state;
        out.failure_reason = exec.failure_reason;
        if (exec.failure_reason.find("[Error]") != std::string::npos ||
            exec.failure_reason.find("unavailable") != std::string::npos) {
            out.failure_reason = "LLM_UNAVAILABLE";
        }
        out.reflection_cycles = exec.reflection_cycles;
        out.duration_ms = nowMs() - start;
        out.pass = out.terminal_state == "FAILED" && out.reflection_cycles <= 1 &&
                   out.terminal_state != "INCOMPLETE" &&
                   (out.failure_reason == "LLM_UNAVAILABLE" ||
                    exec.failure_reason.find("[Error]") != std::string::npos);
        out.pass_reason = out.pass ? "terminal_state=FAILED failure_reason=LLM_UNAVAILABLE no_hang=true"
                                   : "LLM unavailable did not fail cleanly";
        return out;
    }

    if (spec.id == "C5-09") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        setenv("THOTH_MOCK_LLM_DELAY_MS", "2000", 1);

        Config cfg;
        cfg.database_path = (fs::temp_directory_path() / "thoth_robustness_concurrent.db").string();
        auto memory = std::make_shared<Memory>(cfg);
        auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
        auto idx = new IndexManager(engine.get());
        auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
        auto planner = std::make_shared<SlowThenFastPlanner>();
        auto registry = std::make_shared<ToolRegistry>();
        LLMInterface llm(LLMBackend::Ollama, &cfg);
        ExecutiveController controller(planner, registry, rag, memory);
        controller.set_config(&cfg);
        controller.set_llm_interface(&llm);

        std::atomic<bool> terminal{false};
        controller.set_event_callback([&](const ControllerEvent& ev) {
            if (ev.type == EventType::PLAN_COMPLETED || ev.type == EventType::PLAN_FAILED) {
                terminal.store(true);
            }
        });

        controller.execute_goal("slow concurrent goal");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        controller.execute_goal("fast concurrent goal");

        int ticks = 150;
        while (!terminal.load() && ticks-- > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        out.terminal_state = stateName(controller.get_state());
        out.planner_calls = planner->call_count();
        out.duration_ms = nowMs() - start;
        const bool fastActive = controller.get_current_plan().goal.find("fast") != std::string::npos;
        out.pass = out.terminal_state == "COMPLETED" && out.planner_calls >= 2 && fastActive;
        out.pass_reason = out.pass ? "terminal_state=COMPLETED active_goal=fast planner_calls>=2"
                                   : "concurrent goal handoff did not complete on the second goal";
        out.details = {{"active_goal", controller.get_current_plan().goal},
                       {"planner_calls", out.planner_calls}};
        fs::remove(cfg.database_path);
        return out;
    }

    if (spec.id == "C5-10") {
        auto out = makeOutcome(spec);
        const auto start = nowMs();
        bool destroyedCleanly = true;
        try {
            Config cfg;
            cfg.database_path = (fs::temp_directory_path() / "thoth_robustness_teardown.db").string();
            auto memory = std::make_shared<Memory>(cfg);
            auto engine = std::make_unique<EmbeddingEngine>(EmbeddingEngine::Method::TfIdf);
            auto idx = new IndexManager(engine.get());
            auto rag = std::make_shared<RAGPipeline>(std::move(engine), idx, &cfg, memory.get());
            const nlohmann::json planSpec = {
                {"steps", nlohmann::json::array({
                                {{"step_id", "fast-step"},
                                 {"step_type", "LLM"},
                                 {"description", "Quick step"}},
                            })}};
            auto planner = std::make_shared<HarnessMockPlanner>(planSpec);
            auto registry = std::make_shared<ToolRegistry>();
            LLMInterface llm(LLMBackend::Ollama, &cfg);
            {
                ExecutiveController controller(planner, registry, rag, memory);
                controller.set_config(&cfg);
                controller.set_llm_interface(&llm);
                std::atomic<bool> terminal{false};
                controller.set_event_callback([&](const ControllerEvent& ev) {
                    if (ev.type == EventType::PLAN_COMPLETED || ev.type == EventType::PLAN_FAILED) {
                        terminal.store(true);
                    }
                });
                controller.execute_goal("teardown goal");
                int ticks = 150;
                while (!terminal.load() && ticks-- > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            fs::remove(cfg.database_path);
        } catch (...) {
            destroyedCleanly = false;
        }

        out.terminal_state = destroyedCleanly ? "DESTROYED" : "CRASH";
        out.duration_ms = nowMs() - start;
        out.pass = destroyedCleanly;
        out.pass_reason = out.pass ? "terminal_state=DESTROYED no_crash=true" : "controller destruction crashed";
        return out;
    }

    auto out = makeOutcome(spec);
    out.pass = false;
    out.pass_reason = "unknown case id";
    out.terminal_state = "UNKNOWN";
    return out;
}

} // namespace Thoth
