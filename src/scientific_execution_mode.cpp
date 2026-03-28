/*
 * Copyright (c) 2026 Steve Meierotto
 * 
 * Thoth — ScientificExecutionMode (Cognate V2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/scientific_execution_mode.h"
#include "../include/executive_controller.h"
#include "../include/logger.h"
#include "../include/llm_interface.h"
#include "../include/prompt_factory.h"
#include "../include/config.h"
#include <set>
#include <algorithm>
#include <cmath>

namespace Thoth {

void ScientificExecutionMode::execute_step(ExecutiveController& controller) {
    // Terminal state check
    {
        std::lock_guard<std::mutex> lock(controller.mutex_);
        if (controller.state_ == ControllerState::COMPLETED || 
            controller.state_ == ControllerState::ABORTED || 
            controller.state_ == ControllerState::FAILED) {
            return;
        }
    }

    switch (reasoning_stage_) {
        case 0: generate_hypotheses(controller); break;
        case 1: extract_constraints(controller); break;
        case 2: evaluate_feasibility(controller); break;
        case 3: finalize_selection(controller); break;
        default:
            reasoning_stage_ = 0;
            break;
    }
}

void ScientificExecutionMode::generate_hypotheses(ExecutiveController& controller) {
    StructuredLogger::instance().log(LogLevel::Info, "scientific_mode", "HYPOTHESIS_STAGE", "Generating hypotheses", {});
    
    auto state = controller.get_problem_state();
    last_hypothesis_set_ = state.hypotheses;
    
    // Simulate TOOL call for trajectory recording
    PlanStep mock_step;
    mock_step.step_id = "sci-hyp-" + std::to_string(state.iteration_count);
    mock_step.description = "Scientific Hypothesis Generation";
    mock_step.type = StepType::TOOL;
    mock_step.tool = {{"tool", "llm_reasoning"}};
    
    Thoth::StepResult mock_res;
    mock_res.success = true;
    mock_res.data = {{"hypothesis", "generated"}};
    
    controller.record_trajectory_step(mock_step, mock_res);

    state.hypotheses.push_back("Hypothesis " + std::to_string(state.iteration_count + 1) + ".A");
    state.hypotheses.push_back("Hypothesis " + std::to_string(state.iteration_count + 1) + ".B");
    
    controller.update_problem_state(state);
    controller.emit_event(EventType::STATE_CHANGED, "", {
        {"reasoning_stage", "hypothesis_generation"},
        {"hypotheses_count", state.hypotheses.size()}
    });
    
    reasoning_stage_ = 1;
}

void ScientificExecutionMode::extract_constraints(ExecutiveController& controller) {
    StructuredLogger::instance().log(LogLevel::Info, "scientific_mode", "CONSTRAINT_STAGE", "Extracting constraints", {});
    
    auto state = controller.get_problem_state();
    
    // Simulate TOOL call
    PlanStep mock_step;
    mock_step.step_id = "sci-const-" + std::to_string(state.iteration_count);
    mock_step.description = "Constraint Extraction";
    mock_step.type = StepType::TOOL;
    mock_step.tool = {{"tool", "project_analyze"}};
    
    Thoth::StepResult mock_res;
    mock_res.success = true;
    
    controller.record_trajectory_step(mock_step, mock_res);

    state.constraints.push_back("Constraint " + std::to_string(state.iteration_count + 1));
    
    controller.update_problem_state(state);
    controller.emit_event(EventType::STATE_CHANGED, "", {
        {"reasoning_stage", "constraint_extraction"},
        {"constraints_count", state.constraints.size()}
    });
    
    reasoning_stage_ = 2;
}

void ScientificExecutionMode::evaluate_feasibility(ExecutiveController& controller) {
    StructuredLogger::instance().log(LogLevel::Info, "scientific_mode", "EVALUATION_STAGE", "Evaluating feasibility", {});
    
    auto state = controller.get_problem_state();
    
    // Rigorous simulation of confidence gain
    current_confidence_ = 0.6f + (state.iteration_count * 0.15f);
    if (current_confidence_ > 0.95f) current_confidence_ = 0.95f; 
    
    state.confidence_score = current_confidence_;
    state.confidence_history.push_back(current_confidence_);
    
    controller.update_problem_state(state);
    controller.emit_event(EventType::STATE_CHANGED, "", {
        {"reasoning_stage", "feasibility_evaluation"},
        {"confidence_score", current_confidence_}
    });
    
    reasoning_stage_ = 3;
}

void ScientificExecutionMode::finalize_selection(ExecutiveController& controller) {
    StructuredLogger::instance().log(LogLevel::Info, "scientific_mode", "SELECTION_STAGE", "Finalizing selection", {});
    
    auto state = controller.get_problem_state();
    state.iteration_count++;
    
    bool converged = is_converged(controller, state);
    
    int max_iters = 5;
    if (controller.rag_ && controller.rag_->config) {
        max_iters = controller.rag_->config->max_scientific_iterations;
    }
    bool max_hit = (state.iteration_count >= max_iters);

    controller.update_problem_state(state);
    controller.emit_event(EventType::STATE_CHANGED, "", {
        {"reasoning_stage", "final_selection"},
        {"iteration_count", state.iteration_count},
        {"converged", converged},
        {"max_iterations_hit", max_hit}
    });

    if (converged || max_hit) {
        std::string reason = converged ? "convergence" : "max_iterations";
        StructuredLogger::instance().log(LogLevel::Info, "scientific_mode", "LOOP_EXIT", 
            "Exiting scientific reasoning loop due to " + reason, {{"reason", reason}});
            
        controller.transition_to(ControllerState::IDLE);
        reasoning_stage_ = 0; 
    } else {
        reasoning_stage_ = 0; 
    }
}

bool ScientificExecutionMode::is_converged(ExecutiveController& controller, const ProblemState& state) const {
    float epsilon = 0.05f;
    int window = 2;
    
    if (controller.rag_ && controller.rag_->config) {
        epsilon = controller.rag_->config->convergence_epsilon;
        window = controller.rag_->config->stability_window;
    }

    // 1. Numerical Stability Check
    if (state.confidence_history.size() < static_cast<size_t>(window)) {
        return false;
    }

    size_t n = state.confidence_history.size();
    bool numerically_stable = true;
    for (int i = 1; i < window; ++i) {
        float delta = std::abs(state.confidence_history[n - i] - state.confidence_history[n - i - 1]);
        if (delta > epsilon) {
            numerically_stable = false;
            break;
        }
    }

    // 2. Semantic Stability Check (Jaccard Similarity)
    float jaccard = calculate_jaccard(last_hypothesis_set_, state.hypotheses);
    bool semantically_stable = (jaccard > 0.9f); 

    // 3. High Confidence Gate
    bool high_confidence = (state.confidence_score > 0.8f);

    return numerically_stable && semantically_stable && high_confidence;
}

float ScientificExecutionMode::calculate_jaccard(const std::vector<std::string>& a, const std::vector<std::string>& b) const {
    if (a.empty() && b.empty()) return 1.0f;
    if (a.empty() || b.empty()) return 0.0f;

    std::set<std::string> set_a(a.begin(), a.end());
    std::set<std::string> set_b(b.begin(), b.end());

    std::vector<std::string> intersection;
    std::set_intersection(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
                          std::back_inserter(intersection));

    std::vector<std::string> union_set;
    std::set_union(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
                   std::back_inserter(union_set));

    if (union_set.empty()) return 0.0f;
    return static_cast<float>(intersection.size()) / static_cast<float>(union_set.size());
}

} // namespace Thoth
