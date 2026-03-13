/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 8.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/strategy_engine.h"
#include "../include/logger.h"
#include <unordered_map>
#include <algorithm>
#include <chrono>

namespace Thoth {

StrategyEngine::StrategyEngine(std::shared_ptr<Memory> memory) : memory_(memory) {}

void StrategyEngine::processTrajectories() {
    if (!memory_) return;

    auto all_trajs = memory_->getAllTrajectories();
    if (all_trajs.size() < 3) return;

    std::unordered_map<std::string, PatternCandidate> candidates;

    for (const auto& rec : all_trajs) {
        try {
            nlohmann::json tj = nlohmann::json::parse(rec.trajectory_json);
            if (!tj.contains("steps") || !tj["steps"].is_array()) continue;

            std::vector<std::string> step_sequence;
            for (const auto& s : tj["steps"]) {
                // Use StepType as the pattern key
                int type = s.value("type", 0);
                step_sequence.push_back(std::to_string(type));
            }

            if (step_sequence.size() < 2) continue;

            std::string key = generate_pattern_key(step_sequence);
            auto& cand = candidates[key];
            if (cand.steps.empty()) cand.steps = step_sequence;
            cand.count++;
            cand.total_success += rec.success_score;

        } catch (...) {
            continue;
        }
    }

    // Phase 8.2: Strategy Extraction
    for (const auto& [key, cand] : candidates) {
        float avg_success = cand.total_success / static_cast<float>(cand.count);
        
        // Strategy Selection Criteria: min 3 occurrences, >= 0.8 success rate
        if (cand.count >= 3 && avg_success >= 0.8f) {
            Memory::CognateStrategyRecord strategy;
            strategy.strategy_id = "strat-" + key.substr(0, 8);
            strategy.description = "Autonomous strategy for sequence: " + key;
            
            nlohmann::json pattern_j = cand.steps;
            strategy.step_pattern_json = pattern_j.dump();
            strategy.success_rate = avg_success;
            strategy.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();

            memory_->saveStrategy(strategy);
            
            StructuredLogger::instance().log(LogLevel::Info, "strategy_engine", "STRATEGY_EMERGED", 
                "New strategy extracted from trajectories", 
                {{"strategy_id", strategy.strategy_id}, {"occurrences", cand.count}, {"success_rate", avg_success}});
        }
    }
}

std::string StrategyEngine::generate_pattern_key(const std::vector<std::string>& steps) {
    std::string key;
    for (size_t i = 0; i < steps.size(); ++i) {
        key += steps[i];
        if (i < steps.size() - 1) key += "->";
    }
    return key;
}

} // namespace Thoth
