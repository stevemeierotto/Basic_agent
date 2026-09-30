/*
 * Copyright (c) 2026 Steve Meierotto
 * 
 * Thoth — StrategyEngine 2.0 (Cognate V2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/strategy_engine.h"
#include "../include/logger.h"
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <set>
#include <iostream>

namespace Thoth {

StrategyEngine::StrategyEngine(std::shared_ptr<Memory> memory) : memory_(memory) {}

void StrategyEngine::processTrajectories() {
    if (!memory_) return;

    auto all_trajs = memory_->getAllTrajectories();
    std::cout << "  [StrategyEngine] Processing " << all_trajs.size() << " trajectories...\n";
    // Thesis Rule: Minimum of 3 trajectories required for extraction
    if (all_trajs.size() < 3) {
        StructuredLogger::instance().log(LogLevel::Info, "strategy_engine", "INSUFFICIENT_DATA", 
            "Not enough trajectories for strategy extraction", {{"count", all_trajs.size()}});
        return;
    }

    std::unordered_map<std::string, PatternCandidate> candidates;

    for (const auto& rec : all_trajs) {
        try {
            nlohmann::json tj = nlohmann::json::parse(rec.trajectory_json);
            if (!tj.contains("steps") || !tj["steps"].is_array()) continue;

            std::vector<std::string> step_sequence;
            for (const auto& s : tj["steps"]) {
                if (!s.is_object()) {
                    continue;
                }
                std::string label;
                if (s.contains("tool") && s["tool"].is_string()) {
                    const std::string tool_name = s["tool"].get<std::string>();
                    if (!tool_name.empty() && tool_name != "none") {
                        label = "TOOL:" + tool_name;
                    }
                }
                if (label.empty()) {
                    if (!s.contains("type") || !s["type"].is_number_integer()) {
                        continue;
                    }
                    const int type = s["type"].get<int>();
                    switch (type) {
                        case 1: label = "RETRIEVAL"; break;
                        case 2: label = "LLM"; break;
                        default: label = "STEP_" + std::to_string(type); break;
                    }
                }
                step_sequence.push_back(std::move(label));
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

    // Phase 3.1: Strategy Promotion & Library
    for (const auto& [key, cand] : candidates) {
        float avg_success = cand.total_success / static_cast<float>(cand.count);
        
        // Thesis Differentiator: 80% success / 3-run threshold
        if (cand.count >= 3 && avg_success >= 0.8f) {
            
            // Check if strategy already exists to avoid duplication
            // (In a real implementation we'd use a database lookup by pattern key)
            
            Memory::CognateStrategyRecord strategy;
            // Generate a deterministic ID based on the pattern key for stability
            std::hash<std::string> hasher;
            size_t hash_val = hasher(key);
            std::stringstream ss;
            ss << "strat-" << std::hex << (hash_val & 0xFFFFFFFF);
            strategy.strategy_id = ss.str();
            
            strategy.description = "Successful pattern detected: " + key;
            
            nlohmann::json pattern_j = cand.steps;
            strategy.step_pattern_json = pattern_j.dump();
            strategy.success_rate = avg_success;
            strategy.created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();

            memory_->saveStrategy(strategy);
            
            StructuredLogger::instance().log(LogLevel::Info, "strategy_engine", "STRATEGY_PROMOTED", 
                "Pattern promoted to Strategy (Threshold Met)", 
                {
                    {"strategy_id", strategy.strategy_id}, 
                    {"occurrences", cand.count}, 
                    {"success_rate", avg_success},
                    {"pattern", key},
                    {"thesis_threshold_met", true}
                });
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
