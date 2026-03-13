/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 8.1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include "strategy.h"
#include "trajectory.h"
#include "memory.h"

namespace Thoth {

/**
 * @brief Analyzes trajectories to extract successful patterns.
 */
class StrategyEngine {
public:
    explicit StrategyEngine(std::shared_ptr<Memory> memory);
    
    /**
     * @brief Scans memory for repeating successful patterns.
     */
    void processTrajectories();

private:
    std::shared_ptr<Memory> memory_;
    
    struct PatternCandidate {
        std::vector<std::string> steps;
        int count = 0;
        float total_success = 0.0f;
    };

    std::string generate_pattern_key(const std::vector<std::string>& steps);
};

} // namespace Thoth
