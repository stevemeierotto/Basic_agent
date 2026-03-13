/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include <string>
#include "plan.h"
#include "json.hpp"

class IPlanner {
public:
    virtual Plan create_plan(const std::string& goal) = 0;
    virtual Plan revise_plan(const Plan& existing_plan,
                             const nlohmann::json& step_result) = 0;
    virtual ~IPlanner() = default;
};
