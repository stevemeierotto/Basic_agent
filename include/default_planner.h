/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — ExecutiveController Phase 1
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "iplanner.h"

class DefaultPlanner : public IPlanner {
public:
    Plan create_plan(const std::string& goal) override;
    Plan revise_plan(const Plan& existing_plan,
                     const nlohmann::json& step_result) override;
};
