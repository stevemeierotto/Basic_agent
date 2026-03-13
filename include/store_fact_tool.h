/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — store_fact tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_STORE_FACT_TOOL_H
#define THOTH_STORE_FACT_TOOL_H

#include "itool.h"
#include "fact_store.h"

/**
 * @class StoreFactTool
 * @brief Enables the agent to persist structured knowledge.
 *
 * Implemented according to TOOLS.md v1.0.
 */
class StoreFactTool : public ITool {
public:
    explicit StoreFactTool(Thoth::FactStore& fact_store);

    std::string name() const override { return "store_fact"; }
    
    std::string description() const override {
        return "Stores a structured fact into the agent's long-term knowledge base.";
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;


private:
    Thoth::FactStore& fact_store_;
};

#endif // THOTH_STORE_FACT_TOOL_H
