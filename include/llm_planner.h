/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * Thoth — Cognate Phase 1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#pragma once

#include "iplanner.h"
#include "memory.h"
#include "rag.h"
#include "llm_interface.h"
#include "prompt_factory.h"

/**
 * @brief Concrete implementation of IPlanner that generates plans using an LLM.
 */
class LLMPlanner : public IPlanner {
public:
    explicit LLMPlanner(std::shared_ptr<Memory> memory, 
                        std::shared_ptr<RAGPipeline> rag, 
                        std::shared_ptr<PromptFactory> prompt_factory,
                        LLMInterface* llm);
    ~LLMPlanner() override = default;

    Plan create_plan(const std::string& goal) override;
    Plan revise_plan(const Plan& existing_plan,
                     const nlohmann::json& step_result) override;

private:
    std::shared_ptr<Memory> memory_;
    std::shared_ptr<RAGPipeline> rag_;
    std::shared_ptr<PromptFactory> prompt_factory_;
    LLMInterface* llm_;
    
    std::string generate_uuid();
    void save_plan(const Plan& plan);
};
