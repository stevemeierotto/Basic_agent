/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — LLM-backed episodic extraction for memory consolidation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_SUMMARY_GENERATOR_H
#define THOTH_SUMMARY_GENERATOR_H

#include "episodic_memory.h"
#include "memory_repository.h"
#include <string>
#include <vector>

class LLMInterface;

namespace Thoth {

struct SummaryGenerationResult {
    EpisodicMemory memory;
    bool llm_success = false;
    bool parse_success = false;
    bool truncated = false;
    bool fields_inferred = false;
    std::string prompt_version = "episodic_v1";
    std::string llm_model;
};

class SummaryGenerator {
public:
    explicit SummaryGenerator(LLMInterface* llm = nullptr);

    SummaryGenerationResult extract(const std::vector<MessageRecord>& batch) const;

private:
    LLMInterface* llm_;
};

} // namespace Thoth

#endif
