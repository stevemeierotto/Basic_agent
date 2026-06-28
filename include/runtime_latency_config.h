/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C7 runtime latency defaults
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_RUNTIME_LATENCY_CONFIG_H
#define THOTH_RUNTIME_LATENCY_CONFIG_H

#include <cstddef>

namespace Thoth {

namespace RuntimeLatency {

/** Max chars of retrieved context injected into LLM synthesis steps. */
inline constexpr std::size_t kDefaultSynthesisMaxContextChars = 8192;

/** Default Ollama num_predict for synthesis steps (shorter than planner JSON). */
inline constexpr int kDefaultSynthesisNumPredict = 512;

/** Max concurrent RETRIEVAL steps (ready + prefetch). */
inline constexpr int kDefaultMaxParallelRetrieval = 4;

/** Overlap RETRIEVAL with its last blocking dependency when safe. */
inline constexpr bool kDefaultEnableRetrievalPrefetch = true;

} // namespace RuntimeLatency

} // namespace Thoth

#endif // THOTH_RUNTIME_LATENCY_CONFIG_H
