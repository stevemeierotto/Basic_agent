/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 2 chat retrieval tuning constants
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_CHAT_RETRIEVAL_CONFIG_H
#define THOTH_CHAT_RETRIEVAL_CONFIG_H

#include <cstddef>

namespace Thoth {

namespace ChatRetrieval {

/** Chunks shorter than this are deprioritized and skipped for injection when possible. */
inline constexpr std::size_t kMinChunkChars = 80;

/** Added to final_score when chunk filename matches a salient query token (e.g. GRAG → GRAG.md). */
inline constexpr float kFilenameMatchBoost = 0.55f;

/** Multiplier applied to scores of chunks below kMinChunkChars before ranking. */
inline constexpr float kTinyChunkScoreFactor = 0.25f;

/** Extra boost for substantive definition paragraphs on definitional queries. */
inline constexpr float kSubstantiveChunkBoost = 0.08f;
inline constexpr std::size_t kSubstantiveChunkChars = 200;

/** Boost for HOWTO-style docs on "how do I use …" queries. */
inline constexpr float kUsageDocBoost = 0.12f;

/**
 * Plan M G1 (R1) — fail-closed grounding floor.
 * A chat chunk may set grounded=true only if its post-boost final_score is finite
 * and >= this floor. This applies to the boosted retrieval-pipeline score, NOT raw
 * embedding cosine; it is a floor to reject zero / near-zero / broken scores, not a
 * calibrated relevance threshold. Any stronger "meaningful retrieval" bar is deferred
 * until telemetry exists (see docs/plan_m_grounded_retrieval_gate.md).
 */
inline constexpr float kMinGroundingFinalScore = 0.01f;

} // namespace ChatRetrieval

} // namespace Thoth

#endif // THOTH_CHAT_RETRIEVAL_CONFIG_H
