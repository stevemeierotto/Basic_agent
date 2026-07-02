/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 STRICT enforcement (evaluation kernel boundary)
 *
 * Protocol: docs/E2_PROTOCOL.md v1.2
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_E2_STRICT_ENFORCEMENT_H
#define THOTH_E2_STRICT_ENFORCEMENT_H

#include "episodic_learning_eval.h"

#include <stdexcept>
#include <string>

namespace Thoth {

/** Protocol revision baked into evaluation fingerprint. */
constexpr const char* kE2ProtocolVersion = "1.2";

/**
 * Canonical retrieval engine id for STRICT deterministic path.
 * Must match linked e2_eval_kernel / e2_strict_retrieval build.
 */
constexpr const char* kE2StrictRetrievalEngineVersion = "e2_strict_retrieval_v1";

/** Thrown when STRICT official run preconditions are not met — no silent defaults. */
class E2StrictValidationError : public std::runtime_error {
public:
    explicit E2StrictValidationError(const std::string& message)
        : std::runtime_error(message) {}
};

/** A5 runtime fuse — STRICT context reached heuristic retrieval (architectural violation). */
class E2RuntimeHeuristicGuardViolation : public std::runtime_error {
public:
    E2RuntimeHeuristicGuardViolation();
};

/**
 * A5 guard entry — inspects authoritative E2EvalConfig::tier only (not THOTH_* env).
 * Throws E2RuntimeHeuristicGuardViolation when tier == STRICT.
 */
void guardAgainstStrictHeuristicRetrieval(const E2EvalConfig* eval_config);

/**
 * Single reproducibility unit: hash of full evaluation configuration.
 * Logged on every STRICT arm and summary.
 */
struct E2EvaluationFingerprint {
    std::string fingerprint_hash;
    std::string canonical_json;

    nlohmann::json toJson() const;
};

/** Deterministic SHA-256 over canonical config JSON + protocol + scoring function ids. */
E2EvaluationFingerprint computeEvaluationFingerprint(const E2EvalConfig& config);

/**
 * Canonical embedding pin: "{method}:{internal_version}" (e.g. "TfIdf:2").
 * Use instead of assigning getInternalVersion() int to std::string (int→char coercion bug).
 */
std::string makeEmbeddingModelVersionPin(const std::string& embedding_method,
                                         int internal_version);

/** True when every byte is printable ASCII (0x20–0x7E). Empty → false. */
bool isPrintableVersionPin(const std::string& value);

/**
 * STRICT official run gate — throws E2StrictValidationError on any missing field.
 * No implicit defaults. Call before any scored arm executes.
 */
void validateStrictConfigForOfficialRun(const E2EvalConfig& config, bool uses_embeddings);

/** Build-time marker: official harness calls this at startup (link anchor). */
void assertOfficialHarnessBuild();

} // namespace Thoth

#endif // THOTH_E2_STRICT_ENFORCEMENT_H
