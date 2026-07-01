/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — E2 STRICT enforcement (evaluation kernel boundary)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/e2_strict_enforcement.h"
#include "../include/benchmark_environment.h"

#include <sstream>

namespace Thoth {

namespace {

void requireNonEmpty(const std::string& field, const char* name) {
    if (field.empty()) {
        throw E2StrictValidationError(std::string("STRICT missing required field: ") + name);
    }
}

} // namespace

nlohmann::json E2EvaluationFingerprint::toJson() const {
    return {{"fingerprint_hash", fingerprint_hash}, {"canonical_json", canonical_json}};
}

E2EvaluationFingerprint computeEvaluationFingerprint(const E2EvalConfig& config) {
    nlohmann::json canonical = {
        {"protocol_version", kE2ProtocolVersion},
        {"scoring_function", kEpisodicLearningScoringFunction},
        {"strict_retrieval_engine", kE2StrictRetrievalEngineVersion},
        {"tier", e2EvalTierToString(config.tier)},
        {"versions", config.versions.toJson()},
    };

    E2EvaluationFingerprint fp;
    fp.canonical_json = canonical.dump();
    fp.fingerprint_hash = sha256Hex(fp.canonical_json);
    return fp;
}

void validateStrictConfigForOfficialRun(const E2EvalConfig& config, bool uses_embeddings) {
    if (config.tier != E2EvalTier::STRICT) {
        throw E2StrictValidationError("validateStrictConfigForOfficialRun requires STRICT tier");
    }

    requireNonEmpty(config.versions.corpus_snapshot_id, "corpus_snapshot_id");
    requireNonEmpty(config.versions.model_version_or_weights_hash,
                    "model_version_or_weights_hash");
    requireNonEmpty(config.versions.retrieval_engine_version, "retrieval_engine_version");

    if (uses_embeddings) {
        requireNonEmpty(config.versions.embedding_model_version, "embedding_model_version");
    }

    if (config.versions.retrieval_engine_version != kE2StrictRetrievalEngineVersion) {
        throw E2StrictValidationError(
            "STRICT retrieval_engine_version must be " +
            std::string(kE2StrictRetrievalEngineVersion));
    }
}

void assertOfficialHarnessBuild() {
    // Link anchor for run_episodic_learning_benchmark (THOTH_E2_OFFICIAL_HARNESS).
}

} // namespace Thoth
