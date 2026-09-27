/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C6.4 successor episodic evaluation (episodic_authoritative_v2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_EPISODIC_AUTHORITATIVE_V2_H
#define THOTH_EPISODIC_AUTHORITATIVE_V2_H

#include "inference_client.h"
#include "json.hpp"

#include <memory>
#include <string>
#include <vector>

class Config;

namespace Thoth {

/** Successor identity. Not the sealed E2 / Phase E evaluation. */
inline constexpr const char* kEpisodicAuthoritativeV2Id = "episodic_authoritative_v2";

/**
 * One arm of the existing cold/warm pairing.
 * baseline = cold (no declared episode). episodic = warm (episode available).
 * The harness does not treat a higher episodic score as required.
 */
struct EpisodicV2Arm {
    bool present = false;
    bool response_valid = false;
};

struct EpisodicV2Case {
    std::string case_id;
    EpisodicV2Arm baseline;
    EpisodicV2Arm episodic;
};

/**
 * Identity is supplied. c64_cohort_fingerprint must be produced by
 * scripts/c64_environment_identity.py. This type does not recompute it.
 */
struct EpisodicV2Request {
    std::string evaluation_tier;
    std::string inference_backend_name;
    std::string llm_model;
    std::string embedding_model;
    std::string embedding_method;
    int embedding_dimension = 0;
    std::string thoth_git_sha;
    std::string basic_agent_git_sha;
    std::string corpus_fingerprint;
    std::string environment_schema_version;
    std::string protocol_version;
    std::string metric_schema_version;
    std::string c64_cohort_fingerprint;
    std::vector<EpisodicV2Case> cases;
};

struct EpisodicV2Report {
    bool authoritative_valid = false;
    std::string status;
    std::string reason;
    nlohmann::json evidence;
};

/** Configured InferenceClient. Does not call generate or health. */
std::unique_ptr<InferenceClient> makeEpisodicV2InferenceClient(const Config* config);

/**
 * Validate a successor run against the client that would serve it.
 * Does not call the model. Does not open a C6.4 window.
 */
EpisodicV2Report buildEpisodicAuthoritativeV2Report(const EpisodicV2Request& request,
                                                    const InferenceClient* client);

} // namespace Thoth

#endif
