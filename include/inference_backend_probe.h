/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — G1d-CO A0 inference backend probe (client-identity provenance)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_INFERENCE_BACKEND_PROBE_H
#define THOTH_INFERENCE_BACKEND_PROBE_H

#include "inference_client.h"
#include "inference_endpoint.h"
#include "json.hpp"

#include <optional>
#include <string>

class Config;

namespace Thoth {

/** Provenance snapshot for the inference backend actually exercised. */
struct InferenceBackendSnapshot {
    /** From InferenceClient::backendName() after instantiation — source of truth. */
    std::string backend_name;
    std::string base_url;
    std::string embed_base_url;
    bool reachable = false;
    std::string error;
    /** Backend-specific diagnostic fields when available (not required for all backends). */
    nlohmann::json diagnostics = nlohmann::json::object();
};

/**
 * Build a snapshot from an already-instantiated client and a health result.
 * Backend identity comes from client.backendName(), not from config.
 */
InferenceBackendSnapshot snapshotFromInferenceClient(
    const InferenceClient& client,
    const InferenceEndpointConfig& endpoints,
    const InferenceHealthResult& health);

/**
 * Resolve endpoints, createInferenceClient, health-check, and snapshot.
 * On construction failure, returns unreachable snapshot with error set.
 */
InferenceBackendSnapshot probeInferenceBackend(const Config* config = nullptr);

/** True when probeInferenceBackend reports reachable. */
bool isInferenceBackendReachable(const Config* config = nullptr);

} // namespace Thoth

#endif // THOTH_INFERENCE_BACKEND_PROBE_H
