/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — EGAR-LX fixed-evidence containment policy
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_FIXED_EVIDENCE_CONTAINMENT_H
#define THOTH_FIXED_EVIDENCE_CONTAINMENT_H

namespace Thoth {

/**
 * Read-only admission result.
 *
 * experimental_environment_ready stays false. This check does not inspect a
 * workspace, database, snapshot, manifest, or execution authorization.
 *
 * Future experimental entry must call enable() before goal execution begins,
 * while prefetch state is empty. A prefetched step cached before enable() can
 * be replayed without another registry call. Package 1 does not switch
 * containment mid-run and does not establish full experimental isolation.
 */
struct ContainmentConfirmation {
    bool containment_enabled = false;
    bool experimental_environment_ready = false;
    const char* detail = "containment_disabled";
};

/**
 * Process-wide registry-tool gate. Starts disabled. enable() is idempotent
 * and there is no production disable. The flag is not read from configuration
 * or the environment.
 */
class FixedEvidenceContainment {
public:
    static void enable();
    static bool enabled();
    static ContainmentConfirmation confirmContainmentEnabled();
};

} // namespace Thoth

#endif // THOTH_FIXED_EVIDENCE_CONTAINMENT_H
