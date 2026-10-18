/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — EGAR-LX fixed-evidence containment policy
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/fixed_evidence_containment.h"

#include <atomic>

namespace Thoth {

namespace {

std::atomic<bool> g_fixed_evidence_containment{false};

} // namespace

void FixedEvidenceContainment::enable() {
    g_fixed_evidence_containment.store(true, std::memory_order_release);
}

bool FixedEvidenceContainment::enabled() {
    return g_fixed_evidence_containment.load(std::memory_order_acquire);
}

ContainmentConfirmation FixedEvidenceContainment::confirmContainmentEnabled() {
    ContainmentConfirmation confirmation;
    confirmation.experimental_environment_ready = false;
    if (enabled()) {
        confirmation.containment_enabled = true;
        confirmation.detail = "containment_only_environment_not_checked";
    } else {
        confirmation.containment_enabled = false;
        confirmation.detail = "containment_disabled";
    }
    return confirmation;
}

} // namespace Thoth
