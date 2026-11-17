/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — synthetic EGAR-LX A/B orchestration
 *
 * Mechanical pilot only. It does not collect historical cases or score diagnoses.
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_SYNTHETIC_AB_PILOT_H
#define THOTH_SYNTHETIC_AB_PILOT_H

#include "memory_snapshot.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Thoth {

enum class SyntheticExecution {
    Mock,
    Live
};

/**
 * Pilot phase. Full preserves the historical acquire→freeze→evaluate pipeline.
 * AcquisitionOnly stops after a validated freeze and writes an acquisition
 * manifest. EvaluationFromFrozen resumes from that freeze without re-running
 * acquisition.
 */
enum class SyntheticAbPhase {
    Full,
    AcquisitionOnly,
    EvaluationFromFrozen
};

/** Evaluation-arm goal order for one case. Independent of llama.cpp seeds. */
enum class ArmRunOrder {
    AThenB,
    BThenA
};

/** Per-evaluation-case arm-order audit record. */
struct EvaluationArmOrderRecord {
    std::string case_name;
    ArmRunOrder assigned_order = ArmRunOrder::AThenB;
    /** Arms for which a goal launch was attempted, in launch order ("A" / "B"). */
    std::vector<std::string> actual_starts;
};

struct HashedFile {
    std::string label;
    std::string sha256;
};

/** Model identity reported by a running inference service. Not a file hash. */
struct ServiceModelIdentity {
    std::string role;
    std::string id;
    int n_embd = 0;
    std::int64_t size = 0;
    std::string ftype;

    bool operator==(const ServiceModelIdentity& other) const {
        return role == other.role && id == other.id && n_embd == other.n_embd
            && size == other.size && ftype == other.ftype;
    }
};

/** Identity recorded for one evaluation arm and compared with the other arm. */
struct ArmIdentity {
    std::string engine_binary_sha256;
    std::string engine_library_sha256;
    std::string engine_image_digest;
    std::string retrieval_config_sha256;
    std::vector<HashedFile> model_files;
    std::vector<ServiceModelIdentity> service_models;
    std::vector<HashedFile> evidence_files;
    std::vector<HashedFile> goal_files;
};

struct EngineProcessStatus {
    bool exited = false;
    bool timed_out = false;
    int exit_code = -1;
};

struct SyntheticCase {
    std::string name;
    std::string goal_text;
    std::string evidence_name;
    std::string evidence_text;
};

struct SyntheticAbRequest {
    std::string engine_binary;
    std::string work_root;
    std::string retrieval_config_path;
    std::string engine_image = "thoth-engine:local";
    SyntheticExecution execution = SyntheticExecution::Mock;
    SyntheticAbPhase phase = SyntheticAbPhase::Full;
    /**
     * Path to the acquisition manifest written by AcquisitionOnly (or Full).
     * Empty means work_root/acquisition_manifest.json.
     */
    std::string acquisition_manifest_path;
    std::string inference_backend = "llama_cpp";
    std::string inference_base_url = "http://127.0.0.1:8080";
    std::string embed_base_url = "http://127.0.0.1:8081";
    bool require_engine_image = false;
    std::vector<std::string> model_files;
    std::vector<SyntheticCase> acquisition;
    std::vector<SyntheticCase> evaluation;
    int engine_timeout_seconds = 45;
    /**
     * When set, evaluation cases receive a deterministic balanced A→B / B→A
     * schedule from this seed. When unset, every case keeps legacy A→B order.
     * Unseeded mode is not an official EGAR-LX pilot schedule.
     */
    std::optional<std::uint64_t> arm_order_seed;
};

struct SyntheticAbReport {
    std::string execution = "mock";
    /** "full", "acquisition_only", or "evaluation_from_frozen". */
    std::string phase = "full";
    std::string acquisition_sha256;
    ExperienceTableCounts acquisition_counts;
    std::string empty_experience_sha256;
    std::string frozen_sha256_after_evaluation;
    /** True only after evaluation confirms the freeze was untouched. False for AcquisitionOnly. */
    bool frozen_unchanged = false;
    /** True only after evaluation confirms Arm A/B identity match. False for AcquisitionOnly. */
    bool identities_match = false;
    bool engine_image_verified = false;
    std::string engine_image_digest;
    std::string engine_library_path;
    std::string engine_library_sha256;
    std::vector<ServiceModelIdentity> service_models;
    ArmIdentity arm_a;
    ArmIdentity arm_b;
    std::string report_path;
    /** Absolute path of the written acquisition manifest when acquisition ran. */
    std::string acquisition_manifest_path;
    std::string frozen_database_path;
    std::string empty_template_path;
    bool arm_order_seeded = false;
    std::uint64_t arm_order_seed = 0;
    /** "legacy_fixed_a_then_b" or "seeded_balanced_shuffle_v1". */
    std::string arm_order_procedure;
    std::vector<EvaluationArmOrderRecord> evaluation_orders;
};

bool armIdentitiesMatch(const ArmIdentity& arm_a,
                         const ArmIdentity& arm_b,
                         std::string& error);

/**
 * Live identity. Compares the executable, the library the loader resolves,
 * and server-reported model records. An unavailable image digest is not a
 * match. It is required only when require_engine_image is true.
 */
bool liveArmIdentitiesMatch(const ArmIdentity& arm_a,
                             const ArmIdentity& arm_b,
                             bool require_engine_image,
                             std::string& error);

/**
 * Ask the dynamic loader which libbasic_agent.so this executable loads.
 * Fails when that path cannot be established.
 */
bool resolveLoadedAgentLibrary(const std::string& engine_binary,
                               std::string& library_path,
                               std::string& error);

/**
 * Snapshot only after a clean engine exit. A live, timed-out, or failed
 * process is refused before the source database is copied.
 */
bool snapshotAfterEngineExit(const EngineProcessStatus& status,
                             const std::string& source_database,
                             const std::string& snapshot_directory,
                             const ExperienceTableCounts* required_counts,
                             MemorySnapshot& snapshot,
                             std::string& error);

/**
 * Build the evaluation arm-order schedule. An empty seed keeps legacy A→B for
 * every case. A set seed yields an approximately balanced shuffle.
 */
std::vector<ArmRunOrder> buildArmOrderSchedule(std::size_t evaluation_count,
                                               const std::optional<std::uint64_t>& seed);

const char* armRunOrderLabel(ArmRunOrder order);

/**
 * Run the synthetic pilot according to request.phase.
 *
 * Full (default): acquisition, freeze, then evaluation.
 * AcquisitionOnly: acquisition and validated freeze; writes acquisition_manifest.json.
 * EvaluationFromFrozen: loads that manifest, verifies frozen hash and identity,
 * then runs evaluation only.
 */
bool runSyntheticAbPilot(const SyntheticAbRequest& request,
                         SyntheticAbReport& report,
                         std::string& error);

} // namespace Thoth

#endif
