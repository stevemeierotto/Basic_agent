/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — consistent SQLite snapshot of an exited engine workspace
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_MEMORY_SNAPSHOT_H
#define THOTH_MEMORY_SNAPSHOT_H

#include <string>

namespace Thoth {

/** Rows the EGAR-LX arm check treats as acquired experience. */
struct ExperienceTableCounts {
    long long trajectories = 0;
    long long strategies = 0;
    long long past_plans = 0;
    long long graph_edges = 0;
    long long facts = 0;

    bool operator==(const ExperienceTableCounts& other) const {
        return trajectories == other.trajectories && strategies == other.strategies
            && past_plans == other.past_plans && graph_edges == other.graph_edges
            && facts == other.facts;
    }
};

struct MemorySnapshot {
    std::string database_path;
    std::string sha256;
    ExperienceTableCounts counts;
};

/**
 * Copy source_database and its WAL sidecar, then checkpoint only the copy.
 * The source files are not opened and are not checkpointed.
 * Call this only after the engine process that owns the source has exited.
 *
 * On success, snapshot_directory contains one checkpointed memory.db.
 * required_counts, when set, fails the snapshot unless those counts match.
 */
bool createConsistentMemorySnapshot(const std::string& source_database,
                                    const std::string& snapshot_directory,
                                    const ExperienceTableCounts* required_counts,
                                    MemorySnapshot& snapshot,
                                    std::string& error);

/** Byte copy of a checkpointed snapshot. Does not open or modify the frozen file. */
bool restoreMemorySnapshot(const std::string& snapshot_database,
                           const std::string& destination_database,
                           std::string& error);

/** Read the five experience tables. Opens the given database read-only. */
bool readExperienceTableCounts(const std::string& database_path,
                               ExperienceTableCounts& counts,
                               std::string& error);

} // namespace Thoth

#endif
