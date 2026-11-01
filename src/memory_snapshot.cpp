/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — consistent SQLite snapshot of an exited engine workspace
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/memory_snapshot.h"
#include "../include/alp_sha256.h"

#include <sqlite3.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace Thoth {
namespace {

const char* kExperienceTables[] = {
    "trajectories",
    "strategies",
    "past_plans",
    "graph_edges",
    "facts",
};

struct FileSeal {
    bool present = false;
    std::uintmax_t size = 0;
    std::string sha256;
};

std::string hashFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const std::streamoff length = in.tellg();
    if (length < 0) {
        return {};
    }
    in.seekg(0, std::ios::beg);
    std::string bytes(static_cast<std::size_t>(length), '\0');
    if (length > 0) {
        in.read(bytes.data(), length);
        if (!in) {
            return {};
        }
    }
    return sha256Hex(bytes);
}

FileSeal sealFile(const fs::path& path) {
    FileSeal seal;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        return seal;
    }
    seal.size = fs::file_size(path, ec);
    if (ec) {
        return seal;
    }
    seal.sha256 = hashFile(path);
    if (seal.sha256.empty()) {
        return seal;
    }
    seal.present = true;
    return seal;
}

bool sameSeal(const FileSeal& before, const FileSeal& after) {
    return before.present == after.present && before.size == after.size
        && before.sha256 == after.sha256;
}

bool copyRegularFile(const fs::path& from, const fs::path& to, std::string& error) {
    std::error_code ec;
    fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        error = "memory snapshot copy failed: " + from.string() + ": " + ec.message();
        return false;
    }
    return true;
}

void removeIfPresent(const fs::path& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

bool tablePresent(sqlite3* db, const char* name) {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(
            db,
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?;",
            -1,
            &stmt,
            nullptr) != SQLITE_OK) {
        return false;
    }
    sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);
    const bool found = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_finalize(stmt);
    return found;
}

bool countTable(sqlite3* db, const char* name, long long& count, std::string& error) {
    const std::string sql = std::string("SELECT COUNT(*) FROM \"") + name + "\";";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        error = "memory snapshot could not count " + std::string(name) + ": "
            + sqlite3_errmsg(db);
        return false;
    }
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        error = "memory snapshot could not count " + std::string(name);
        return false;
    }
    count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return true;
}

bool readCounts(sqlite3* db, ExperienceTableCounts& counts, std::string& error) {
    for (const char* name : kExperienceTables) {
        if (!tablePresent(db, name)) {
            error = std::string("experience table is missing: ") + name;
            return false;
        }
    }
    return countTable(db, "trajectories", counts.trajectories, error)
        && countTable(db, "strategies", counts.strategies, error)
        && countTable(db, "past_plans", counts.past_plans, error)
        && countTable(db, "graph_edges", counts.graph_edges, error)
        && countTable(db, "facts", counts.facts, error);
}

bool integrityOk(sqlite3* db, std::string& error) {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, "PRAGMA integrity_check;", -1, &stmt, nullptr) != SQLITE_OK) {
        error = std::string("memory snapshot integrity_check failed: ") + sqlite3_errmsg(db);
        return false;
    }
    int rows = 0;
    std::string detail;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ++rows;
        const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const std::string value = text ? text : "";
        if (value != "ok") {
            if (!detail.empty()) {
                detail.push_back(';');
            }
            detail += value;
        }
    }
    sqlite3_finalize(stmt);
    if (rows != 1 || !detail.empty()) {
        error = "memory snapshot integrity_check failed: "
            + (detail.empty() ? "no result" : detail);
        return false;
    }
    return true;
}

bool checkpointCopy(sqlite3* db, std::string& error) {
    int log_frames = -1;
    int checkpointed = -1;
    const int rc = sqlite3_wal_checkpoint_v2(
        db, nullptr, SQLITE_CHECKPOINT_TRUNCATE, &log_frames, &checkpointed);
    if (rc != SQLITE_OK) {
        error = std::string("memory snapshot checkpoint failed: ") + sqlite3_errmsg(db);
        return false;
    }
    if (log_frames != checkpointed) {
        error = "memory snapshot checkpoint failed: frames remain in the WAL";
        return false;
    }
    return true;
}

class SqliteConnection {
public:
    explicit SqliteConnection(const fs::path& path, int flags, std::string& error) {
        const int rc = sqlite3_open_v2(path.c_str(), &db_, flags, nullptr);
        if (rc != SQLITE_OK) {
            error = "memory snapshot could not be opened: "
                + std::string(db_ ? sqlite3_errmsg(db_) : "open failed");
            if (db_) {
                sqlite3_close(db_);
                db_ = nullptr;
            }
        }
    }

    ~SqliteConnection() {
        if (db_) {
            sqlite3_close(db_);
        }
    }

    SqliteConnection(const SqliteConnection&) = delete;
    SqliteConnection& operator=(const SqliteConnection&) = delete;

    sqlite3* get() const { return db_; }
    explicit operator bool() const { return db_ != nullptr; }

private:
    sqlite3* db_ = nullptr;
};

} // namespace

bool readExperienceTableCounts(const std::string& database_path,
                               ExperienceTableCounts& counts,
                               std::string& error) {
    counts = ExperienceTableCounts{};
    SqliteConnection connection(database_path, SQLITE_OPEN_READONLY, error);
    if (!connection) {
        return false;
    }
    return readCounts(connection.get(), counts, error);
}

bool createConsistentMemorySnapshot(const std::string& source_database,
                                    const std::string& snapshot_directory,
                                    const ExperienceTableCounts* required_counts,
                                    MemorySnapshot& snapshot,
                                    std::string& error) {
    snapshot = MemorySnapshot{};
    const fs::path source(source_database);
    std::error_code ec;
    if (!fs::is_regular_file(source, ec) || ec) {
        error = "memory database is missing: " + source_database;
        return false;
    }

    const fs::path source_wal = fs::path(source.string() + "-wal");
    const fs::path source_shm = fs::path(source.string() + "-shm");
    const FileSeal source_db_before = sealFile(source);
    const FileSeal source_wal_before = sealFile(source_wal);
    const FileSeal source_shm_before = sealFile(source_shm);

    fs::path dest_dir(snapshot_directory);
    fs::create_directories(dest_dir, ec);
    if (ec) {
        error = "memory snapshot directory could not be created: " + ec.message();
        return false;
    }
    std::error_code canon_ec;
    const fs::path source_dir = fs::weakly_canonical(source.parent_path(), canon_ec);
    const fs::path dest_canon = fs::weakly_canonical(dest_dir, canon_ec);
    if (!source_dir.empty() && source_dir == dest_canon) {
        error = "snapshot directory must be distinct from the source database directory";
        return false;
    }

    const fs::path dest_db = dest_dir / "memory.db";
    const fs::path dest_wal = fs::path(dest_db.string() + "-wal");
    const fs::path dest_shm = fs::path(dest_db.string() + "-shm");
    removeIfPresent(dest_db);
    removeIfPresent(dest_wal);
    removeIfPresent(dest_shm);

    auto fail = [&](const std::string& message) {
        removeIfPresent(dest_db);
        removeIfPresent(dest_wal);
        removeIfPresent(dest_shm);
        const FileSeal source_db_after = sealFile(source);
        const FileSeal source_wal_after = sealFile(source_wal);
        const FileSeal source_shm_after = sealFile(source_shm);
        if (!sameSeal(source_db_before, source_db_after)
            || !sameSeal(source_wal_before, source_wal_after)
            || !sameSeal(source_shm_before, source_shm_after)) {
            error = "source database changed during snapshot";
            return false;
        }
        error = message;
        return false;
    };

    if (!copyRegularFile(source, dest_db, error)) {
        return fail(error);
    }
    if (source_wal_before.present && !copyRegularFile(source_wal, dest_wal, error)) {
        return fail(error);
    }
    if (source_shm_before.present && !copyRegularFile(source_shm, dest_shm, error)) {
        return fail(error);
    }

    {
        SqliteConnection connection(dest_db, SQLITE_OPEN_READWRITE, error);
        if (!connection) {
            return fail(error);
        }
        ExperienceTableCounts counts;
        if (!readCounts(connection.get(), counts, error)) {
            if (!source_wal_before.present) {
                error = "memory database is incomplete without its WAL: " + source_database;
            } else if (error.rfind("experience table is missing", 0) == 0) {
                error = "memory snapshot WAL is invalid: " + error;
            }
            return fail(error);
        }
        if (!checkpointCopy(connection.get(), error)) {
            return fail(error);
        }
        if (!integrityOk(connection.get(), error)) {
            return fail(error);
        }
        if (!readCounts(connection.get(), counts, error)) {
            return fail(error);
        }
        if (required_counts && !(counts == *required_counts)) {
            error = "memory snapshot experience counts mismatch";
            return fail(error);
        }
        snapshot.counts = counts;
    }

    removeIfPresent(dest_wal);
    removeIfPresent(dest_shm);
    snapshot.database_path = dest_db.string();
    snapshot.sha256 = hashFile(dest_db);
    if (snapshot.sha256.empty()) {
        return fail("memory snapshot hash failed");
    }

    const FileSeal source_db_after = sealFile(source);
    const FileSeal source_wal_after = sealFile(source_wal);
    const FileSeal source_shm_after = sealFile(source_shm);
    if (!sameSeal(source_db_before, source_db_after)
        || !sameSeal(source_wal_before, source_wal_after)
        || !sameSeal(source_shm_before, source_shm_after)) {
        error = "source database changed during snapshot";
        removeIfPresent(dest_db);
        snapshot = MemorySnapshot{};
        return false;
    }
    error.clear();
    return true;
}

bool restoreMemorySnapshot(const std::string& snapshot_database,
                           const std::string& destination_database,
                           std::string& error) {
    const fs::path source(snapshot_database);
    std::error_code ec;
    if (!fs::is_regular_file(source, ec) || ec) {
        error = "memory snapshot is missing: " + snapshot_database;
        return false;
    }
    const fs::path destination(destination_database);
    fs::create_directories(destination.parent_path(), ec);
    if (ec) {
        error = "memory snapshot restore directory could not be created: " + ec.message();
        return false;
    }
    if (!copyRegularFile(source, destination, error)) {
        return false;
    }
    if (hashFile(source) != hashFile(destination)) {
        error = "memory snapshot restore did not preserve the snapshot bytes";
        return false;
    }
    error.clear();
    return true;
}

} // namespace Thoth
