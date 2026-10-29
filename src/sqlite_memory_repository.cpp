#include "../include/sqlite_memory_repository.h"
#include "../include/decision_trace.h"
#include "../include/experience_provenance.h"
#include "../include/grag_scorer.h"
#include "../include/memory_consolidation_config.h"
#include <sqlite3.h>
#include <json.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <string>

namespace Thoth {

static std::string safe_col_text(sqlite3_stmt* stmt, int col) {
    const unsigned char* text = sqlite3_column_text(stmt, col);
    return text ? std::string(reinterpret_cast<const char*>(text)) : "";
}

static bool injectConsolidationFail(const char* stage) {
    const char* env = std::getenv("THOTH_INJECT_CONSOLIDATION_FAIL");
    return env && stage && std::string(env) == stage;
}

static bool injectRestoreFail(const char* stage) {
    const char* env = std::getenv("THOTH_INJECT_RESTORE_FAIL");
    return env && stage && std::string(env) == stage;
}

struct SQLiteMemoryRepository::DBHandle {
    sqlite3* handle = nullptr;
};

SQLiteMemoryRepository::SQLiteMemoryRepository(const std::string& dbPath)
    : db_(std::make_unique<DBHandle>()) 
{
    int rc = sqlite3_open(dbPath.c_str(), &db_->handle);
    if (rc != SQLITE_OK) {
        std::cerr << "[SQLiteMemoryRepository] Failed to open database: " << sqlite3_errmsg(db_->handle) << "\n";
        return;
    }

    // Harden the connection against transient lock contention. With a single
    // shared connection and no busy handler, SQLite returns SQLITE_BUSY
    // immediately under contention; a failed COMMIT then leaves the
    // transaction open and poisons every subsequent BEGIN. A busy timeout plus
    // WAL journaling let writers wait briefly and retry instead of failing.
    sqlite3_busy_timeout(db_->handle, 5000);
    sqlite3_set_authorizer(db_->handle, &SQLiteMemoryRepository::authorizeConnection, this);
    sqlite3_exec(db_->handle, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_->handle, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);

    try {
        const char* schema = 
            "CREATE TABLE IF NOT EXISTS sessions ("
            "  session_id TEXT PRIMARY KEY,"
            "  created_at_ms INTEGER NOT NULL,"
            "  updated_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  session_id TEXT NOT NULL,"
            "  role TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  timestamp_ms INTEGER NOT NULL,"
            "  FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS summaries ("
            "  session_id TEXT NOT NULL,"
            "  summary_type TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  updated_at_ms INTEGER NOT NULL,"
            "  PRIMARY KEY (session_id, summary_type),"
            "  FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS memory_meta ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS plan_embeddings ("
            "  plan_id TEXT NOT NULL,"
            "  type TEXT NOT NULL,"
            "  embedding BLOB NOT NULL,"
            "  embedding_version INTEGER NOT NULL,"
            "  PRIMARY KEY (plan_id, type, embedding_version)"
            ");"
            "CREATE TABLE IF NOT EXISTS past_plans ("
            "  plan_id TEXT PRIMARY KEY,"
            "  goal TEXT NOT NULL,"
            "  outline TEXT NOT NULL,"
            "  success_score REAL NOT NULL,"
            "  duration_ms INTEGER NOT NULL,"
            "  failure_count INTEGER NOT NULL,"
            "  goal_embedding BLOB NOT NULL,"
            "  embedding_version INTEGER NOT NULL,"
            "  plan_origin TEXT NOT NULL DEFAULT 'unknown'"
            ");"
            "CREATE TABLE IF NOT EXISTS graph_nodes ("
            "  id TEXT PRIMARY KEY,"
            "  file_path TEXT NOT NULL,"
            "  symbol TEXT,"
            "  type TEXT NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS graph_edges ("
            "  from_id TEXT NOT NULL,"
            "  to_id TEXT NOT NULL,"
            "  weight REAL NOT NULL,"
            "  success_count INTEGER DEFAULT 0,"
            "  failure_count INTEGER DEFAULT 0,"
            "  last_used_ms INTEGER NOT NULL,"
            "  PRIMARY KEY (from_id, to_id),"
            "  FOREIGN KEY (from_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,"
            "  FOREIGN KEY (to_id) REFERENCES graph_nodes(id) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS step_metrics ("
            "  step_id TEXT PRIMARY KEY,"
            "  plan_id TEXT NOT NULL,"
            "  tool_name TEXT,"
            "  latency_ms INTEGER NOT NULL,"
            "  retry_count INTEGER NOT NULL,"
            "  status TEXT NOT NULL,"
            "  timestamp_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS active_plans ("
            "  plan_id TEXT PRIMARY KEY,"
            "  session_id TEXT NOT NULL,"
            "  goal TEXT NOT NULL,"
            "  steps_json TEXT NOT NULL,"
            "  current_index INTEGER NOT NULL,"
            "  controller_state TEXT NOT NULL,"
            "  created_at_ms INTEGER NOT NULL,"
            "  updated_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS plans ("
            "  plan_id TEXT PRIMARY KEY,"
            "  goal TEXT NOT NULL,"
            "  plan_json TEXT NOT NULL,"
            "  status INTEGER NOT NULL,"
            "  success_score REAL NOT NULL,"
            "  embedding BLOB,"
            "  created_at INTEGER NOT NULL,"
            "  updated_at INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS archived_turns ("
            "  archive_id TEXT PRIMARY KEY,"
            "  session_id TEXT NOT NULL,"
            "  original_timestamp_ms INTEGER NOT NULL,"
            "  role TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  metadata_json TEXT,"
            "  archived_at_ms INTEGER NOT NULL,"
            "  summary_version INTEGER,"
            "  FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS warm_memory ("
            "  id TEXT PRIMARY KEY,"
            "  session_id TEXT NOT NULL,"
            "  scope INTEGER NOT NULL DEFAULT 0,"
            "  episodic_payload TEXT NOT NULL,"
            "  rendered_summary TEXT NOT NULL,"
            "  importance REAL NOT NULL,"
            "  novelty REAL NOT NULL,"
            "  confidence REAL NOT NULL,"
            "  covered_turn_start INTEGER,"
            "  covered_turn_end INTEGER,"
            "  covered_ts_start INTEGER,"
            "  covered_ts_end INTEGER,"
            "  parent_archive_ids TEXT,"
            "  derived_from_hash TEXT NOT NULL,"
            "  summary_version INTEGER NOT NULL,"
            "  prompt_version TEXT,"
            "  llm_model TEXT,"
            "  summary_missing INTEGER DEFAULT 0,"
            "  created_at_ms INTEGER NOT NULL,"
            "  FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS warm_memory_embeddings ("
            "  memory_id TEXT NOT NULL,"
            "  embedding BLOB NOT NULL,"
            "  embedding_version INTEGER NOT NULL,"
            "  PRIMARY KEY (memory_id, embedding_version),"
            "  FOREIGN KEY (memory_id) REFERENCES warm_memory(id) ON DELETE CASCADE"
            ");"
            "CREATE TABLE IF NOT EXISTS facts ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL,"
            "  confidence REAL DEFAULT 1.0,"
            "  source TEXT,"
            "  last_updated_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS episode_steps ("
            "  episode_id TEXT,"
            "  goal_id TEXT,"
            "  step_index INTEGER,"
            "  state_summary TEXT,"
            "  action_taken TEXT,"
            "  result_status TEXT,"
            "  embedding_blob BLOB,"
            "  timestamp_ms INTEGER"
            ");"
            "CREATE TABLE IF NOT EXISTS trajectories ("
            "  trajectory_id TEXT PRIMARY KEY,"
            "  goal TEXT NOT NULL,"
            "  trajectory_json TEXT NOT NULL,"
            "  success_score REAL NOT NULL,"
            "  embedding BLOB,"
            "  created_at INTEGER NOT NULL,"
            "  usage_count INTEGER DEFAULT 0,"
            "  tier INTEGER DEFAULT 0,"
            "  plan_origin TEXT NOT NULL DEFAULT 'unknown'"
            ");"
            "CREATE TABLE IF NOT EXISTS strategies ("
            "  strategy_id TEXT PRIMARY KEY,"
            "  description TEXT NOT NULL,"
            "  step_pattern_json TEXT NOT NULL,"
            "  success_rate REAL NOT NULL,"
            "  created_at INTEGER NOT NULL,"
            "  experience_origin TEXT NOT NULL DEFAULT 'unknown'"
            ");"
            "CREATE TABLE IF NOT EXISTS cognate_experiments ("
            "  experiment_id TEXT PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  hypothesis TEXT,"
            "  configuration_json TEXT,"
            "  results_json TEXT,"
            "  created_at INTEGER,"
            "  status TEXT"
            ");"
            "CREATE TABLE IF NOT EXISTS problem_states ("
            "  problem_id TEXT PRIMARY KEY,"
            "  goal_id TEXT NOT NULL,"
            "  state_json TEXT NOT NULL,"
            "  iteration_count INTEGER NOT NULL,"
            "  confidence_score REAL NOT NULL,"
            "  created_at INTEGER NOT NULL,"
            "  updated_at INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cognitive_metrics_outbox ("
            "  metrics_id TEXT PRIMARY KEY,"
            "  plan_id TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL,"
            "  flushed INTEGER NOT NULL DEFAULT 0"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_confidence_snapshots ("
            "  snapshot_id TEXT PRIMARY KEY,"
            "  plan_id TEXT NOT NULL,"
            "  step_id TEXT NOT NULL,"
            "  snapshot_type TEXT NOT NULL,"
            "  lifecycle_stage TEXT NOT NULL,"
            "  features_json TEXT NOT NULL,"
            "  measurement_status TEXT NOT NULL,"
            "  schema_version TEXT NOT NULL,"
            "  estimator_status TEXT NOT NULL,"
            "  created_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_answer_versions ("
            "  answer_id TEXT PRIMARY KEY,"
            "  plan_id TEXT NOT NULL,"
            "  step_id TEXT NOT NULL,"
            "  generation_id TEXT NOT NULL,"
            "  content_sha256 TEXT NOT NULL,"
            "  created_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_breadth_estimates ("
            "  estimate_id TEXT PRIMARY KEY,"
            "  snapshot_id TEXT NOT NULL,"
            "  plan_id TEXT NOT NULL,"
            "  step_id TEXT NOT NULL,"
            "  estimator_version TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL,"
            "  created_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_answer_feedback ("
            "  feedback_id TEXT PRIMARY KEY,"
            "  answer_id TEXT NOT NULL,"
            "  content_sha256 TEXT NOT NULL,"
            "  thumbs_status TEXT NOT NULL,"
            "  quality_rating_status TEXT NOT NULL,"
            "  quality_rating INTEGER,"
            "  explanation TEXT NOT NULL,"
            "  supersedes_feedback_id TEXT NOT NULL,"
            "  revision_index INTEGER NOT NULL,"
            "  created_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_verified_labels ("
            "  label_id TEXT PRIMARY KEY,"
            "  answer_id TEXT NOT NULL,"
            "  content_sha256 TEXT NOT NULL,"
            "  protocol_id TEXT NOT NULL,"
            "  reviewer_id TEXT NOT NULL,"
            "  factual_bin INTEGER NOT NULL,"
            "  supersedes_label_id TEXT NOT NULL,"
            "  revision_index INTEGER NOT NULL,"
            "  created_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_evidence_judgments ("
            "  judgment_id TEXT PRIMARY KEY,"
            "  answer_id TEXT NOT NULL,"
            "  content_sha256 TEXT NOT NULL,"
            "  estimator_version TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL,"
            "  created_at_ms INTEGER NOT NULL"
            ");"
            "CREATE TABLE IF NOT EXISTS cm1_claim_verifications ("
            "  verification_id TEXT PRIMARY KEY,"
            "  claim_id TEXT NOT NULL,"
            "  claim_sha256 TEXT NOT NULL,"
            "  answer_id TEXT NOT NULL,"
            "  content_sha256 TEXT NOT NULL,"
            "  protocol_id TEXT NOT NULL,"
            "  protocol_version TEXT NOT NULL,"
            "  reviewer_id TEXT NOT NULL,"
            "  verifier_kind TEXT NOT NULL,"
            "  verifier_version TEXT NOT NULL,"
            "  method_id TEXT NOT NULL,"
            "  outcome TEXT NOT NULL,"
            "  source_kind TEXT NOT NULL,"
            "  source_ref TEXT NOT NULL,"
            "  calibration_eligibility TEXT NOT NULL,"
            "  egar_eligibility TEXT NOT NULL,"
            "  revision_reason TEXT NOT NULL,"
            "  supersedes_verification_id TEXT NOT NULL,"
            "  revision_index INTEGER NOT NULL,"
            "  created_at_ms INTEGER NOT NULL,"
            "  evidence_label TEXT NOT NULL DEFAULT ''"
            ");";

        char* errMsg = nullptr;
        rc = sqlite3_exec(db_->handle, schema, nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "[SQLiteMemoryRepository] Schema creation failed: " << (errMsg ? errMsg : "Unknown error") << "\n";
            sqlite3_free(errMsg);
        }

        // Session table migrations
        auto add_col = [&](const std::string& table, const std::string& col, const std::string& type) -> bool {
            bool exists = false;
            sqlite3_stmt* p;
            std::string pragma = "PRAGMA table_info(" + table + ");";
            if (sqlite3_prepare_v2(db_->handle, pragma.c_str(), -1, &p, nullptr) == SQLITE_OK) {
                while (sqlite3_step(p) == SQLITE_ROW) {
                    if (safe_col_text(p, 1) == col) { exists = true; break; }
                }
                sqlite3_finalize(p);
            }
            if (!exists) {
                std::string alter = "ALTER TABLE " + table + " ADD COLUMN " + col + " " + type + ";";
                sqlite3_exec(db_->handle, alter.c_str(), nullptr, nullptr, nullptr);
                return true;
            }
            return false;
        };

        add_col("sessions", "title", "TEXT");
        add_col("sessions", "active_goal", "TEXT");
        add_col("sessions", "summary", "TEXT");
        add_col("sessions", "metadata_json", "TEXT");

        // Trajectory table migrations
        add_col("trajectories", "trajectory_json", "TEXT");
        add_col("trajectories", "success_score", "REAL");
        add_col("trajectories", "created_at", "INTEGER");
        add_col("trajectories", "usage_count", "INTEGER");
        add_col("trajectories", "tier", "INTEGER");
        const bool added_trajectory_origin =
            add_col("trajectories", "plan_origin", "TEXT DEFAULT 'unknown'");
        const bool added_past_plan_origin =
            add_col("past_plans", "plan_origin", "TEXT DEFAULT 'unknown'");

        // Strategy table migrations
        add_col("strategies", "description", "TEXT");
        add_col("strategies", "step_pattern_json", "TEXT");
        add_col("strategies", "success_rate", "REAL");
        add_col("strategies", "created_at", "INTEGER");
        add_col("strategies", "experience_origin", "TEXT DEFAULT 'unknown'");
        add_col("cm1_claim_verifications", "evidence_label", "TEXT NOT NULL DEFAULT ''");

        auto backfill_origin = [&](const char* select_sql, const char* update_sql, const char* pointer) {
            sqlite3_stmt* select = nullptr;
            if (sqlite3_prepare_v2(db_->handle, select_sql, -1, &select, nullptr) != SQLITE_OK) {
                return;
            }
            while (sqlite3_step(select) == SQLITE_ROW) {
                const std::string id = safe_col_text(select, 0);
                const std::string stored = safe_col_text(select, 1);
                if (normalizePlanOrigin(stored) != kPlanOriginUnknown) {
                    continue;
                }
                const std::string body = safe_col_text(select, 2);
                std::string explicit_origin = kPlanOriginUnknown;
                try {
                    const nlohmann::json parsed = nlohmann::json::parse(body);
                    const nlohmann::json* node = &parsed;
                    if (pointer != nullptr && parsed.contains(pointer) && parsed[pointer].is_object()) {
                        node = &parsed[pointer];
                    }
                    if (node->contains("plan_origin") && (*node)["plan_origin"].is_string()) {
                        explicit_origin = normalizePlanOrigin((*node)["plan_origin"].get<std::string>());
                    }
                } catch (...) {
                    explicit_origin = kPlanOriginUnknown;
                }
                if (explicit_origin == kPlanOriginUnknown) {
                    continue;
                }
                sqlite3_stmt* update = nullptr;
                if (sqlite3_prepare_v2(db_->handle, update_sql, -1, &update, nullptr) != SQLITE_OK) {
                    continue;
                }
                sqlite3_bind_text(update, 1, explicit_origin.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(update, 2, id.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_step(update);
                sqlite3_finalize(update);
            }
            sqlite3_finalize(select);
        };
        if (added_past_plan_origin) {
            backfill_origin(
                "SELECT plan_id, plan_origin, outline FROM past_plans;",
                "UPDATE past_plans SET plan_origin = ? WHERE plan_id = ?;",
                nullptr);
        }
        if (added_trajectory_origin) {
            backfill_origin(
                "SELECT trajectory_id, plan_origin, trajectory_json FROM trajectories;",
                "UPDATE trajectories SET plan_origin = ? WHERE trajectory_id = ?;",
                "plan_initial");
        }

        configureEpisodeIdentity();

    } catch (const std::exception& e) {
        std::cerr << "[SQLiteMemoryRepository] Initialization exception: " << e.what() << "\n";
    }
}

SQLiteMemoryRepository::~SQLiteMemoryRepository() {
    if (db_ && db_->handle && sqlite3_get_autocommit(db_->handle) == 0) {
        sqlite3_exec(db_->handle, "ROLLBACK;", nullptr, nullptr, nullptr);
    }
}

int SQLiteMemoryRepository::authorizeConnection(void* user, int, const char*, const char*,
                                                const char*, const char*) {
    auto* self = static_cast<SQLiteMemoryRepository*>(user);
    if (!self) {
        return SQLITE_OK;
    }
    std::lock_guard<std::mutex> lock(self->transaction_mu_);
    if (self->transaction_held_ && std::this_thread::get_id() != self->transaction_owner_) {
        return SQLITE_DENY;
    }
    return SQLITE_OK;
}

void SQLiteMemoryRepository::clearTransactionOwnership() {
    std::lock_guard<std::mutex> lock(transaction_mu_);
    transaction_held_ = false;
}

bool SQLiteMemoryRepository::beginTransaction() {
    if (!db_ || !db_->handle) return false;
    // An open transaction belongs to its owner. Rolling it back here would let
    // an unrelated caller discard graph or aging work. Refuse instead.
    if (sqlite3_get_autocommit(db_->handle) == 0) {
        std::cerr << "[SQLiteMemoryRepository] Refusing BEGIN while a transaction is open.\n";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(transaction_mu_);
        transaction_owner_ = std::this_thread::get_id();
        transaction_held_ = true;
    }
    if (sqlite3_exec(db_->handle, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) != SQLITE_OK) {
        clearTransactionOwnership();
        return false;
    }
    return true;
}

bool SQLiteMemoryRepository::commit() {
    if (!db_ || !db_->handle) return false;
    if (sqlite3_exec(db_->handle, "COMMIT;", nullptr, nullptr, nullptr) == SQLITE_OK) {
        clearTransactionOwnership();
        return true;
    }
    // COMMIT failed (e.g. SQLITE_BUSY). Leaving the transaction open would
    // poison every future beginTransaction() on this shared connection, so
    // roll back to return the connection to autocommit mode before reporting
    // failure to the caller.
    std::cerr << "[SQLiteMemoryRepository] COMMIT failed: " << sqlite3_errmsg(db_->handle)
              << " — rolling back to release transaction.\n";
    if (sqlite3_get_autocommit(db_->handle) == 0) {
        sqlite3_exec(db_->handle, "ROLLBACK;", nullptr, nullptr, nullptr);
    }
    clearTransactionOwnership();
    return false;
}

bool SQLiteMemoryRepository::rollback() {
    if (!db_ || !db_->handle) return false;
    // No-op if no transaction is active, otherwise SQLite returns a harmless
    // but noisy "cannot rollback - no transaction is active" error.
    if (sqlite3_get_autocommit(db_->handle) != 0) {
        clearTransactionOwnership();
        return true;
    }
    const bool rolled_back =
        sqlite3_exec(db_->handle, "ROLLBACK;", nullptr, nullptr, nullptr) == SQLITE_OK;
    clearTransactionOwnership();
    return rolled_back;
}

bool SQLiteMemoryRepository::createSession(const std::string& sessionId, int64_t createdAtMs) {
    try {
        const char* sql = "INSERT OR IGNORE INTO sessions (session_id, created_at_ms, updated_at_ms) VALUES (?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 2, createdAtMs);
        sqlite3_bind_int64(stmt, 3, createdAtMs);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<SessionRecord> SQLiteMemoryRepository::getSession(const std::string& sessionId) {
    try {
        const char* sql = "SELECT session_id, created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        std::optional<SessionRecord> record;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            record = SessionRecord{
                safe_col_text(stmt, 0),
                sqlite3_column_int64(stmt, 1),
                sqlite3_column_int64(stmt, 2)
            };
        }
        sqlite3_finalize(stmt);
        return record;
    } catch (...) { return std::nullopt; }
}

std::vector<std::string> SQLiteMemoryRepository::getAllSessionIds() {
    std::vector<std::string> ids;
    try {
        const char* sql = "SELECT session_id FROM sessions;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                ids.push_back(safe_col_text(stmt, 0));
            }
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return ids;
}

bool SQLiteMemoryRepository::appendMessage(const std::string& sessionId, const MessageRecord& msg) {
    try {
        const char* sql = "INSERT INTO messages (session_id, role, content, timestamp_ms) VALUES (?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, msg.role.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, msg.content.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 4, msg.timestamp_ms);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        
        if (success) {
            const char* upSql = "UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?;";
            if (sqlite3_prepare_v2(db_->handle, upSql, -1, &stmt, nullptr) == SQLITE_OK) {
                sqlite3_bind_int64(stmt, 1, msg.timestamp_ms);
                sqlite3_bind_text(stmt, 2, sessionId.c_str(), -1, SQLITE_STATIC);
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);
            }
        }
        return success;
    } catch (...) { return false; }
}

std::vector<MessageRecord> SQLiteMemoryRepository::getMessages(const std::string& sessionId) {
    std::vector<MessageRecord> messages;
    try {
        const char* sql = "SELECT role, content, timestamp_ms FROM messages WHERE session_id = ? ORDER BY timestamp_ms ASC;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return messages;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            messages.push_back({
                safe_col_text(stmt, 0),
                safe_col_text(stmt, 1),
                sqlite3_column_int64(stmt, 2)
            });
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return messages;
}

bool SQLiteMemoryRepository::clearMessages(const std::string& sessionId) {
    try {
        const char* sql = "DELETE FROM messages WHERE session_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

bool SQLiteMemoryRepository::storeSummary(const std::string& sessionId, const std::string& type, const std::string& content) {
    try {
        const char* sql = "INSERT OR REPLACE INTO summaries (session_id, summary_type, content, updated_at_ms) VALUES (?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, type.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, content.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 4, std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::string SQLiteMemoryRepository::getSummary(const std::string& sessionId, const std::string& type) {
    std::string content = "";
    try {
        const char* sql = "SELECT content FROM summaries WHERE session_id = ? AND summary_type = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return content;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, type.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            content = safe_col_text(stmt, 0);
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return content;
}

bool SQLiteMemoryRepository::setMeta(const std::string& key, const std::string& value) {
    try {
        const char* sql = "INSERT OR REPLACE INTO memory_meta (key, value) VALUES (?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_STATIC);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::string SQLiteMemoryRepository::getMeta(const std::string& key) {
    std::string val = "";
    try {
        const char* sql = "SELECT value FROM memory_meta WHERE key = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return val;
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            val = safe_col_text(stmt, 0);
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return val;
}

bool SQLiteMemoryRepository::storePlanEmbedding(const std::string& planId, const std::string& type, const std::vector<float>& embedding, int version) {
    try {
        const char* sql = "INSERT OR REPLACE INTO plan_embeddings (plan_id, type, embedding, embedding_version) VALUES (?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, planId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, type.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 3, embedding.data(), static_cast<int>(embedding.size() * sizeof(float)), SQLITE_STATIC);
        sqlite3_bind_int(stmt, 4, version);
        
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::vector<float> SQLiteMemoryRepository::getPlanEmbedding(const std::string& planId, const std::string& type, int version) {
    std::vector<float> embedding;
    try {
        const char* sql = "SELECT embedding FROM plan_embeddings WHERE plan_id = ? AND type = ? AND embedding_version = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return embedding;
        
        sqlite3_bind_text(stmt, 1, planId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, type.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 3, version);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const void* blob = sqlite3_column_blob(stmt, 0);
            int bytes = sqlite3_column_bytes(stmt, 0);
            int count = bytes / sizeof(float);
            embedding.resize(count);
            std::memcpy(embedding.data(), blob, bytes);
        }
        
        sqlite3_finalize(stmt);
    } catch (...) {}
    return embedding;
}

bool SQLiteMemoryRepository::clearPlanEmbeddings(int version) {
    try {
        const char* sql = "DELETE FROM plan_embeddings WHERE embedding_version = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_int(stmt, 1, version);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

bool SQLiteMemoryRepository::storePastPlan(const PastPlanRecord& plan, int version) {
    try {
        const char* sql = "INSERT OR REPLACE INTO past_plans (plan_id, goal, outline, success_score, duration_ms, failure_count, goal_embedding, embedding_version, plan_origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, plan.plan_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, plan.goal.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, plan.outline.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, plan.success_score);
        sqlite3_bind_int64(stmt, 5, plan.duration_ms);
        sqlite3_bind_int(stmt, 6, plan.failure_count);
        // An empty vector has no buffer. Binding that pointer stores NULL and
        // violates NOT NULL. A zero-length blob keeps the required write.
        const float emptyEmbedding = 0.0f;
        const void* embeddingBytes = plan.goal_embedding.empty()
            ? static_cast<const void*>(&emptyEmbedding)
            : static_cast<const void*>(plan.goal_embedding.data());
        const int embeddingBytesCount = plan.goal_embedding.empty()
            ? 0
            : static_cast<int>(plan.goal_embedding.size() * sizeof(float));
        sqlite3_bind_blob(stmt, 7, embeddingBytes, embeddingBytesCount, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 8, version);
        const std::string origin = normalizePlanOrigin(plan.plan_origin);
        sqlite3_bind_text(stmt, 9, origin.c_str(), -1, SQLITE_TRANSIENT);
        
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::vector<MemoryRepository::PastPlanRecord> SQLiteMemoryRepository::getAllPastPlans(int version) {
    std::vector<PastPlanRecord> plans;
    try {
        const char* sql = "SELECT plan_id, goal, outline, success_score, duration_ms, failure_count, goal_embedding, plan_origin FROM past_plans WHERE embedding_version = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return plans;
        
        sqlite3_bind_int(stmt, 1, version);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            PastPlanRecord p;
            p.plan_id = safe_col_text(stmt, 0);
            p.goal = safe_col_text(stmt, 1);
            p.outline = safe_col_text(stmt, 2);
            p.success_score = static_cast<float>(sqlite3_column_double(stmt, 3));
            p.duration_ms = sqlite3_column_int64(stmt, 4);
            p.failure_count = sqlite3_column_int(stmt, 5);
            
            const void* blob = sqlite3_column_blob(stmt, 6);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 6);
                int count = bytes / sizeof(float);
                p.goal_embedding.resize(count);
                std::memcpy(p.goal_embedding.data(), blob, bytes);
            }
            p.plan_origin = normalizePlanOrigin(safe_col_text(stmt, 7));
            
            plans.push_back(std::move(p));
        }
        
        sqlite3_finalize(stmt);
    } catch (...) {}
    return plans;
}

bool SQLiteMemoryRepository::addNode(const Node& node) {
    try {
        const char* sql = "INSERT OR REPLACE INTO graph_nodes (id, file_path, symbol, type) VALUES (?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, node.id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, node.file_path.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, node.symbol.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, node.type.c_str(), -1, SQLITE_STATIC);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

bool SQLiteMemoryRepository::addEdge(const Edge& edge) {
    try {
        // GraphRefiner and processMemoryAging both pass absolute counters.
        // The conflict update stores those values. Aging rewrites weight without
        // adding the outcome again.
        const char* sql = "INSERT INTO graph_edges (from_id, to_id, weight, success_count, failure_count, last_used_ms) VALUES (?, ?, ?, ?, ?, ?) "
                          "ON CONFLICT(from_id, to_id) DO UPDATE SET weight=excluded.weight, success_count=excluded.success_count, "
                          "failure_count=excluded.failure_count, last_used_ms=excluded.last_used_ms;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, edge.from_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, edge.to_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 3, static_cast<double>(edge.weight));
        sqlite3_bind_int(stmt, 4, edge.success_count);
        sqlite3_bind_int(stmt, 5, edge.failure_count);
        sqlite3_bind_int64(stmt, 6, edge.last_used_ms);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::vector<MemoryRepository::Node> SQLiteMemoryRepository::getNodesByType(const std::string& type) {
    std::vector<Node> nodes;
    try {
        const char* sql = "SELECT id, file_path, symbol, type FROM graph_nodes WHERE type = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return nodes;

        sqlite3_bind_text(stmt, 1, type.c_str(), -1, SQLITE_STATIC);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Node n;
            n.id = safe_col_text(stmt, 0);
            n.file_path = safe_col_text(stmt, 1);
            n.symbol = safe_col_text(stmt, 2);
            n.type = safe_col_text(stmt, 3);
            nodes.push_back(std::move(n));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return nodes;
}

std::vector<MemoryRepository::Edge> SQLiteMemoryRepository::getEdgesFrom(const std::string& nodeId) {
    std::vector<Edge> edges;
    try {
        const char* sql = "SELECT from_id, to_id, weight, success_count, failure_count, last_used_ms FROM graph_edges WHERE from_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return edges;

        sqlite3_bind_text(stmt, 1, nodeId.c_str(), -1, SQLITE_STATIC);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Edge e;
            e.from_id = safe_col_text(stmt, 0);
            e.to_id = safe_col_text(stmt, 1);
            e.weight = static_cast<float>(sqlite3_column_double(stmt, 2));
            e.success_count = sqlite3_column_int(stmt, 3);
            e.failure_count = sqlite3_column_int(stmt, 4);
            e.last_used_ms = sqlite3_column_int64(stmt, 5);
            edges.push_back(std::move(e));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return edges;
}

std::vector<MemoryRepository::Edge> SQLiteMemoryRepository::getAllEdges() {
    std::vector<Edge> edges;
    try {
        const char* sql = "SELECT from_id, to_id, weight, success_count, failure_count, last_used_ms FROM graph_edges;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return edges;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Edge e;
            e.from_id = safe_col_text(stmt, 0);
            e.to_id = safe_col_text(stmt, 1);
            e.weight = static_cast<float>(sqlite3_column_double(stmt, 2));
            e.success_count = sqlite3_column_int(stmt, 3);
            e.failure_count = sqlite3_column_int(stmt, 4);
            e.last_used_ms = sqlite3_column_int64(stmt, 5);
            edges.push_back(std::move(e));
        }

        sqlite3_finalize(stmt);
    } catch (...) {}
    return edges;
}

bool SQLiteMemoryRepository::deleteEdge(const std::string& from_id, const std::string& to_id) {
    try {
        const char* sql = "DELETE FROM graph_edges WHERE from_id = ? AND to_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, from_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, to_id.c_str(), -1, SQLITE_STATIC);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

MemoryRepository::GraphStatistics SQLiteMemoryRepository::getGraphStatistics() {
    GraphStatistics stats;
    try {
        const char* sql_nodes = "SELECT COUNT(*) FROM graph_nodes;";
        sqlite3_stmt* stmt_nodes;
        if (sqlite3_prepare_v2(db_->handle, sql_nodes, -1, &stmt_nodes, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt_nodes) == SQLITE_ROW) {
                stats.total_nodes = sqlite3_column_int(stmt_nodes, 0);
            }
            sqlite3_finalize(stmt_nodes);
        }

        const char* sql_edges = "SELECT COUNT(*), AVG(weight), MIN(weight), MAX(weight), SUM(success_count), SUM(failure_count) FROM graph_edges;";
        sqlite3_stmt* stmt_edges;
        if (sqlite3_prepare_v2(db_->handle, sql_edges, -1, &stmt_edges, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt_edges) == SQLITE_ROW) {
                stats.total_edges = sqlite3_column_int(stmt_edges, 0);
                if (stats.total_edges > 0) {
                    stats.avg_edge_weight = static_cast<float>(sqlite3_column_double(stmt_edges, 1));
                    stats.min_edge_weight = static_cast<float>(sqlite3_column_double(stmt_edges, 2));
                    stats.max_edge_weight = static_cast<float>(sqlite3_column_double(stmt_edges, 3));
                    stats.total_success_count = sqlite3_column_int(stmt_edges, 4);
                    stats.total_failure_count = sqlite3_column_int(stmt_edges, 5);
                }
            }
            sqlite3_finalize(stmt_edges);
        }
    } catch (...) {}
    return stats;
}

bool SQLiteMemoryRepository::storeStepMetric(const StepMetricRecord& metric) {
    try {
        const char* sql = "INSERT OR REPLACE INTO step_metrics (step_id, plan_id, tool_name, latency_ms, retry_count, status, timestamp_ms) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, metric.step_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, metric.plan_id.c_str(), -1, SQLITE_STATIC);
        if (metric.tool_name.empty()) {
            sqlite3_bind_null(stmt, 3);
        } else {
            sqlite3_bind_text(stmt, 3, metric.tool_name.c_str(), -1, SQLITE_STATIC);
        }
        sqlite3_bind_int64(stmt, 4, metric.latency_ms);
        sqlite3_bind_int(stmt, 5, metric.retry_count);
        sqlite3_bind_text(stmt, 6, metric.status.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 7, metric.timestamp_ms);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

void SQLiteMemoryRepository::setActivePlanCommitObserverForTests(ActivePlanCommitObserver observer) {
    std::lock_guard<std::mutex> lock(active_plan_observer_mu_);
    active_plan_observer_ = std::move(observer);
}

bool SQLiteMemoryRepository::storeActivePlan(const ActivePlanRecord& plan) {
    // Copy the observer out before any SQLite work. The callback runs later,
    // without this mutex, so it cannot deadlock against a nested install.
    ActivePlanCommitObserver observer;
    {
        std::lock_guard<std::mutex> lock(active_plan_observer_mu_);
        observer = active_plan_observer_;
    }

    bool statement_done = false;
    bool autocommit_after_step = false;
    try {
        const char* sql = "INSERT OR REPLACE INTO active_plans (plan_id, session_id, goal, steps_json, current_index, controller_state, created_at_ms, updated_at_ms) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, plan.plan_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, plan.session_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, plan.goal.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, plan.steps_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 5, plan.current_index);
        sqlite3_bind_text(stmt, 6, plan.controller_state.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 7, plan.created_at_ms);
        sqlite3_bind_int64(stmt, 8, plan.updated_at_ms);

        // Autocommit INSERT commits inside sqlite3_step. SQLITE_DONE means that
        // commit has finished. sqlite3_get_autocommit() != 0 then confirms the
        // connection is not still inside an explicit transaction.
        statement_done = (sqlite3_step(stmt) == SQLITE_DONE);
        autocommit_after_step = db_ && db_->handle && sqlite3_get_autocommit(db_->handle) != 0;
        sqlite3_finalize(stmt);
    } catch (...) {
        return false;
    }

    if (!statement_done) return false;

    // Statement is finalized. A second connection can read the row. Production
    // callers with a null observer skip this block entirely.
    if (autocommit_after_step && observer) {
        try {
            observer(plan);
        } catch (const std::exception& ex) {
            std::cerr << "[SQLiteMemoryRepository] active-plan commit observer failed: "
                      << ex.what() << "\n";
        } catch (...) {
            std::cerr << "[SQLiteMemoryRepository] active-plan commit observer failed\n";
        }
    }
    return true;
}

std::optional<MemoryRepository::ActivePlanRecord> SQLiteMemoryRepository::getActivePlan(const std::string& session_id) {
    try {
        const char* sql = "SELECT plan_id, session_id, goal, steps_json, current_index, controller_state, created_at_ms, updated_at_ms "
                          "FROM active_plans WHERE session_id = ? ORDER BY updated_at_ms DESC LIMIT 1;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

        sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_STATIC);

        std::optional<ActivePlanRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            ActivePlanRecord rec;
            rec.plan_id = safe_col_text(stmt, 0);
            rec.session_id = safe_col_text(stmt, 1);
            rec.goal = safe_col_text(stmt, 2);
            rec.steps_json = safe_col_text(stmt, 3);
            rec.current_index = sqlite3_column_int(stmt, 4);
            rec.controller_state = safe_col_text(stmt, 5);
            rec.created_at_ms = sqlite3_column_int64(stmt, 6);
            rec.updated_at_ms = sqlite3_column_int64(stmt, 7);
            result = rec;
        }

        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

bool SQLiteMemoryRepository::deleteActivePlan(const std::string& plan_id) {
    try {
        const char* sql = "DELETE FROM active_plans WHERE plan_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, plan_id.c_str(), -1, SQLITE_STATIC);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

bool SQLiteMemoryRepository::saveCognatePlan(const CognatePlanRecord& record) {
    try {
        const char* sql = "INSERT OR REPLACE INTO plans (plan_id, goal, plan_json, status, success_score, embedding, created_at, updated_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, record.plan_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, record.goal.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, record.plan_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 4, record.status);
        sqlite3_bind_double(stmt, 5, static_cast<double>(record.success_score));
        if (!record.embedding.empty()) {
            sqlite3_bind_blob(stmt, 6, record.embedding.data(), static_cast<int>(record.embedding.size() * sizeof(float)), SQLITE_STATIC);
        } else {
            sqlite3_bind_null(stmt, 6);
        }
        sqlite3_bind_int64(stmt, 7, record.created_at);
        sqlite3_bind_int64(stmt, 8, record.updated_at);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::CognatePlanRecord> SQLiteMemoryRepository::loadCognatePlan(const std::string& plan_id) {
    try {
        const char* sql = "SELECT plan_id, goal, plan_json, status, success_score, embedding, created_at, updated_at FROM plans WHERE plan_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

        sqlite3_bind_text(stmt, 1, plan_id.c_str(), -1, SQLITE_STATIC);

        std::optional<CognatePlanRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            CognatePlanRecord rec;
            rec.plan_id = safe_col_text(stmt, 0);
            rec.goal = safe_col_text(stmt, 1);
            rec.plan_json = safe_col_text(stmt, 2);
            rec.status = sqlite3_column_int(stmt, 3);
            rec.success_score = static_cast<float>(sqlite3_column_double(stmt, 4));
            
            const void* blob = sqlite3_column_blob(stmt, 5);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 5);
                int count = bytes / sizeof(float);
                rec.embedding.resize(count);
                std::memcpy(rec.embedding.data(), blob, bytes);
            }

            rec.created_at = sqlite3_column_int64(stmt, 6);
            rec.updated_at = sqlite3_column_int64(stmt, 7);
            result = rec;
        }

        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::vector<MemoryRepository::CognatePlanRecord> SQLiteMemoryRepository::getAllCognatePlans() {
    std::vector<CognatePlanRecord> plans;
    try {
        const char* sql = "SELECT plan_id, goal, plan_json, status, success_score, embedding, created_at, updated_at FROM plans;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return plans;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            CognatePlanRecord rec;
            rec.plan_id = safe_col_text(stmt, 0);
            rec.goal = safe_col_text(stmt, 1);
            rec.plan_json = safe_col_text(stmt, 2);
            rec.status = sqlite3_column_int(stmt, 3);
            rec.success_score = static_cast<float>(sqlite3_column_double(stmt, 4));
            
            const void* blob = sqlite3_column_blob(stmt, 5);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 5);
                int count = bytes / sizeof(float);
                rec.embedding.resize(count);
                std::memcpy(rec.embedding.data(), blob, bytes);
            }

            rec.created_at = sqlite3_column_int64(stmt, 6);
            rec.updated_at = sqlite3_column_int64(stmt, 7);
            plans.push_back(std::move(rec));
        }

        sqlite3_finalize(stmt);
    } catch (...) {}
    return plans;
}

bool SQLiteMemoryRepository::saveTrajectory(const CognateTrajectoryRecord& record) {
    try {
        const char* sql = "INSERT OR REPLACE INTO trajectories (trajectory_id, goal, trajectory_json, success_score, embedding, created_at, usage_count, tier, plan_origin) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, record.trajectory_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, record.goal.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, record.trajectory_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, static_cast<double>(record.success_score));
        if (!record.embedding.empty()) {
            sqlite3_bind_blob(stmt, 5, record.embedding.data(), static_cast<int>(record.embedding.size() * sizeof(float)), SQLITE_STATIC);
        } else {
            sqlite3_bind_null(stmt, 5);
        }
        sqlite3_bind_int64(stmt, 6, record.created_at);
        sqlite3_bind_int(stmt, 7, record.usage_count);
        sqlite3_bind_int(stmt, 8, record.tier);
        const std::string origin = normalizePlanOrigin(record.plan_origin);
        sqlite3_bind_text(stmt, 9, origin.c_str(), -1, SQLITE_TRANSIENT);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::CognateTrajectoryRecord> SQLiteMemoryRepository::loadTrajectory(const std::string& trajectory_id) {
    try {
        const char* sql = "SELECT trajectory_id, goal, trajectory_json, success_score, embedding, created_at, usage_count, tier, plan_origin FROM trajectories WHERE trajectory_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

        sqlite3_bind_text(stmt, 1, trajectory_id.c_str(), -1, SQLITE_STATIC);

        std::optional<CognateTrajectoryRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateTrajectoryRecord rec;
            rec.trajectory_id = safe_col_text(stmt, 0);
            rec.goal = safe_col_text(stmt, 1);
            rec.trajectory_json = safe_col_text(stmt, 2);
            rec.success_score = static_cast<float>(sqlite3_column_double(stmt, 3));
            
            const void* blob = sqlite3_column_blob(stmt, 4);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 4);
                int count = bytes / sizeof(float);
                rec.embedding.resize(count);
                std::memcpy(rec.embedding.data(), blob, bytes);
            }

            rec.created_at = sqlite3_column_int64(stmt, 5);
            rec.usage_count = sqlite3_column_int(stmt, 6);
            rec.tier = sqlite3_column_int(stmt, 7);
            rec.plan_origin = normalizePlanOrigin(safe_col_text(stmt, 8));
            result = rec;
        }

        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::vector<MemoryRepository::CognateTrajectoryRecord> SQLiteMemoryRepository::getAllTrajectories() {
    std::vector<CognateTrajectoryRecord> results;
    try {
        const char* sql = "SELECT trajectory_id, goal, trajectory_json, success_score, embedding, created_at, usage_count, tier, plan_origin FROM trajectories;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateTrajectoryRecord rec;
            rec.trajectory_id = safe_col_text(stmt, 0);
            rec.goal = safe_col_text(stmt, 1);
            rec.trajectory_json = safe_col_text(stmt, 2);
            rec.success_score = static_cast<float>(sqlite3_column_double(stmt, 3));
            
            const void* blob = sqlite3_column_blob(stmt, 4);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 4);
                int count = bytes / sizeof(float);
                rec.embedding.resize(count);
                std::memcpy(rec.embedding.data(), blob, bytes);
            }

            rec.created_at = sqlite3_column_int64(stmt, 5);
            rec.usage_count = sqlite3_column_int(stmt, 6);
            rec.tier = sqlite3_column_int(stmt, 7);
            rec.plan_origin = normalizePlanOrigin(safe_col_text(stmt, 8));
            results.push_back(std::move(rec));
        }

        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

bool SQLiteMemoryRepository::saveStrategy(const CognateStrategyRecord& record) {
    try {
        const char* sql = "INSERT OR REPLACE INTO strategies (strategy_id, description, step_pattern_json, success_rate, created_at, experience_origin) "
                          "VALUES (?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, record.strategy_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, record.description.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, record.step_pattern_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, static_cast<double>(record.success_rate));
        sqlite3_bind_int64(stmt, 5, record.created_at);
        const std::string origin = normalizePlanOrigin(record.experience_origin);
        sqlite3_bind_text(stmt, 6, origin.c_str(), -1, SQLITE_TRANSIENT);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::CognateStrategyRecord> SQLiteMemoryRepository::loadStrategy(const std::string& strategy_id) {
    try {
        const char* sql = "SELECT strategy_id, description, step_pattern_json, success_rate, created_at, experience_origin FROM strategies WHERE strategy_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

        sqlite3_bind_text(stmt, 1, strategy_id.c_str(), -1, SQLITE_STATIC);

        std::optional<CognateStrategyRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateStrategyRecord rec;
            rec.strategy_id = safe_col_text(stmt, 0);
            rec.description = safe_col_text(stmt, 1);
            rec.step_pattern_json = safe_col_text(stmt, 2);
            rec.success_rate = static_cast<float>(sqlite3_column_double(stmt, 3));
            rec.created_at = sqlite3_column_int64(stmt, 4);
            rec.experience_origin = normalizePlanOrigin(safe_col_text(stmt, 5));
            result = rec;
        }

        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::vector<MemoryRepository::CognateStrategyRecord> SQLiteMemoryRepository::getAllStrategies() {
    std::vector<CognateStrategyRecord> results;
    try {
        const char* sql = "SELECT strategy_id, description, step_pattern_json, success_rate, created_at, experience_origin FROM strategies;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateStrategyRecord rec;
            rec.strategy_id = safe_col_text(stmt, 0);
            rec.description = safe_col_text(stmt, 1);
            rec.step_pattern_json = safe_col_text(stmt, 2);
            rec.success_rate = static_cast<float>(sqlite3_column_double(stmt, 3));
            rec.created_at = sqlite3_column_int64(stmt, 4);
            rec.experience_origin = normalizePlanOrigin(safe_col_text(stmt, 5));
            results.push_back(std::move(rec));
        }

        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

bool SQLiteMemoryRepository::archiveMessages(const std::string& sessionId, int count, int summaryVersion) {
    if (count <= 0) return true;

    try {
        if (!beginTransaction()) return false;

        // 1. Get oldest messages to archive
        const char* select_sql = "SELECT role, content, timestamp_ms FROM messages WHERE session_id = ? ORDER BY timestamp_ms ASC LIMIT ?;";
        sqlite3_stmt* select_stmt;
        if (sqlite3_prepare_v2(db_->handle, select_sql, -1, &select_stmt, nullptr) != SQLITE_OK) {
            rollback();
            return false;
        }
        sqlite3_bind_text(select_stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(select_stmt, 2, count);

        struct TempRecord { std::string role; std::string content; int64_t ts; };
        std::vector<TempRecord> to_archive;
        while (sqlite3_step(select_stmt) == SQLITE_ROW) {
            to_archive.push_back({
                safe_col_text(select_stmt, 0),
                safe_col_text(select_stmt, 1),
                sqlite3_column_int64(select_stmt, 2)
            });
        }
        sqlite3_finalize(select_stmt);

        if (to_archive.empty()) {
            commit();
            return true;
        }

        // 2. Insert into archived_turns
        const char* insert_sql = "INSERT INTO archived_turns (archive_id, session_id, original_timestamp_ms, role, content, archived_at_ms, summary_version) "
                                 "VALUES (?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* insert_stmt;
        if (sqlite3_prepare_v2(db_->handle, insert_sql, -1, &insert_stmt, nullptr) != SQLITE_OK) {
            rollback();
            return false;
        }

        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        int idx = 0;
        for (const auto& rec : to_archive) {
            std::string archive_id = sessionId + "-arc-" + std::to_string(rec.ts) + "-" + std::to_string(idx++);
            sqlite3_bind_text(insert_stmt, 1, archive_id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_stmt, 2, sessionId.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(insert_stmt, 3, rec.ts);
            sqlite3_bind_text(insert_stmt, 4, rec.role.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_stmt, 5, rec.content.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(insert_stmt, 6, now);
            sqlite3_bind_int(insert_stmt, 7, summaryVersion);

            if (sqlite3_step(insert_stmt) != SQLITE_DONE) {
                sqlite3_finalize(insert_stmt);
                rollback();
                return false;
            }
            sqlite3_reset(insert_stmt);
        }
        sqlite3_finalize(insert_stmt);

        // 3. Delete from messages
        const char* delete_sql = "DELETE FROM messages WHERE session_id = ? AND timestamp_ms IN (";
        std::string delete_str = delete_sql;
        for (size_t i = 0; i < to_archive.size(); ++i) {
            delete_str += (i == 0 ? "?" : ", ?");
        }
        delete_str += ");";

        sqlite3_stmt* delete_stmt;
        if (sqlite3_prepare_v2(db_->handle, delete_str.c_str(), -1, &delete_stmt, nullptr) != SQLITE_OK) {
            rollback();
            return false;
        }
        sqlite3_bind_text(delete_stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        for (size_t i = 0; i < to_archive.size(); ++i) {
            sqlite3_bind_int64(delete_stmt, static_cast<int>(i + 2), to_archive[i].ts);
        }

        if (sqlite3_step(delete_stmt) != SQLITE_DONE) {
            sqlite3_finalize(delete_stmt);
            rollback();
            return false;
        }
        sqlite3_finalize(delete_stmt);

        return commit();
    } catch (...) {
        rollback();
        return false;
    }
}

std::vector<MemoryRepository::ArchivedTurnRecord> SQLiteMemoryRepository::getArchivedMessages(
    const std::string& sessionId) {
    return getArchivedMessages(sessionId, RestoreRange{});
}

std::vector<MemoryRepository::ArchivedTurnRecord> SQLiteMemoryRepository::getArchivedMessages(
    const std::string& sessionId,
    const RestoreRange& range) {
    std::vector<ArchivedTurnRecord> results;
    if (range.isInvalid()) {
        return results;
    }
    try {
        std::string sql =
            "SELECT archive_id, session_id, original_timestamp_ms, role, content, metadata_json, "
            "archived_at_ms, summary_version FROM archived_turns WHERE session_id = ?";
        if (range.start_ms.has_value()) {
            sql += " AND original_timestamp_ms >= ?";
        }
        if (range.end_ms.has_value()) {
            sql += " AND original_timestamp_ms <= ?";
        }
        sql += " ORDER BY original_timestamp_ms ASC, archive_id ASC;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            return results;
        }

        int bind = 1;
        sqlite3_bind_text(stmt, bind++, sessionId.c_str(), -1, SQLITE_STATIC);
        if (range.start_ms.has_value()) {
            sqlite3_bind_int64(stmt, bind++, *range.start_ms);
        }
        if (range.end_ms.has_value()) {
            sqlite3_bind_int64(stmt, bind++, *range.end_ms);
        }

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            ArchivedTurnRecord rec;
            rec.archive_id = safe_col_text(stmt, 0);
            rec.session_id = safe_col_text(stmt, 1);
            rec.original_timestamp_ms = sqlite3_column_int64(stmt, 2);
            rec.role = safe_col_text(stmt, 3);
            rec.content = safe_col_text(stmt, 4);
            rec.metadata_json = safe_col_text(stmt, 5);
            rec.archived_at_ms = sqlite3_column_int64(stmt, 6);
            rec.summary_version = sqlite3_column_int(stmt, 7);
            results.push_back(std::move(rec));
        }
        sqlite3_finalize(stmt);
    } catch (...) {
    }
    return results;
}

RehydrateBatchResult SQLiteMemoryRepository::rehydrateArchivedMessages(
    const std::string& sessionId,
    const RestoreRange& range) {
    RehydrateBatchResult outcome;
    if (range.isInvalid()) {
        outcome.ok = false;
        return outcome;
    }

    const auto matched = getArchivedMessages(sessionId, range);
    outcome.matched = static_cast<int>(matched.size());
    if (matched.empty()) {
        return outcome;
    }

    try {
        if (!beginTransaction()) {
            outcome.ok = false;
            return outcome;
        }

        // Load hot messages for duplicate checks inside this transaction.
        const char* hot_sql =
            "SELECT role, content, timestamp_ms FROM messages WHERE session_id = ?;";
        sqlite3_stmt* hot_stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, hot_sql, -1, &hot_stmt, nullptr) != SQLITE_OK) {
            rollback();
            outcome.ok = false;
            return outcome;
        }
        sqlite3_bind_text(hot_stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);

        struct HotKey {
            int64_t ts;
            std::string role;
            std::string content;
        };
        std::vector<HotKey> hot_keys;
        while (sqlite3_step(hot_stmt) == SQLITE_ROW) {
            hot_keys.push_back({
                sqlite3_column_int64(hot_stmt, 2),
                safe_col_text(hot_stmt, 0),
                safe_col_text(hot_stmt, 1)
            });
        }
        sqlite3_finalize(hot_stmt);

        auto isDuplicate = [&](const ArchivedTurnRecord& a) {
            for (const auto& h : hot_keys) {
                if (h.ts == a.original_timestamp_ms && h.role == a.role && h.content == a.content) {
                    return true;
                }
            }
            return false;
        };

        const char* insert_sql =
            "INSERT INTO messages (session_id, role, content, timestamp_ms) VALUES (?, ?, ?, ?);";
        sqlite3_stmt* insert_stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, insert_sql, -1, &insert_stmt, nullptr) != SQLITE_OK) {
            rollback();
            outcome.ok = false;
            return outcome;
        }

        for (const auto& rec : matched) {
            if (isDuplicate(rec)) {
                outcome.skipped_dup++;
                continue;
            }

            if (injectRestoreFail("insert")) {
                sqlite3_finalize(insert_stmt);
                rollback();
                outcome.ok = false;
                outcome.restored = 0;
                outcome.skipped_dup = 0;
                return outcome;
            }

            sqlite3_bind_text(insert_stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_stmt, 2, rec.role.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_stmt, 3, rec.content.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(insert_stmt, 4, rec.original_timestamp_ms);

            if (sqlite3_step(insert_stmt) != SQLITE_DONE) {
                sqlite3_finalize(insert_stmt);
                rollback();
                outcome.ok = false;
                outcome.restored = 0;
                outcome.skipped_dup = 0;
                return outcome;
            }
            sqlite3_reset(insert_stmt);

            // Subsequent turns in this batch must see prior inserts as duplicates.
            hot_keys.push_back({rec.original_timestamp_ms, rec.role, rec.content});
            outcome.restored++;
        }
        sqlite3_finalize(insert_stmt);

        if (injectRestoreFail("commit")) {
            rollback();
            outcome.ok = false;
            outcome.restored = 0;
            outcome.skipped_dup = 0;
            return outcome;
        }

        if (!commit()) {
            outcome.ok = false;
            outcome.restored = 0;
            outcome.skipped_dup = 0;
            return outcome;
        }
        return outcome;
    } catch (...) {
        rollback();
        outcome.ok = false;
        outcome.restored = 0;
        outcome.skipped_dup = 0;
        return outcome;
    }
}

int SQLiteMemoryRepository::getHotMessageCount(const std::string& sessionId) {
    int count = 0;
    try {
        const char* sql = "SELECT COUNT(*) FROM messages WHERE session_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return count;
}

std::vector<MessageRecord> SQLiteMemoryRepository::getOldestMessages(const std::string& sessionId, int count) {
    std::vector<MessageRecord> messages;
    if (count <= 0) {
        return messages;
    }
    try {
        const char* sql = "SELECT role, content, timestamp_ms FROM messages WHERE session_id = ? "
                          "ORDER BY timestamp_ms ASC LIMIT ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return messages;
        }
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 2, count);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            messages.push_back({
                safe_col_text(stmt, 0),
                safe_col_text(stmt, 1),
                sqlite3_column_int64(stmt, 2)
            });
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return messages;
}

std::optional<int64_t> SQLiteMemoryRepository::getOldestHotMessageTimestamp(const std::string& sessionId) {
    try {
        const char* sql = "SELECT MIN(timestamp_ms) FROM messages WHERE session_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return std::nullopt;
        }
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        std::optional<int64_t> ts;
        if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
            ts = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
        return ts;
    } catch (...) {
        return std::nullopt;
    }
}

bool SQLiteMemoryRepository::consolidateSessionBatch(const MemoryConsolidationRequest& request) {
    if (request.messages_to_archive.empty()) {
        return true;
    }

    const bool wantsWarm = request.warm.has_value();
    if (wantsWarm) {
        if (request.warm->embedding.empty()) {
            return false;
        }
    }

    try {
        if (!beginTransaction()) {
            return false;
        }

        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::vector<std::string> archiveIds;
        archiveIds.reserve(request.messages_to_archive.size());

        const char* insert_archive_sql =
            "INSERT INTO archived_turns (archive_id, session_id, original_timestamp_ms, role, content, "
            "metadata_json, archived_at_ms, summary_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* insert_archive = nullptr;
        if (sqlite3_prepare_v2(db_->handle, insert_archive_sql, -1, &insert_archive, nullptr) != SQLITE_OK) {
            rollback();
            return false;
        }

        const std::string warmId = wantsWarm ? request.warm->id : "";
        const std::string digest = wantsWarm ? request.warm->derived_from_hash : "";

        int idx = 0;
        for (const auto& rec : request.messages_to_archive) {
            if (injectConsolidationFail("archive")) {
                sqlite3_finalize(insert_archive);
                rollback();
                return false;
            }

            const std::string archive_id =
                request.session_id + "-arc-" + std::to_string(rec.timestamp_ms) + "-" + std::to_string(idx++);
            archiveIds.push_back(archive_id);

            nlohmann::json metadata;
            metadata["warm_memory_id"] = warmId;
            metadata["derived_from_hash"] = digest;
            metadata["turn_index"] = idx - 1;

            // Keep the dumped JSON alive through sqlite3_step().
            // SQLITE_STATIC requires the caller to own the buffer for the duration
            // of statement execution.
            const std::string metadataJson = metadata.dump();
            sqlite3_bind_text(insert_archive, 1, archive_id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_archive, 2, request.session_id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(insert_archive, 3, rec.timestamp_ms);
            sqlite3_bind_text(insert_archive, 4, rec.role.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_archive, 5, rec.content.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_archive, 6, metadataJson.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int64(insert_archive, 7, now);
            sqlite3_bind_int(insert_archive, 8, wantsWarm ? request.warm->summary_version : 0);

            if (sqlite3_step(insert_archive) != SQLITE_DONE) {
                sqlite3_finalize(insert_archive);
                rollback();
                return false;
            }
            sqlite3_reset(insert_archive);
        }
        sqlite3_finalize(insert_archive);

        if (wantsWarm) {
            const auto& warm = *request.warm;
            nlohmann::json parentIds = nlohmann::json::array();
            for (const auto& id : archiveIds) {
                parentIds.push_back(id);
            }

            if (injectConsolidationFail("warm")) {
                rollback();
                return false;
            }

            const char* insert_warm_sql =
                "INSERT INTO warm_memory (id, session_id, scope, episodic_payload, rendered_summary, "
                "importance, novelty, confidence, covered_turn_start, covered_turn_end, covered_ts_start, "
                "covered_ts_end, parent_archive_ids, derived_from_hash, summary_version, prompt_version, "
                "llm_model, summary_missing, created_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
            sqlite3_stmt* insert_warm = nullptr;
            if (sqlite3_prepare_v2(db_->handle, insert_warm_sql, -1, &insert_warm, nullptr) != SQLITE_OK) {
                rollback();
                return false;
            }

            const std::string parentJson = parentIds.dump();
            sqlite3_bind_text(insert_warm, 1, warm.id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_warm, 2, warm.session_id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(insert_warm, 3, static_cast<int>(warm.scope));
            sqlite3_bind_text(insert_warm, 4, warm.episodic_payload.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_warm, 5, warm.rendered_summary.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_double(insert_warm, 6, warm.importance);
            sqlite3_bind_double(insert_warm, 7, warm.novelty);
            sqlite3_bind_double(insert_warm, 8, warm.confidence);
            sqlite3_bind_int(insert_warm, 9, warm.covered_turn_start);
            sqlite3_bind_int(insert_warm, 10, warm.covered_turn_end);
            sqlite3_bind_int64(insert_warm, 11, warm.covered_ts_start);
            sqlite3_bind_int64(insert_warm, 12, warm.covered_ts_end);
            sqlite3_bind_text(insert_warm, 13, parentJson.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_warm, 14, warm.derived_from_hash.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(insert_warm, 15, warm.summary_version);
            sqlite3_bind_text(insert_warm, 16, warm.prompt_version.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(insert_warm, 17, warm.llm_model.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_int(insert_warm, 18, warm.summary_missing ? 1 : 0);
            sqlite3_bind_int64(insert_warm, 19, warm.created_at_ms > 0 ? warm.created_at_ms : now);

            if (sqlite3_step(insert_warm) != SQLITE_DONE) {
                sqlite3_finalize(insert_warm);
                rollback();
                return false;
            }
            sqlite3_finalize(insert_warm);

            const char* insert_embed_sql =
                "INSERT INTO warm_memory_embeddings (memory_id, embedding, embedding_version) VALUES (?, ?, ?);";
            sqlite3_stmt* insert_embed = nullptr;
            if (sqlite3_prepare_v2(db_->handle, insert_embed_sql, -1, &insert_embed, nullptr) != SQLITE_OK) {
                rollback();
                return false;
            }
            sqlite3_bind_text(insert_embed, 1, warm.id.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_blob(insert_embed, 2, warm.embedding.data(),
                              static_cast<int>(warm.embedding.size() * sizeof(float)), SQLITE_STATIC);
            sqlite3_bind_int(insert_embed, 3, warm.embedding_version);
            if (sqlite3_step(insert_embed) != SQLITE_DONE) {
                sqlite3_finalize(insert_embed);
                rollback();
                return false;
            }
            sqlite3_finalize(insert_embed);
        }

        std::string delete_sql = "DELETE FROM messages WHERE session_id = ? AND timestamp_ms IN (";
        for (size_t i = 0; i < request.messages_to_archive.size(); ++i) {
            delete_sql += (i == 0 ? "?" : ", ?");
        }
        delete_sql += ");";

        sqlite3_stmt* delete_stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, delete_sql.c_str(), -1, &delete_stmt, nullptr) != SQLITE_OK) {
            rollback();
            return false;
        }
        sqlite3_bind_text(delete_stmt, 1, request.session_id.c_str(), -1, SQLITE_STATIC);
        for (size_t i = 0; i < request.messages_to_archive.size(); ++i) {
            sqlite3_bind_int64(delete_stmt, static_cast<int>(i + 2), request.messages_to_archive[i].timestamp_ms);
        }
        if (sqlite3_step(delete_stmt) != SQLITE_DONE) {
            sqlite3_finalize(delete_stmt);
            rollback();
            return false;
        }
        sqlite3_finalize(delete_stmt);

        if (injectConsolidationFail("commit")) {
            rollback();
            return false;
        }

        return commit();
    } catch (...) {
        rollback();
        return false;
    }
}

namespace {

static MemoryRepository::WarmMemoryRecord loadWarmRow(sqlite3_stmt* stmt, sqlite3* db, int embeddingVersion) {
    MemoryRepository::WarmMemoryRecord rec;
    rec.id = safe_col_text(stmt, 0);
    rec.session_id = safe_col_text(stmt, 1);
    rec.scope = static_cast<MemoryScope>(sqlite3_column_int(stmt, 2));
    rec.episodic_payload = safe_col_text(stmt, 3);
    rec.rendered_summary = safe_col_text(stmt, 4);
    rec.importance = static_cast<float>(sqlite3_column_double(stmt, 5));
    rec.novelty = static_cast<float>(sqlite3_column_double(stmt, 6));
    rec.confidence = static_cast<float>(sqlite3_column_double(stmt, 7));
    rec.covered_turn_start = sqlite3_column_int(stmt, 8);
    rec.covered_turn_end = sqlite3_column_int(stmt, 9);
    rec.covered_ts_start = sqlite3_column_int64(stmt, 10);
    rec.covered_ts_end = sqlite3_column_int64(stmt, 11);
    rec.parent_archive_ids_json = safe_col_text(stmt, 12);
    rec.derived_from_hash = safe_col_text(stmt, 13);
    rec.summary_version = sqlite3_column_int(stmt, 14);
    rec.prompt_version = safe_col_text(stmt, 15);
    rec.llm_model = safe_col_text(stmt, 16);
    rec.summary_missing = sqlite3_column_int(stmt, 17) != 0;
    rec.created_at_ms = sqlite3_column_int64(stmt, 18);
    rec.embedding_version = embeddingVersion;

    const char* embed_sql =
        "SELECT embedding FROM warm_memory_embeddings WHERE memory_id = ? AND embedding_version = ?;";
    sqlite3_stmt* embed_stmt = nullptr;
    if (sqlite3_prepare_v2(db, embed_sql, -1, &embed_stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(embed_stmt, 1, rec.id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(embed_stmt, 2, embeddingVersion);
        if (sqlite3_step(embed_stmt) == SQLITE_ROW) {
            const void* blob = sqlite3_column_blob(embed_stmt, 0);
            const int bytes = sqlite3_column_bytes(embed_stmt, 0);
            const int count = bytes / static_cast<int>(sizeof(float));
            rec.embedding.resize(static_cast<size_t>(count));
            if (blob && bytes > 0) {
                std::memcpy(rec.embedding.data(), blob, static_cast<size_t>(bytes));
            }
        }
        sqlite3_finalize(embed_stmt);
    }
    return rec;
}

} // namespace

std::vector<SQLiteMemoryRepository::WarmMemoryRecord> SQLiteMemoryRepository::getRecentWarmMemory(const std::string& sessionId, int limit) {
    std::vector<WarmMemoryRecord> results;
    if (limit <= 0) {
        return results;
    }
    try {
        const char* sql =
            "SELECT id, session_id, scope, episodic_payload, rendered_summary, importance, novelty, confidence, "
            "covered_turn_start, covered_turn_end, covered_ts_start, covered_ts_end, parent_archive_ids, "
            "derived_from_hash, summary_version, prompt_version, llm_model, summary_missing, created_at_ms "
            "FROM warm_memory WHERE session_id = ? ORDER BY created_at_ms DESC LIMIT ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return results;
        }
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 2, limit);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            results.push_back(loadWarmRow(stmt, db_->handle, Thoth::MemoryConsolidation::kWarmMemoryEmbeddingVersion));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

std::vector<SQLiteMemoryRepository::WarmMemoryRecord> SQLiteMemoryRepository::getAllRecentWarmMemory(int limit) {
    std::vector<WarmMemoryRecord> results;
    if (limit <= 0) {
        return results;
    }
    try {
        const char* sql =
            "SELECT id, session_id, scope, episodic_payload, rendered_summary, importance, novelty, confidence, "
            "covered_turn_start, covered_turn_end, covered_ts_start, covered_ts_end, parent_archive_ids, "
            "derived_from_hash, summary_version, prompt_version, llm_model, summary_missing, created_at_ms "
            "FROM warm_memory ORDER BY created_at_ms DESC LIMIT ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return results;
        }
        sqlite3_bind_int(stmt, 1, limit);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            results.push_back(loadWarmRow(stmt, db_->handle, Thoth::MemoryConsolidation::kWarmMemoryEmbeddingVersion));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

std::vector<SQLiteMemoryRepository::WarmMemoryRecord> SQLiteMemoryRepository::searchWarmMemoryByEmbedding(
    const std::string& sessionId,
    MemoryScope scope,
    const std::vector<float>& queryEmbedding,
    int limit,
    int embeddingVersion) {
    if (queryEmbedding.empty() || limit <= 0) {
        return {};
    }

    auto rows = sessionId.empty() ? getAllRecentWarmMemory(100) : getRecentWarmMemory(sessionId, 100);
    std::vector<std::pair<WarmMemoryRecord, float>> ranked;
    ranked.reserve(rows.size());
    for (auto& row : rows) {
        if (row.scope != scope || row.embedding.empty()) {
            continue;
        }
        const float score = GragScorer::cosine_similarity(queryEmbedding, row.embedding);
        ranked.push_back({std::move(row), score * (0.5f + 0.5f * row.importance)});
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<WarmMemoryRecord> results;
    const int count = std::min(limit, static_cast<int>(ranked.size()));
    results.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        results.push_back(std::move(ranked[static_cast<size_t>(i)].first));
    }
    return results;
}

bool SQLiteMemoryRepository::upsertFact(const FactRecord& fact) {
    try {
        const char* sql = "INSERT OR REPLACE INTO facts (key, value, confidence, source, last_updated_ms) VALUES (?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, fact.key.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, fact.value.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 3, static_cast<double>(fact.confidence));
        sqlite3_bind_text(stmt, 4, fact.source.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 5, fact.last_updated_ms);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::FactRecord> SQLiteMemoryRepository::getFact(const std::string& key) {
    try {
        const char* sql = "SELECT key, value, confidence, source, last_updated_ms FROM facts WHERE key = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_STATIC);
        
        std::optional<FactRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            result = FactRecord{
                safe_col_text(stmt, 0),
                safe_col_text(stmt, 1),
                static_cast<float>(sqlite3_column_double(stmt, 2)),
                safe_col_text(stmt, 3),
                sqlite3_column_int64(stmt, 4)
            };
        }
        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::vector<MemoryRepository::FactRecord> SQLiteMemoryRepository::searchFacts(const std::string& query) {
    std::vector<FactRecord> results;
    try {
        // Case-insensitive substring match on key, value, OR source
        const char* sql = "SELECT key, value, confidence, source, last_updated_ms FROM facts "
                          "WHERE key LIKE ? OR value LIKE ? OR source LIKE ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;
        
        std::string like_query = "%" + query + "%";
        sqlite3_bind_text(stmt, 1, like_query.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, like_query.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, like_query.c_str(), -1, SQLITE_STATIC);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            results.push_back({
                safe_col_text(stmt, 0),
                safe_col_text(stmt, 1),
                static_cast<float>(sqlite3_column_double(stmt, 2)),
                safe_col_text(stmt, 3),
                sqlite3_column_int64(stmt, 4)
            });
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

bool SQLiteMemoryRepository::deleteFact(const std::string& key) {
    try {
        const char* sql = "DELETE FROM facts WHERE key = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_STATIC);
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

void SQLiteMemoryRepository::configureEpisodeIdentity() {
    if (!db_ || !db_->handle) return;
    const EpisodeIdentityReport report = inspectEpisodeIdentity();
    if (report.identical_duplicate_groups == 0) {
        char* err = nullptr;
        const int rc = sqlite3_exec(
            db_->handle,
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_episode_step_observation "
            "ON episode_steps(goal_id, step_index, result_status, state_summary, action_taken);",
            nullptr, nullptr, &err);
        if (rc != SQLITE_OK) {
            std::cerr << "[SQLiteMemoryRepository] Episode observation index was not created: "
                      << (err ? err : "unknown") << "\n";
            sqlite3_free(err);
        }
    } else {
        std::cerr << "[SQLiteMemoryRepository] Preserving " << report.identical_duplicate_groups
                  << " identical duplicate episode groups and "
                  << report.conflicting_goal_step_groups
                  << " conflicting (goal_id, step_index) groups. No rows were deleted.\n";
    }
    observation_unique_ = false;
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(
            db_->handle,
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_episode_step_observation';",
            -1, &stmt, nullptr) == SQLITE_OK) {
        observation_unique_ = sqlite3_step(stmt) == SQLITE_ROW;
        sqlite3_finalize(stmt);
    }
}

bool SQLiteMemoryRepository::episodeObservationExists(const EpisodeStepRecord& step) {
    const char* sql = "SELECT 1 FROM episode_steps WHERE goal_id = ? AND step_index = ? "
                      "AND result_status = ? AND IFNULL(state_summary, '') = ? "
                      "AND IFNULL(action_taken, '') = ? LIMIT 1;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_text(stmt, 1, step.goal_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, step.step_index);
    sqlite3_bind_text(stmt, 3, step.result_status.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, step.state_summary.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, step.action_taken.c_str(), -1, SQLITE_STATIC);
    const bool found = sqlite3_step(stmt) == SQLITE_ROW;
    sqlite3_finalize(stmt);
    return found;
}

MemoryRepository::EpisodeIdentityReport SQLiteMemoryRepository::inspectEpisodeIdentity() {
    EpisodeIdentityReport report;
    if (!db_ || !db_->handle) return report;
    sqlite3_stmt* count = nullptr;
    if (sqlite3_prepare_v2(db_->handle, "SELECT COUNT(*) FROM episode_steps;", -1, &count, nullptr) == SQLITE_OK &&
        sqlite3_step(count) == SQLITE_ROW) {
        report.row_count = sqlite3_column_int(count, 0);
    }
    if (count) sqlite3_finalize(count);

    const char* groups =
        "SELECT COUNT(*) AS n, "
        "COUNT(DISTINCT result_status || X'1F' || IFNULL(state_summary, '') || X'1F' || IFNULL(action_taken, '')) AS variants "
        "FROM episode_steps GROUP BY goal_id, step_index HAVING n > 1;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_->handle, groups, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            report.duplicate_goal_step_groups += 1;
            if (sqlite3_column_int(stmt, 1) > 1) report.conflicting_goal_step_groups += 1;
            else report.identical_duplicate_groups += 1;
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_stmt* index = nullptr;
    if (sqlite3_prepare_v2(
            db_->handle,
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_episode_step_observation';",
            -1, &index, nullptr) == SQLITE_OK) {
        report.observation_unique_index = sqlite3_step(index) == SQLITE_ROW;
        sqlite3_finalize(index);
    }
    return report;
}

bool SQLiteMemoryRepository::commitStepCheckpoint(
    const ActivePlanRecord& plan,
    const std::optional<CognatePlanRecord>& cognate,
    const std::optional<EpisodeStepRecord>& episode,
    const std::optional<Cm1SnapshotRecord>& cm1_snapshot,
    const std::optional<Cm1AnswerVersionRecord>& cm1_answer,
    const std::optional<Cm1BreadthEstimate>& cm1_breadth) {
    if (!beginTransaction()) return false;
    if (!storeActivePlan(plan)) {
        rollback();
        return false;
    }
    if (cognate && !saveCognatePlan(*cognate)) {
        rollback();
        return false;
    }
    if (episode && !storeEpisodeStep(*episode)) {
        rollback();
        return false;
    }
    if (const char* env = std::getenv("THOTH_INJECT_EPISODE_CHECKPOINT_FAIL")) {
        if (std::string(env) == "1") {
            rollback();
            return false;
        }
    }
    // CM-1 is observational. A telemetry insert failure must not discard the
    // plan/episode checkpoint. SAVEPOINT isolates that failure.
    if (cm1_snapshot && !cm1_snapshot->snapshot_id.empty()) {
        if (sqlite3_exec(db_->handle, "SAVEPOINT cm1_snapshot;", nullptr, nullptr, nullptr)
            == SQLITE_OK) {
            if (!insertCm1Snapshot(*cm1_snapshot)) {
                sqlite3_exec(db_->handle, "ROLLBACK TO cm1_snapshot;", nullptr, nullptr, nullptr);
                std::cerr << "[SQLiteMemoryRepository] CM-1 snapshot insert failed; "
                             "step checkpoint preserved without telemetry\n";
            }
            sqlite3_exec(db_->handle, "RELEASE cm1_snapshot;", nullptr, nullptr, nullptr);
        }
    }
    if (cm1_answer && !cm1_answer->answer_id.empty()) {
        if (sqlite3_exec(db_->handle, "SAVEPOINT cm1_answer;", nullptr, nullptr, nullptr)
            == SQLITE_OK) {
            if (!insertCm1AnswerVersion(*cm1_answer)) {
                sqlite3_exec(db_->handle, "ROLLBACK TO cm1_answer;", nullptr, nullptr, nullptr);
                std::cerr << "[SQLiteMemoryRepository] CM-1 answer-version insert failed; "
                             "step checkpoint preserved without answer provenance\n";
            }
            sqlite3_exec(db_->handle, "RELEASE cm1_answer;", nullptr, nullptr, nullptr);
        }
    }
    if (cm1_breadth && !cm1_breadth->estimate_id.empty()) {
        if (sqlite3_exec(db_->handle, "SAVEPOINT cm1_breadth;", nullptr, nullptr, nullptr)
            == SQLITE_OK) {
            if (!insertCm1BreadthEstimate(*cm1_breadth)) {
                sqlite3_exec(db_->handle, "ROLLBACK TO cm1_breadth;", nullptr, nullptr, nullptr);
                std::cerr << "[SQLiteMemoryRepository] CM-1 breadth insert failed; "
                             "step checkpoint and raw snapshot preserved\n";
            }
            sqlite3_exec(db_->handle, "RELEASE cm1_breadth;", nullptr, nullptr, nullptr);
        }
    }
    if (const char* inj = std::getenv("THOTH_INJECT_CM1_CHECKPOINT_FAIL")) {
        if (std::string(inj) == "1" && cm1_snapshot) {
            // Test hook: force the telemetry savepoint path without failing the step.
            sqlite3_exec(db_->handle, "SAVEPOINT cm1_inject;", nullptr, nullptr, nullptr);
            sqlite3_exec(db_->handle, "ROLLBACK TO cm1_inject;", nullptr, nullptr, nullptr);
            sqlite3_exec(db_->handle, "RELEASE cm1_inject;", nullptr, nullptr, nullptr);
        }
    }
    if (!commit()) return false;
    if (!(db_ && db_->handle && sqlite3_get_autocommit(db_->handle) != 0)) return true;

    ActivePlanCommitObserver observer;
    {
        std::lock_guard<std::mutex> lock(active_plan_observer_mu_);
        observer = active_plan_observer_;
    }
    if (!observer) return true;
    try {
        observer(plan);
    } catch (const std::exception& ex) {
        std::cerr << "[SQLiteMemoryRepository] active-plan commit observer failed: "
                  << ex.what() << "\n";
    } catch (...) {
        std::cerr << "[SQLiteMemoryRepository] active-plan commit observer failed\n";
    }
    return true;
}

bool SQLiteMemoryRepository::insertCm1Snapshot(const Cm1SnapshotRecord& record) {
    try {
        if (!db_ || !db_->handle || record.snapshot_id.empty()) return false;
        if (const char* inj = std::getenv("THOTH_INJECT_CM1_WRITE_FAIL")) {
            if (std::string(inj) == "1") return false;
        }
        const char* sql =
            "INSERT OR IGNORE INTO cm1_confidence_snapshots "
            "(snapshot_id, plan_id, step_id, snapshot_type, lifecycle_stage, features_json, "
            "measurement_status, schema_version, estimator_status, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, record.snapshot_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.plan_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.step_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.snapshot_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, record.lifecycle_stage.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, record.features_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 7, record.measurement_status.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 8, record.schema_version.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 9, record.estimator_status.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 10, record.created_at_ms);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

std::optional<Cm1SnapshotRecord> SQLiteMemoryRepository::loadCm1Snapshot(
    const std::string& snapshot_id) {
    try {
        if (!db_ || !db_->handle || snapshot_id.empty()) return std::nullopt;
        const char* sql =
            "SELECT snapshot_id, plan_id, step_id, snapshot_type, lifecycle_stage, features_json, "
            "measurement_status, schema_version, estimator_status, created_at_ms "
            "FROM cm1_confidence_snapshots WHERE snapshot_id = ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        sqlite3_bind_text(stmt, 1, snapshot_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1SnapshotRecord> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            Cm1SnapshotRecord record;
            record.snapshot_id = safe_col_text(stmt, 0);
            record.plan_id = safe_col_text(stmt, 1);
            record.step_id = safe_col_text(stmt, 2);
            record.snapshot_type = safe_col_text(stmt, 3);
            record.lifecycle_stage = safe_col_text(stmt, 4);
            record.features_json = safe_col_text(stmt, 5);
            record.measurement_status = safe_col_text(stmt, 6);
            record.schema_version = safe_col_text(stmt, 7);
            record.estimator_status = safe_col_text(stmt, 8);
            record.created_at_ms = sqlite3_column_int64(stmt, 9);
            out = std::move(record);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<Cm1SnapshotRecord> SQLiteMemoryRepository::listCm1SnapshotsForPlan(
    const std::string& plan_id) {
    std::vector<Cm1SnapshotRecord> rows;
    try {
        if (!db_ || !db_->handle || plan_id.empty()) return rows;
        const char* sql =
            "SELECT snapshot_id, plan_id, step_id, snapshot_type, lifecycle_stage, features_json, "
            "measurement_status, schema_version, estimator_status, created_at_ms "
            "FROM cm1_confidence_snapshots WHERE plan_id = ? ORDER BY created_at_ms, snapshot_id;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return rows;
        sqlite3_bind_text(stmt, 1, plan_id.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            Cm1SnapshotRecord record;
            record.snapshot_id = safe_col_text(stmt, 0);
            record.plan_id = safe_col_text(stmt, 1);
            record.step_id = safe_col_text(stmt, 2);
            record.snapshot_type = safe_col_text(stmt, 3);
            record.lifecycle_stage = safe_col_text(stmt, 4);
            record.features_json = safe_col_text(stmt, 5);
            record.measurement_status = safe_col_text(stmt, 6);
            record.schema_version = safe_col_text(stmt, 7);
            record.estimator_status = safe_col_text(stmt, 8);
            record.created_at_ms = sqlite3_column_int64(stmt, 9);
            rows.push_back(std::move(record));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return rows;
}

bool SQLiteMemoryRepository::insertCm1AnswerVersion(const Cm1AnswerVersionRecord& record) {
    try {
        if (!db_ || !db_->handle || record.answer_id.empty()) return false;
        if (record.generation_id.empty() || record.content_sha256.empty()) return false;
        if (const char* inj = std::getenv("THOTH_INJECT_CM1_WRITE_FAIL")) {
            if (std::string(inj) == "1") return false;
        }
        const char* sql =
            "INSERT OR IGNORE INTO cm1_answer_versions "
            "(answer_id, plan_id, step_id, generation_id, content_sha256, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, record.answer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.plan_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.step_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.generation_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, record.content_sha256.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 6, record.created_at_ms);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

bool SQLiteMemoryRepository::insertCm1BreadthEstimate(const Cm1BreadthEstimate& record) {
    try {
        if (!db_ || !db_->handle || record.estimate_id.empty()) return false;
        if (const char* inj = std::getenv("THOTH_INJECT_CM1_WRITE_FAIL")) {
            if (std::string(inj) == "1") return false;
        }
        const char* sql =
            "INSERT OR IGNORE INTO cm1_breadth_estimates "
            "(estimate_id, snapshot_id, plan_id, step_id, estimator_version, payload_json, "
            "created_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, record.estimate_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.snapshot_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.plan_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.step_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, record.estimator_version.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, record.payload_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 7, record.created_at_ms);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

bool SQLiteMemoryRepository::insertCm1EvidenceJudgment(const Cm1EvidenceJudgmentRecord& record) {
    try {
        if (!db_ || !db_->handle || record.judgment_id.empty()) return false;
        if (record.answer_id.empty() || record.content_sha256.empty()) return false;
        if (record.estimator_version.empty() || record.payload_json.empty()) return false;
        if (const char* inj = std::getenv("THOTH_INJECT_CM1_WRITE_FAIL")) {
            if (std::string(inj) == "1") return false;
        }
        const char* sql =
            "INSERT OR IGNORE INTO cm1_evidence_judgments "
            "(judgment_id, answer_id, content_sha256, estimator_version, payload_json, "
            "created_at_ms) VALUES (?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, record.judgment_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.answer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.content_sha256.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.estimator_version.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, record.payload_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 6, record.created_at_ms);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

std::optional<Cm1EvidenceJudgmentRecord> SQLiteMemoryRepository::loadCm1EvidenceJudgment(
    const std::string& judgment_id) {
    try {
        if (!db_ || !db_->handle || judgment_id.empty()) return std::nullopt;
        const char* sql =
            "SELECT judgment_id, answer_id, content_sha256, estimator_version, payload_json, "
            "created_at_ms FROM cm1_evidence_judgments WHERE judgment_id = ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return std::nullopt;
        }
        sqlite3_bind_text(stmt, 1, judgment_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1EvidenceJudgmentRecord> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            Cm1EvidenceJudgmentRecord record;
            record.judgment_id = safe_col_text(stmt, 0);
            record.answer_id = safe_col_text(stmt, 1);
            record.content_sha256 = safe_col_text(stmt, 2);
            record.estimator_version = safe_col_text(stmt, 3);
            record.payload_json = safe_col_text(stmt, 4);
            record.created_at_ms = sqlite3_column_int64(stmt, 5);
            out = std::move(record);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<Cm1BreadthEstimate> SQLiteMemoryRepository::loadCm1BreadthEstimate(
    const std::string& estimate_id) {
    try {
        if (!db_ || !db_->handle || estimate_id.empty()) return std::nullopt;
        const char* sql =
            "SELECT estimate_id, snapshot_id, plan_id, step_id, estimator_version, payload_json, "
            "created_at_ms FROM cm1_breadth_estimates WHERE estimate_id = ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return std::nullopt;
        }
        sqlite3_bind_text(stmt, 1, estimate_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1BreadthEstimate> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            Cm1BreadthEstimate record;
            record.estimate_id = safe_col_text(stmt, 0);
            record.snapshot_id = safe_col_text(stmt, 1);
            record.plan_id = safe_col_text(stmt, 2);
            record.step_id = safe_col_text(stmt, 3);
            record.estimator_version = safe_col_text(stmt, 4);
            record.payload_json = safe_col_text(stmt, 5);
            record.created_at_ms = sqlite3_column_int64(stmt, 6);
            out = std::move(record);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

namespace {

Cm1UserFeedback feedbackFromRow(sqlite3_stmt* stmt) {
    Cm1UserFeedback record;
    record.feedback_id = safe_col_text(stmt, 0);
    record.answer_id = safe_col_text(stmt, 1);
    record.content_sha256 = safe_col_text(stmt, 2);
    record.thumbs_status = safe_col_text(stmt, 3);
    record.quality_rating_status = safe_col_text(stmt, 4);
    if (sqlite3_column_type(stmt, 5) != SQLITE_NULL) {
        record.quality_rating = sqlite3_column_int(stmt, 5);
    }
    record.explanation = safe_col_text(stmt, 6);
    record.supersedes_feedback_id = safe_col_text(stmt, 7);
    record.revision_index = sqlite3_column_int(stmt, 8);
    record.created_at_ms = sqlite3_column_int64(stmt, 9);
    return record;
}

Cm1VerifiedLabel labelFromRow(sqlite3_stmt* stmt) {
    Cm1VerifiedLabel record;
    record.label_id = safe_col_text(stmt, 0);
    record.answer_id = safe_col_text(stmt, 1);
    record.content_sha256 = safe_col_text(stmt, 2);
    record.protocol_id = safe_col_text(stmt, 3);
    record.reviewer_id = safe_col_text(stmt, 4);
    record.factual_bin = sqlite3_column_int(stmt, 5);
    record.supersedes_label_id = safe_col_text(stmt, 6);
    record.revision_index = sqlite3_column_int(stmt, 7);
    record.created_at_ms = sqlite3_column_int64(stmt, 8);
    return record;
}

Cm1ClaimVerification claimVerificationFromRow(sqlite3_stmt* stmt) {
    Cm1ClaimVerification record;
    record.verification_id = safe_col_text(stmt, 0);
    record.claim_id = safe_col_text(stmt, 1);
    record.claim_sha256 = safe_col_text(stmt, 2);
    record.answer_id = safe_col_text(stmt, 3);
    record.content_sha256 = safe_col_text(stmt, 4);
    record.protocol_id = safe_col_text(stmt, 5);
    record.protocol_version = safe_col_text(stmt, 6);
    record.reviewer_id = safe_col_text(stmt, 7);
    record.verifier_kind = safe_col_text(stmt, 8);
    record.verifier_version = safe_col_text(stmt, 9);
    record.method_id = safe_col_text(stmt, 10);
    record.outcome = safe_col_text(stmt, 11);
    record.source_kind = safe_col_text(stmt, 12);
    record.source_ref = safe_col_text(stmt, 13);
    record.calibration_eligibility = safe_col_text(stmt, 14);
    record.egar_eligibility = safe_col_text(stmt, 15);
    record.revision_reason = safe_col_text(stmt, 16);
    record.supersedes_verification_id = safe_col_text(stmt, 17);
    record.revision_index = sqlite3_column_int(stmt, 18);
    record.created_at_ms = sqlite3_column_int64(stmt, 19);
    record.evidence_label = safe_col_text(stmt, 20);
    return record;
}

} // namespace

bool SQLiteMemoryRepository::insertCm1UserFeedback(const Cm1UserFeedback& record) {
    try {
        if (!db_ || !db_->handle || record.feedback_id.empty() || record.answer_id.empty()) {
            return false;
        }
        const char* sql =
            "INSERT OR IGNORE INTO cm1_answer_feedback "
            "(feedback_id, answer_id, content_sha256, thumbs_status, quality_rating_status, "
            "quality_rating, explanation, supersedes_feedback_id, revision_index, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, record.feedback_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.answer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.content_sha256.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.thumbs_status.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, record.quality_rating_status.c_str(), -1, SQLITE_TRANSIENT);
        if (record.quality_rating_status == "present") {
            sqlite3_bind_int(stmt, 6, record.quality_rating);
        } else {
            sqlite3_bind_null(stmt, 6);
        }
        sqlite3_bind_text(stmt, 7, record.explanation.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 8, record.supersedes_feedback_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 9, record.revision_index);
        sqlite3_bind_int64(stmt, 10, record.created_at_ms);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

std::optional<Cm1UserFeedback> SQLiteMemoryRepository::latestCm1UserFeedback(
    const std::string& answer_id) {
    try {
        if (!db_ || !db_->handle || answer_id.empty()) return std::nullopt;
        const char* sql =
            "SELECT feedback_id, answer_id, content_sha256, thumbs_status, quality_rating_status, "
            "quality_rating, explanation, supersedes_feedback_id, revision_index, created_at_ms "
            "FROM cm1_answer_feedback WHERE answer_id = ? "
            "ORDER BY revision_index DESC LIMIT 1;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1UserFeedback> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            out = feedbackFromRow(stmt);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<Cm1UserFeedback> SQLiteMemoryRepository::listCm1UserFeedback(
    const std::string& answer_id) {
    std::vector<Cm1UserFeedback> rows;
    try {
        if (!db_ || !db_->handle || answer_id.empty()) return rows;
        const char* sql =
            "SELECT feedback_id, answer_id, content_sha256, thumbs_status, quality_rating_status, "
            "quality_rating, explanation, supersedes_feedback_id, revision_index, created_at_ms "
            "FROM cm1_answer_feedback WHERE answer_id = ? ORDER BY revision_index;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return rows;
        sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            rows.push_back(feedbackFromRow(stmt));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return rows;
}

bool SQLiteMemoryRepository::insertCm1VerifiedLabel(const Cm1VerifiedLabel& record) {
    try {
        if (!db_ || !db_->handle || record.label_id.empty() || record.answer_id.empty()) {
            return false;
        }
        const char* sql =
            "INSERT OR IGNORE INTO cm1_verified_labels "
            "(label_id, answer_id, content_sha256, protocol_id, reviewer_id, factual_bin, "
            "supersedes_label_id, revision_index, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, record.label_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, record.answer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, record.content_sha256.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, record.protocol_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, record.reviewer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 6, record.factual_bin);
        sqlite3_bind_text(stmt, 7, record.supersedes_label_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 8, record.revision_index);
        sqlite3_bind_int64(stmt, 9, record.created_at_ms);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

std::optional<Cm1VerifiedLabel> SQLiteMemoryRepository::latestCm1VerifiedLabel(
    const std::string& answer_id, const std::string& protocol_id) {
    try {
        if (!db_ || !db_->handle || answer_id.empty() || protocol_id.empty()) return std::nullopt;
        const char* sql =
            "SELECT label_id, answer_id, content_sha256, protocol_id, reviewer_id, factual_bin, "
            "supersedes_label_id, revision_index, created_at_ms "
            "FROM cm1_verified_labels WHERE answer_id = ? AND protocol_id = ? "
            "ORDER BY revision_index DESC LIMIT 1;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, protocol_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1VerifiedLabel> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            out = labelFromRow(stmt);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<Cm1VerifiedLabel> SQLiteMemoryRepository::listCm1VerifiedLabels(
    const std::string& answer_id) {
    std::vector<Cm1VerifiedLabel> rows;
    try {
        if (!db_ || !db_->handle || answer_id.empty()) return rows;
        const char* sql =
            "SELECT label_id, answer_id, content_sha256, protocol_id, reviewer_id, factual_bin, "
            "supersedes_label_id, revision_index, created_at_ms "
            "FROM cm1_verified_labels WHERE answer_id = ? ORDER BY revision_index;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return rows;
        sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            rows.push_back(labelFromRow(stmt));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return rows;
}

namespace {

const char* kClaimVerificationColumns =
    "verification_id, claim_id, claim_sha256, answer_id, content_sha256, "
    "protocol_id, protocol_version, reviewer_id, verifier_kind, verifier_version, method_id, "
    "outcome, source_kind, source_ref, calibration_eligibility, egar_eligibility, "
    "revision_reason, supersedes_verification_id, revision_index, created_at_ms, "
    "evidence_label";

bool sameStoredVerification(const Cm1ClaimVerification& a, const Cm1ClaimVerification& b) {
    return a.verification_id == b.verification_id
        && a.claim_id == b.claim_id
        && a.claim_sha256 == b.claim_sha256
        && a.answer_id == b.answer_id
        && a.content_sha256 == b.content_sha256
        && a.protocol_id == b.protocol_id
        && a.protocol_version == b.protocol_version
        && a.reviewer_id == b.reviewer_id
        && a.verifier_kind == b.verifier_kind
        && a.verifier_version == b.verifier_version
        && a.method_id == b.method_id
        && a.outcome == b.outcome
        && a.evidence_label == b.evidence_label
        && a.source_kind == b.source_kind
        && a.source_ref == b.source_ref
        && a.calibration_eligibility == b.calibration_eligibility
        && a.egar_eligibility == b.egar_eligibility
        && a.revision_reason == b.revision_reason
        && a.supersedes_verification_id == b.supersedes_verification_id
        && a.revision_index == b.revision_index
        && a.created_at_ms == b.created_at_ms;
}

std::optional<Cm1ClaimVerification> loadClaimVerificationById(sqlite3* db, const std::string& id) {
    const std::string sql = std::string("SELECT ") + kClaimVerificationColumns
        + " FROM cm1_claim_verifications WHERE verification_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
    sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
    std::optional<Cm1ClaimVerification> out;
    if (sqlite3_step(stmt) == SQLITE_ROW) out = claimVerificationFromRow(stmt);
    sqlite3_finalize(stmt);
    return out;
}

std::vector<Cm1EvidenceJudgmentRecord> loadJudgmentsForAnswer(sqlite3* db, const std::string& answer_id) {
    std::vector<Cm1EvidenceJudgmentRecord> rows;
    const char* sql =
        "SELECT judgment_id, answer_id, content_sha256, estimator_version, payload_json, created_at_ms "
        "FROM cm1_evidence_judgments WHERE answer_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return rows;
    sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Cm1EvidenceJudgmentRecord record;
        record.judgment_id = safe_col_text(stmt, 0);
        record.answer_id = safe_col_text(stmt, 1);
        record.content_sha256 = safe_col_text(stmt, 2);
        record.estimator_version = safe_col_text(stmt, 3);
        record.payload_json = safe_col_text(stmt, 4);
        record.created_at_ms = sqlite3_column_int64(stmt, 5);
        rows.push_back(std::move(record));
    }
    sqlite3_finalize(stmt);
    return rows;
}

enum class StepInclusionStatus { MissingPlan, StepAbsent, Unknown, Known };

struct StepInclusionRead {
    StepInclusionStatus status = StepInclusionStatus::MissingPlan;
    std::vector<std::string> hashes;
    std::string body;
};

StepInclusionRead readPlanInclusion(const std::string& body, const std::string& step_id) {
    StepInclusionRead read;
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(body);
    } catch (...) {
        read.status = StepInclusionStatus::Unknown;
        return read;
    }
    if (!parsed.is_object() || !parsed.contains("steps") || !parsed["steps"].is_array()) {
        read.status = StepInclusionStatus::Unknown;
        return read;
    }
    for (const auto& step : parsed["steps"]) {
        if (!step.is_object() || step.value("step_id", "") != step_id) continue;
        if (!step.contains("result") || !step["result"].is_object()) {
            read.status = StepInclusionStatus::Unknown;
            return read;
        }
        const auto& result = step["result"];
        if (!result.value("cm1_included_in_synthesis_available", false)
            || !result.contains("cm1_included_content_sha256")
            || !result["cm1_included_content_sha256"].is_array()) {
            read.status = StepInclusionStatus::Unknown;
            return read;
        }
        for (const auto& hash : result["cm1_included_content_sha256"]) {
            if (!hash.is_string()) {
                read.hashes.clear();
                read.status = StepInclusionStatus::Unknown;
                return read;
            }
            read.hashes.push_back(hash.get<std::string>());
        }
        read.status = StepInclusionStatus::Known;
        return read;
    }
    read.status = StepInclusionStatus::StepAbsent;
    return read;
}

StepInclusionRead inclusionFromPlanTable(sqlite3* db,
                                         const char* sql,
                                         const std::string& plan_id,
                                         const std::string& step_id) {
    StepInclusionRead read;
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        read.status = StepInclusionStatus::Unknown;
        return read;
    }
    sqlite3_bind_text(stmt, 1, plan_id.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        read.status = StepInclusionStatus::MissingPlan;
        return read;
    }
    const std::string body = safe_col_text(stmt, 0);
    sqlite3_finalize(stmt);
    if (body.empty()) {
        read.status = StepInclusionStatus::Unknown;
        return read;
    }
    read = readPlanInclusion(body, step_id);
    read.body = body;
    return read;
}

std::optional<std::string> knownInclusionPlanBody(sqlite3* db,
                                                  const std::string& plan_id,
                                                  const std::string& step_id) {
    if (!db || plan_id.empty() || step_id.empty()) return std::nullopt;
    const auto active = inclusionFromPlanTable(
        db, "SELECT steps_json FROM active_plans WHERE plan_id = ?;", plan_id, step_id);
    if (active.status == StepInclusionStatus::Known) return active.body;
    if (active.status == StepInclusionStatus::Unknown) return std::nullopt;
    const auto durable = inclusionFromPlanTable(
        db, "SELECT plan_json FROM plans WHERE plan_id = ?;", plan_id, step_id);
    if (durable.status == StepInclusionStatus::Known) return durable.body;
    return std::nullopt;
}

std::optional<Cm1SynthesisInclusion> readSynthesisInclusion(sqlite3* db,
                                                            const std::string& plan_id,
                                                            const std::string& step_id) {
    if (!db || plan_id.empty() || step_id.empty()) return std::nullopt;
    const auto active = inclusionFromPlanTable(
        db, "SELECT steps_json FROM active_plans WHERE plan_id = ?;", plan_id, step_id);
    if (active.status == StepInclusionStatus::Known) {
        Cm1SynthesisInclusion inclusion;
        inclusion.content_sha256 = active.hashes;
        return inclusion;
    }
    if (active.status == StepInclusionStatus::Unknown) return std::nullopt;
    const auto durable = inclusionFromPlanTable(
        db, "SELECT plan_json FROM plans WHERE plan_id = ?;", plan_id, step_id);
    if (durable.status == StepInclusionStatus::Known) {
        Cm1SynthesisInclusion inclusion;
        inclusion.content_sha256 = durable.hashes;
        return inclusion;
    }
    return std::nullopt;
}

bool lowercaseSha256Hex(const std::string& text) {
    if (text.size() != 64) return false;
    for (unsigned char ch : text) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    }
    return true;
}

std::optional<std::vector<std::string>> heldOutMembers(const Cm1ClaimVerification& record) {
    if (record.source_kind == kCm1HeldOutPassageSource) {
        if (!lowercaseSha256Hex(record.source_ref)) return std::nullopt;
        return std::vector<std::string>{record.source_ref};
    }
    if (record.source_kind != kCm1HeldOutBatchSource) return std::vector<std::string>{};
    std::vector<std::string> members;
    std::size_t begin = 0;
    const std::string& text = record.source_ref;
    while (begin <= text.size()) {
        const std::size_t end = text.find('\n', begin);
        if (end == std::string::npos) {
            members.push_back(text.substr(begin));
            break;
        }
        members.push_back(text.substr(begin, end - begin));
        begin = end + 1;
        if (begin == text.size()) {
            members.emplace_back();
            break;
        }
    }
    if (members.size() < 2) return std::nullopt;
    for (std::size_t i = 0; i < members.size(); ++i) {
        if (!lowercaseSha256Hex(members[i])) return std::nullopt;
        if (i > 0 && !(members[i - 1] < members[i])) return std::nullopt;
    }
    return members;
}

bool heldOutSourceAllowed(sqlite3* db,
                          const Cm1ClaimVerification& record,
                          const Cm1AnswerVersionRecord& answer,
                          const std::string& evidence_chunk_sha256) {
    if (record.source_kind != kCm1HeldOutPassageSource && record.source_kind != kCm1HeldOutBatchSource) {
        return true;
    }
    const auto members = heldOutMembers(record);
    if (!members || (record.source_kind == kCm1HeldOutBatchSource && members->size() < 2)) {
        return false;
    }
    const auto inclusion = readSynthesisInclusion(db, answer.plan_id, answer.step_id);
    if (!inclusion) return false;
    for (const auto& member : *members) {
        if (!evidence_chunk_sha256.empty() && member == evidence_chunk_sha256) return false;
        for (const auto& hash : inclusion->content_sha256) {
            if (member == hash) return false;
        }
    }
    return true;
}

bool bindClaimVerification(sqlite3_stmt* stmt, const Cm1ClaimVerification& record) {
    sqlite3_bind_text(stmt, 1, record.verification_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, record.claim_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, record.claim_sha256.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, record.answer_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, record.content_sha256.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, record.protocol_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, record.protocol_version.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 8, record.reviewer_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 9, record.verifier_kind.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 10, record.verifier_version.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 11, record.method_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 12, record.outcome.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 13, record.source_kind.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 14, record.source_ref.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 15, record.calibration_eligibility.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 16, record.egar_eligibility.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 17, record.revision_reason.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 18, record.supersedes_verification_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 19, record.revision_index);
    sqlite3_bind_int64(stmt, 20, record.created_at_ms);
    sqlite3_bind_text(stmt, 21, record.evidence_label.c_str(), -1, SQLITE_TRANSIENT);
    return true;
}

} // namespace

bool SQLiteMemoryRepository::insertCm1ClaimVerification(const Cm1ClaimVerification& record) {
    try {
        if (!db_ || !db_->handle || record.verification_id.empty() || record.claim_id.empty()
            || record.answer_id.empty()) {
            return false;
        }
        if (const auto existing = loadClaimVerificationById(db_->handle, record.verification_id)) {
            return sameStoredVerification(*existing, record);
        }

        const auto answer = loadCm1AnswerVersion(record.answer_id);
        if (!answer || answer->content_sha256 != record.content_sha256) return false;

        std::optional<Cm1StoredClaim> matched;
        for (const auto& judgment : loadJudgmentsForAnswer(db_->handle, record.answer_id)) {
            const auto claim = findCm1StoredClaim(judgment, record.claim_id);
            if (!claim || claim->claim_sha256 != record.claim_sha256) continue;
            if (claim->answer_content_sha256 != record.content_sha256) continue;
            matched = claim;
            break;
        }
        if (!matched || !cm1ClaimVerificationConforms(record, *answer, *matched)) return false;
        if (!heldOutSourceAllowed(db_->handle, record, *answer, matched->evidence_chunk_sha256)) return false;

        const auto latest = latestCm1ClaimVerification(
            record.claim_id, record.protocol_id, record.protocol_version, record.reviewer_id,
            record.verifier_kind, record.verifier_version, record.method_id);
        if (record.revision_reason == "initial") {
            if (latest) return false;
        } else if (record.revision_reason == "correction") {
            if (!latest || record.revision_index != latest->revision_index + 1
                || record.supersedes_verification_id != latest->verification_id) {
                return false;
            }
        } else {
            return false;
        }

        const std::string sql = std::string("INSERT INTO cm1_claim_verifications (")
            + kClaimVerificationColumns + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return false;
        bindClaimVerification(stmt, record);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

std::optional<Cm1ClaimVerification> SQLiteMemoryRepository::latestCm1ClaimVerification(
    const std::string& claim_id,
    const std::string& protocol_id,
    const std::string& protocol_version,
    const std::string& reviewer_id,
    const std::string& verifier_kind,
    const std::string& verifier_version,
    const std::string& method_id) {
    try {
        if (!db_ || !db_->handle || claim_id.empty() || protocol_id.empty() || protocol_version.empty()
            || reviewer_id.empty() || verifier_kind.empty() || method_id.empty()) {
            return std::nullopt;
        }
        const std::string sql = std::string("SELECT ") + kClaimVerificationColumns
            + " FROM cm1_claim_verifications "
            "WHERE claim_id = ? AND protocol_id = ? AND protocol_version = ? AND reviewer_id = ? "
            "AND verifier_kind = ? AND verifier_version = ? AND method_id = ? "
            "ORDER BY revision_index DESC LIMIT 1;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        sqlite3_bind_text(stmt, 1, claim_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, protocol_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, protocol_version.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, reviewer_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, verifier_kind.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, verifier_version.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 7, method_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1ClaimVerification> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            out = claimVerificationFromRow(stmt);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<Cm1ClaimVerification> SQLiteMemoryRepository::listCm1ClaimVerifications(
    const std::string& answer_id) {
    std::vector<Cm1ClaimVerification> rows;
    try {
        if (!db_ || !db_->handle || answer_id.empty()) return rows;
        const std::string sql = std::string("SELECT ") + kClaimVerificationColumns
            + " FROM cm1_claim_verifications WHERE answer_id = ? "
            "ORDER BY created_at_ms, verification_id;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return rows;
        sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            rows.push_back(claimVerificationFromRow(stmt));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return rows;
}

std::optional<Cm1AnswerVersionRecord> SQLiteMemoryRepository::loadCm1AnswerVersion(
    const std::string& answer_id) {
    try {
        if (!db_ || !db_->handle || answer_id.empty()) return std::nullopt;
        const char* sql =
            "SELECT answer_id, plan_id, step_id, generation_id, content_sha256, created_at_ms "
            "FROM cm1_answer_versions WHERE answer_id = ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        sqlite3_bind_text(stmt, 1, answer_id.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<Cm1AnswerVersionRecord> out;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            Cm1AnswerVersionRecord record;
            record.answer_id = safe_col_text(stmt, 0);
            record.plan_id = safe_col_text(stmt, 1);
            record.step_id = safe_col_text(stmt, 2);
            record.generation_id = safe_col_text(stmt, 3);
            record.content_sha256 = safe_col_text(stmt, 4);
            record.created_at_ms = sqlite3_column_int64(stmt, 5);
            out = std::move(record);
        }
        sqlite3_finalize(stmt);
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<Cm1SynthesisInclusion> SQLiteMemoryRepository::loadCm1SynthesisInclusion(
    const std::string& plan_id, const std::string& step_id) {
    try {
        if (!db_ || !db_->handle) return std::nullopt;
        return readSynthesisInclusion(db_->handle, plan_id, step_id);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> SQLiteMemoryRepository::loadCm1PersistedPlanBody(
    const std::string& plan_id, const std::string& step_id) {
    try {
        if (!db_ || !db_->handle) return std::nullopt;
        return knownInclusionPlanBody(db_->handle, plan_id, step_id);
    } catch (...) {
        return std::nullopt;
    }
}

bool SQLiteMemoryRepository::enqueueCognitiveMetrics(const std::string& plan_id,
                                                     const std::string& payload_json) {
    try {
        if (!db_ || !db_->handle || plan_id.empty() || payload_json.empty()) return false;
        const char* sql =
            "INSERT OR IGNORE INTO cognitive_metrics_outbox "
            "(metrics_id, plan_id, payload_json, flushed) VALUES (?, ?, ?, 0);";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, plan_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, plan_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, payload_json.c_str(), -1, SQLITE_TRANSIENT);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

std::vector<MemoryRepository::CognitiveMetricsOutboxRow>
SQLiteMemoryRepository::pendingCognitiveMetrics() {
    std::vector<CognitiveMetricsOutboxRow> rows;
    try {
        if (!db_ || !db_->handle) return rows;
        const char* sql =
            "SELECT metrics_id, plan_id, payload_json FROM cognitive_metrics_outbox "
            "WHERE flushed = 0 ORDER BY metrics_id;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return rows;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            CognitiveMetricsOutboxRow row;
            row.metrics_id = safe_col_text(stmt, 0);
            row.plan_id = safe_col_text(stmt, 1);
            row.payload_json = safe_col_text(stmt, 2);
            rows.push_back(std::move(row));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return rows;
}

bool SQLiteMemoryRepository::markCognitiveMetricsFlushed(const std::string& metrics_id) {
    try {
        if (!db_ || !db_->handle || metrics_id.empty()) return false;
        const char* sql =
            "UPDATE cognitive_metrics_outbox SET flushed = 1 WHERE metrics_id = ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        sqlite3_bind_text(stmt, 1, metrics_id.c_str(), -1, SQLITE_TRANSIENT);
        const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    } catch (...) {
        return false;
    }
}

bool SQLiteMemoryRepository::storeEpisodeStep(const EpisodeStepRecord& step) {
    try {
        if (step.goal_id.empty()) return false;
        if (!observation_unique_ && episodeObservationExists(step)) return true;
        const char* sql = observation_unique_
            ? "INSERT INTO episode_steps (episode_id, goal_id, step_index, state_summary, action_taken, result_status, embedding_blob, timestamp_ms) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
              "ON CONFLICT(goal_id, step_index, result_status, state_summary, action_taken) DO NOTHING;"
            : "INSERT INTO episode_steps (episode_id, goal_id, step_index, state_summary, action_taken, result_status, embedding_blob, timestamp_ms) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, step.episode_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, step.goal_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 3, step.step_index);
        sqlite3_bind_text(stmt, 4, step.state_summary.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 5, step.action_taken.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 6, step.result_status.c_str(), -1, SQLITE_STATIC);
        if (!step.embedding.empty()) {
            sqlite3_bind_blob(stmt, 7, step.embedding.data(), static_cast<int>(step.embedding.size() * sizeof(float)), SQLITE_STATIC);
        } else {
            sqlite3_bind_null(stmt, 7);
        }
        sqlite3_bind_int64(stmt, 8, step.timestamp_ms);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::vector<MemoryRepository::EpisodeStepRecord> SQLiteMemoryRepository::getRecentEpisodeSteps(const std::string& goal_id, int n) {
    std::vector<EpisodeStepRecord> results;
    try {
        const char* sql = "SELECT episode_id, goal_id, step_index, state_summary, action_taken, result_status, embedding_blob, timestamp_ms "
                          "FROM episode_steps WHERE goal_id = ? ORDER BY timestamp_ms DESC LIMIT ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;
        
        sqlite3_bind_text(stmt, 1, goal_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 2, n);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            EpisodeStepRecord rec;
            rec.episode_id = safe_col_text(stmt, 0);
            rec.goal_id = safe_col_text(stmt, 1);
            rec.step_index = sqlite3_column_int(stmt, 2);
            rec.state_summary = safe_col_text(stmt, 3);
            rec.action_taken = safe_col_text(stmt, 4);
            rec.result_status = safe_col_text(stmt, 5);
            
            const void* blob = sqlite3_column_blob(stmt, 6);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 6);
                int count = bytes / sizeof(float);
                rec.embedding.resize(count);
                std::memcpy(rec.embedding.data(), blob, bytes);
            }
            rec.timestamp_ms = sqlite3_column_int64(stmt, 7);
            results.push_back(std::move(rec));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    // Reverse to get chronological order
    std::reverse(results.begin(), results.end());
    return results;
}

std::vector<MemoryRepository::EpisodeStepRecord> SQLiteMemoryRepository::getAllEpisodeSteps() {
    std::vector<EpisodeStepRecord> results;
    try {
        const char* sql = "SELECT episode_id, goal_id, step_index, state_summary, action_taken, result_status, embedding_blob, timestamp_ms "
                          "FROM episode_steps ORDER BY timestamp_ms DESC;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            EpisodeStepRecord rec;
            rec.episode_id = safe_col_text(stmt, 0);
            rec.goal_id = safe_col_text(stmt, 1);
            rec.step_index = sqlite3_column_int(stmt, 2);
            rec.state_summary = safe_col_text(stmt, 3);
            rec.action_taken = safe_col_text(stmt, 4);
            rec.result_status = safe_col_text(stmt, 5);
            
            const void* blob = sqlite3_column_blob(stmt, 6);
            if (blob) {
                int bytes = sqlite3_column_bytes(stmt, 6);
                int count = bytes / sizeof(float);
                rec.embedding.resize(count);
                std::memcpy(rec.embedding.data(), blob, bytes);
            }
            rec.timestamp_ms = sqlite3_column_int64(stmt, 7);
            results.push_back(std::move(rec));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

bool SQLiteMemoryRepository::saveExperiment(const CognateExperimentRecord& record) {
    try {
        const char* sql = "INSERT OR REPLACE INTO cognate_experiments (experiment_id, name, hypothesis, configuration_json, results_json, created_at, status) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, record.experiment_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, record.name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, record.hypothesis.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, record.configuration_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 5, record.results_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 6, record.created_at);
        sqlite3_bind_text(stmt, 7, record.status.c_str(), -1, SQLITE_STATIC);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::CognateExperimentRecord> SQLiteMemoryRepository::loadExperiment(const std::string& experiment_id) {
    try {
        const char* sql = "SELECT experiment_id, name, hypothesis, configuration_json, results_json, created_at, status "
                          "FROM cognate_experiments WHERE experiment_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        
        sqlite3_bind_text(stmt, 1, experiment_id.c_str(), -1, SQLITE_STATIC);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateExperimentRecord rec;
            rec.experiment_id = safe_col_text(stmt, 0);
            rec.name = safe_col_text(stmt, 1);
            rec.hypothesis = safe_col_text(stmt, 2);
            rec.configuration_json = safe_col_text(stmt, 3);
            rec.results_json = safe_col_text(stmt, 4);
            rec.created_at = sqlite3_column_int64(stmt, 5);
            rec.status = safe_col_text(stmt, 6);
            sqlite3_finalize(stmt);
            return rec;
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return std::nullopt;
}

std::vector<MemoryRepository::CognateExperimentRecord> SQLiteMemoryRepository::getAllExperiments() {
    std::vector<CognateExperimentRecord> results;
    try {
        const char* sql = "SELECT experiment_id, name, hypothesis, configuration_json, results_json, created_at, status "
                          "FROM cognate_experiments ORDER BY created_at DESC;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateExperimentRecord rec;
            rec.experiment_id = safe_col_text(stmt, 0);
            rec.name = safe_col_text(stmt, 1);
            rec.hypothesis = safe_col_text(stmt, 2);
            rec.configuration_json = safe_col_text(stmt, 3);
            rec.results_json = safe_col_text(stmt, 4);
            rec.created_at = sqlite3_column_int64(stmt, 5);
            rec.status = safe_col_text(stmt, 6);
            results.push_back(std::move(rec));
        }
        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

bool SQLiteMemoryRepository::saveProblemState(const ProblemStateRecord& record) {
    try {
        const char* sql = "INSERT OR REPLACE INTO problem_states (problem_id, goal_id, state_json, iteration_count, confidence_score, created_at, updated_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, record.problem_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, record.goal_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, record.state_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 4, record.iteration_count);
        sqlite3_bind_double(stmt, 5, static_cast<double>(record.confidence_score));
        sqlite3_bind_int64(stmt, 6, record.created_at);
        sqlite3_bind_int64(stmt, 7, record.updated_at);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::ProblemStateRecord> SQLiteMemoryRepository::loadProblemState(const std::string& problem_id) {
    try {
        const char* sql = "SELECT problem_id, goal_id, state_json, iteration_count, confidence_score, created_at, updated_at FROM problem_states WHERE problem_id = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        
        sqlite3_bind_text(stmt, 1, problem_id.c_str(), -1, SQLITE_STATIC);
        
        std::optional<ProblemStateRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            ProblemStateRecord rec;
            rec.problem_id = safe_col_text(stmt, 0);
            rec.goal_id = safe_col_text(stmt, 1);
            rec.state_json = safe_col_text(stmt, 2);
            rec.iteration_count = sqlite3_column_int(stmt, 3);
            rec.confidence_score = static_cast<float>(sqlite3_column_double(stmt, 4));
            rec.created_at = sqlite3_column_int64(stmt, 5);
            rec.updated_at = sqlite3_column_int64(stmt, 7);
            result = rec;
        }
        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::optional<MemoryRepository::ProblemStateRecord> SQLiteMemoryRepository::getLatestProblemState(const std::string& goal_id) {
    try {
        const char* sql = "SELECT problem_id, goal_id, state_json, iteration_count, confidence_score, created_at, updated_at FROM problem_states WHERE goal_id = ? ORDER BY updated_at DESC LIMIT 1;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;
        
        sqlite3_bind_text(stmt, 1, goal_id.c_str(), -1, SQLITE_STATIC);
        
        std::optional<ProblemStateRecord> result;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            ProblemStateRecord rec;
            rec.problem_id = safe_col_text(stmt, 0);
            rec.goal_id = safe_col_text(stmt, 1);
            rec.state_json = safe_col_text(stmt, 2);
            rec.iteration_count = sqlite3_column_int(stmt, 3);
            rec.confidence_score = static_cast<float>(sqlite3_column_double(stmt, 4));
            rec.created_at = sqlite3_column_int64(stmt, 5);
            rec.updated_at = sqlite3_column_int64(stmt, 6);
            result = rec;
        }
        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

} // namespace Thoth
