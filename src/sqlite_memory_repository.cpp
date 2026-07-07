#include "../include/sqlite_memory_repository.h"
#include "../include/decision_trace.h"
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

namespace Thoth {

static std::string safe_col_text(sqlite3_stmt* stmt, int col) {
    const unsigned char* text = sqlite3_column_text(stmt, col);
    return text ? std::string(reinterpret_cast<const char*>(text)) : "";
}

static bool injectConsolidationFail(const char* stage) {
    const char* env = std::getenv("THOTH_INJECT_CONSOLIDATION_FAIL");
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
            "  embedding_version INTEGER NOT NULL"
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
            "  tier INTEGER DEFAULT 0"
            ");"
            "CREATE TABLE IF NOT EXISTS strategies ("
            "  strategy_id TEXT PRIMARY KEY,"
            "  description TEXT NOT NULL,"
            "  step_pattern_json TEXT NOT NULL,"
            "  success_rate REAL NOT NULL,"
            "  created_at INTEGER NOT NULL"
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
            ");";

        char* errMsg = nullptr;
        rc = sqlite3_exec(db_->handle, schema, nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            std::cerr << "[SQLiteMemoryRepository] Schema creation failed: " << (errMsg ? errMsg : "Unknown error") << "\n";
            sqlite3_free(errMsg);
        }

        // Session table migrations
        auto add_col = [&](const std::string& table, const std::string& col, const std::string& type) {
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
            }
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

        // Strategy table migrations
        add_col("strategies", "description", "TEXT");
        add_col("strategies", "step_pattern_json", "TEXT");
        add_col("strategies", "success_rate", "REAL");
        add_col("strategies", "created_at", "INTEGER");

    } catch (const std::exception& e) {
        std::cerr << "[SQLiteMemoryRepository] Initialization exception: " << e.what() << "\n";
    }
}

SQLiteMemoryRepository::~SQLiteMemoryRepository() = default;

bool SQLiteMemoryRepository::beginTransaction() {
    return sqlite3_exec(db_->handle, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr) == SQLITE_OK;
}

bool SQLiteMemoryRepository::commit() {
    return sqlite3_exec(db_->handle, "COMMIT;", nullptr, nullptr, nullptr) == SQLITE_OK;
}

bool SQLiteMemoryRepository::rollback() {
    return sqlite3_exec(db_->handle, "ROLLBACK;", nullptr, nullptr, nullptr) == SQLITE_OK;
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
        const char* sql = "INSERT OR REPLACE INTO past_plans (plan_id, goal, outline, success_score, duration_ms, failure_count, goal_embedding, embedding_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
        
        sqlite3_bind_text(stmt, 1, plan.plan_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, plan.goal.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, plan.outline.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, plan.success_score);
        sqlite3_bind_int64(stmt, 5, plan.duration_ms);
        sqlite3_bind_int(stmt, 6, plan.failure_count);
        sqlite3_bind_blob(stmt, 7, plan.goal_embedding.data(), static_cast<int>(plan.goal_embedding.size() * sizeof(float)), SQLITE_STATIC);
        sqlite3_bind_int(stmt, 8, version);
        
        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::vector<MemoryRepository::PastPlanRecord> SQLiteMemoryRepository::getAllPastPlans(int version) {
    std::vector<PastPlanRecord> plans;
    try {
        const char* sql = "SELECT plan_id, goal, outline, success_score, duration_ms, failure_count, goal_embedding FROM past_plans WHERE embedding_version = ?;";
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
        const char* sql = "INSERT INTO graph_edges (from_id, to_id, weight, success_count, failure_count, last_used_ms) VALUES (?, ?, ?, ?, ?, ?) "
                          "ON CONFLICT(from_id, to_id) DO UPDATE SET weight=excluded.weight, success_count=graph_edges.success_count + excluded.success_count, "
                          "failure_count=graph_edges.failure_count + excluded.failure_count, last_used_ms=excluded.last_used_ms;";
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

bool SQLiteMemoryRepository::storeActivePlan(const ActivePlanRecord& plan) {
    try {
        const char* sql = "INSERT OR REPLACE INTO active_plans (plan_id, session_id, goal, steps_json, current_index, controller_state, created_at_ms, updated_at_ms) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, plan.plan_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, plan.session_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, plan.goal.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, plan.steps_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 5, plan.current_index);
        sqlite3_bind_text(stmt, 6, plan.controller_state.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 7, plan.created_at_ms);
        sqlite3_bind_int64(stmt, 8, plan.updated_at_ms);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
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
        const char* sql = "INSERT OR REPLACE INTO trajectories (trajectory_id, goal, trajectory_json, success_score, embedding, created_at, usage_count, tier) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
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

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::CognateTrajectoryRecord> SQLiteMemoryRepository::loadTrajectory(const std::string& trajectory_id) {
    try {
        const char* sql = "SELECT trajectory_id, goal, trajectory_json, success_score, embedding, created_at, usage_count, tier FROM trajectories WHERE trajectory_id = ?;";
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
            result = rec;
        }

        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::vector<MemoryRepository::CognateTrajectoryRecord> SQLiteMemoryRepository::getAllTrajectories() {
    std::vector<CognateTrajectoryRecord> results;
    try {
        const char* sql = "SELECT trajectory_id, goal, trajectory_json, success_score, embedding, created_at, usage_count, tier FROM trajectories;";
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
            results.push_back(std::move(rec));
        }

        sqlite3_finalize(stmt);
    } catch (...) {}
    return results;
}

bool SQLiteMemoryRepository::saveStrategy(const CognateStrategyRecord& record) {
    try {
        const char* sql = "INSERT OR REPLACE INTO strategies (strategy_id, description, step_pattern_json, success_rate, created_at) "
                          "VALUES (?, ?, ?, ?, ?);";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, record.strategy_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, record.description.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, record.step_pattern_json.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, static_cast<double>(record.success_rate));
        sqlite3_bind_int64(stmt, 5, record.created_at);

        bool success = (sqlite3_step(stmt) == SQLITE_DONE);
        sqlite3_finalize(stmt);
        return success;
    } catch (...) { return false; }
}

std::optional<MemoryRepository::CognateStrategyRecord> SQLiteMemoryRepository::loadStrategy(const std::string& strategy_id) {
    try {
        const char* sql = "SELECT strategy_id, description, step_pattern_json, success_rate, created_at FROM strategies WHERE strategy_id = ?;";
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
            result = rec;
        }

        sqlite3_finalize(stmt);
        return result;
    } catch (...) { return std::nullopt; }
}

std::vector<MemoryRepository::CognateStrategyRecord> SQLiteMemoryRepository::getAllStrategies() {
    std::vector<CognateStrategyRecord> results;
    try {
        const char* sql = "SELECT strategy_id, description, step_pattern_json, success_rate, created_at FROM strategies;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            CognateStrategyRecord rec;
            rec.strategy_id = safe_col_text(stmt, 0);
            rec.description = safe_col_text(stmt, 1);
            rec.step_pattern_json = safe_col_text(stmt, 2);
            rec.success_rate = static_cast<float>(sqlite3_column_double(stmt, 3));
            rec.created_at = sqlite3_column_int64(stmt, 4);
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

std::vector<MemoryRepository::ArchivedTurnRecord> SQLiteMemoryRepository::getArchivedMessages(const std::string& sessionId) {
    std::vector<ArchivedTurnRecord> results;
    try {
        const char* sql = "SELECT archive_id, session_id, original_timestamp_ms, role, content, metadata_json, archived_at_ms, summary_version "
                          "FROM archived_turns WHERE session_id = ? ORDER BY original_timestamp_ms ASC;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_->handle, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;
        
        sqlite3_bind_text(stmt, 1, sessionId.c_str(), -1, SQLITE_STATIC);
        
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
    } catch (...) {}
    return results;
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

bool SQLiteMemoryRepository::storeEpisodeStep(const EpisodeStepRecord& step) {
    try {
        const char* sql = "INSERT INTO episode_steps (episode_id, goal_id, step_index, state_summary, action_taken, result_status, embedding_blob, timestamp_ms) "
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
