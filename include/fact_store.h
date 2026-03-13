/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — FactStore for structured world knowledge
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_FACT_STORE_H
#define THOTH_FACT_STORE_H

#include "sqlite_memory_repository.h"
#include <string>
#include <vector>
#include <optional>

namespace Thoth {

/**
 * @struct Fact
 * @brief Represents a single structured piece of knowledge.
 */
struct Fact {
    std::string key;
    std::string value;
    float confidence;
    std::string source;
    int64_t last_updated_ms;
};

/**
 * @class FactStore
 * @brief Manages the persistent structured knowledge base.
 */
class FactStore {
public:
    explicit FactStore(SQLiteMemoryRepository& repo);

    bool upsert(const Fact& fact);
    std::optional<Fact> get(const std::string& key);
    std::vector<Fact> search(const std::string& query);
    bool remove(const std::string& key);

private:
    SQLiteMemoryRepository& repo_;
};

} // namespace Thoth

#endif // THOTH_FACT_STORE_H
