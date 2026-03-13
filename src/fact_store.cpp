/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — FactStore implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/fact_store.h"

namespace Thoth {

FactStore::FactStore(SQLiteMemoryRepository& repo) : repo_(repo) {}

bool FactStore::upsert(const Fact& fact) {
    MemoryRepository::FactRecord rec;
    rec.key = fact.key;
    rec.value = fact.value;
    rec.confidence = fact.confidence;
    rec.source = fact.source;
    rec.last_updated_ms = fact.last_updated_ms;
    return repo_.upsertFact(rec);
}

std::optional<Fact> FactStore::get(const std::string& key) {
    auto rec = repo_.getFact(key);
    if (!rec) return std::nullopt;

    Fact fact;
    fact.key = rec->key;
    fact.value = rec->value;
    fact.confidence = rec->confidence;
    fact.source = rec->source;
    fact.last_updated_ms = rec->last_updated_ms;
    return fact;
}

std::vector<Fact> FactStore::search(const std::string& query) {
    auto records = repo_.searchFacts(query);
    std::vector<Fact> results;
    for (const auto& rec : records) {
        results.push_back({
            rec.key,
            rec.value,
            rec.confidence,
            rec.source,
            rec.last_updated_ms
        });
    }
    return results;
}

bool FactStore::remove(const std::string& key) {
    return repo_.deleteFact(key);
}

} // namespace Thoth
