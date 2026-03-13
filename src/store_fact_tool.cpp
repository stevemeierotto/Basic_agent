/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — store_fact tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/store_fact_tool.h"
#include <chrono>

StoreFactTool::StoreFactTool(Thoth::FactStore& fact_store) : fact_store_(fact_store) {}

nlohmann::json StoreFactTool::input_schema() const {
    return {
        {"type", "object"},
        {"properties", {
            {"key", {{"type", "string"}, {"description", "The unique key for the fact."}}},
            {"value", {{"type", "string"}, {"description", "The value or content of the fact."}}},
            {"confidence", {{"type", "number"}, {"description", "Confidence score (0.0 to 1.0). Defaults to 1.0."}}},
            {"source", {{"type", "string"}, {"description", "The source of the information."}}}
        }},
        {"required", {"key", "value"}},
        {"additionalProperties", false}
    };
}

bool StoreFactTool::requires_confirmation() const {
    return false;
}

nlohmann::json StoreFactTool::execute(const nlohmann::json& input) const {
    Thoth::Fact fact;
    fact.key = input.at("key");
    fact.value = input.at("value");
    fact.confidence = input.value("confidence", 1.0f);
    fact.source = input.value("source", "agent");
    fact.last_updated_ms = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);

    if (fact_store_.upsert(fact)) {
        return {
            {"status", "success"},
            {"data", {{"key", fact.key}}},
            {"error_message", nullptr}
        };
    } else {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "Failed to store fact in the database."}
        };
    }
}
