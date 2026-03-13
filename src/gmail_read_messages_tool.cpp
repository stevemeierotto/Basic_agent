/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — gmail_read_messages tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/gmail_read_messages_tool.h"

nlohmann::json GmailReadMessagesTool::input_schema() const {
    return {
        {"type", "object"},
        {"properties", {
            {"operation", {{"type", "string"}, {"enum", {"list", "read"}}}},
            {"label_id", {{"type", "string"}, {"description", "The label ID to list from. Defaults to INBOX."}}},
            {"message_id", {{"type", "string"}, {"description", "The message ID to read. Required for operation: read."}}},
            {"max_results", {{"type", "integer"}, {"description", "Maximum number of messages to list. Defaults to 5."}}},
            {"confirmed", {{"type", "boolean"}, {"description", "Must be true to proceed."}}},
            {"credential_id", {{"type", "string"}, {"description", "Gmail credential ID."}}}
        }},
        {"required", {"operation", "credential_id"}},
        {"additionalProperties", false}
    };
}

bool GmailReadMessagesTool::requires_confirmation() const {
    return true;
}

nlohmann::json GmailReadMessagesTool::execute(const nlohmann::json& input) const {

    std::string operation = input.at("operation");

    if (operation == "list") {
        std::string label = input.value("label_id", "INBOX");
        int max = input.value("max_results", 5);

        nlohmann::json messages = nlohmann::json::array();
        for (int i = 0; i < std::min(max, 3); ++i) {
            messages.push_back({
                {"id", "msg-00" + std::to_string(i)},
                {"snippet", "Hello from Thoth development team! (Mock)"},
                {"subject", "Update on Phase 5.3"},
                {"from", "dev@thoth.ai"}
            });
        }

        return {
            {"status", "success"},
            {"data", {
                {"label", label},
                {"messages", messages}
            }},
            {"error_message", nullptr}
        };
    } else if (operation == "read") {
        if (!input.contains("message_id")) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", "message_id is required for read operation."}
            };
        }

        std::string mid = input.at("message_id");
        return {
            {"status", "success"},
            {"data", {
                {"id", mid},
                {"body", "This is the full body of the mock message " + mid + ".\nIt contains multiple lines of text."},
                {"headers", {{"subject", "Mock Subject"}, {"date", "2026-03-08"}}}
            }},
            {"error_message", nullptr}
        };
    }

    return {
        {"status", "error"},
        {"data", nlohmann::json::object()},
        {"error_message", "Unknown operation: " + operation}
    };
}
