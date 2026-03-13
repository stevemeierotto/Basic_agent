/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — gmail_read_messages tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_GMAIL_READ_MESSAGES_TOOL_H
#define THOTH_GMAIL_READ_MESSAGES_TOOL_H

#include "itool.h"

/**
 * @class GmailReadMessagesTool
 * @brief Lists or reads messages from a Gmail account (Mock for Phase 5.3).
 *
 * Implemented according to TOOLS.md v1.0.
 */
class GmailReadMessagesTool : public ITool {
public:
    std::string name() const override { return "gmail_read_messages"; }
    
    std::string description() const override {
        return "Lists recent messages or reads a specific message from Gmail.";
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;

};

#endif // THOTH_GMAIL_READ_MESSAGES_TOOL_H
