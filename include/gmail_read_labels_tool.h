#pragma once

#include "itool.h"
#include <curl/curl.h>

/**
 * @brief Tool for listing Gmail labels.
 * Implemented according to TOOLS.md v1.0
 */
class GmailReadLabelsTool : public ITool {
public:
    GmailReadLabelsTool();
    ~GmailReadLabelsTool() override;

    std::string name() const override { return "gmail_read_labels"; }
    
    std::string description() const override { 
        return "Lists all available labels for the authenticated Gmail account."; 
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;


private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
};
