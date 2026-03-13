/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — web_scrape tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_WEB_SCRAPE_TOOL_H
#define THOTH_WEB_SCRAPE_TOOL_H

#include "itool.h"

/**
 * @class WebScrapeTool
 * @brief Fetches and cleans text content from a provided URL.
 *
 * Implemented according to TOOLS.md v1.0.
 */
class WebScrapeTool : public ITool {
public:
    std::string name() const override { return "web_scrape"; }
    
    std::string description() const override {
        return "Fetches the content of a web page and returns the cleaned text.";
    }

    nlohmann::json input_schema() const override;
    bool requires_confirmation() const override;
    nlohmann::json execute(const nlohmann::json& input) const override;


private:
    /**
     * @brief Simplistic HTML to text converter.
     */
    std::string cleanHtml(const std::string& html) const;

    /**
     * @brief Curl write callback.
     */
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

#endif // THOTH_WEB_SCRAPE_TOOL_H
