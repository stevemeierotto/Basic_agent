/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — web_scrape tool implementation
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/web_scrape_tool.h"
#include <curl/curl.h>
#include <regex>
#include <iostream>

nlohmann::json WebScrapeTool::input_schema() const {
    return {
        {"type", "object"},
        {"properties", {
            {"url", {{"type", "string"}, {"description", "The URL of the web page to scrape."}}}
        }},
        {"required", {"url"}},
        {"additionalProperties", false}
    };
}

bool WebScrapeTool::requires_confirmation() const {
    return false;
}

size_t WebScrapeTool::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string WebScrapeTool::cleanHtml(const std::string& html) const {
    // 1. Remove script and style elements
    std::string text = std::regex_replace(html, std::regex(R"(<(script|style)[^>]*>[\s\S]*?</\1>)"), " ");
    
    // 2. Remove all other HTML tags
    text = std::regex_replace(text, std::regex(R"(<[^>]*>)"), " ");
    
    // 3. Decode some basic entities
    text = std::regex_replace(text, std::regex(R"(&nbsp;)"), " ");
    text = std::regex_replace(text, std::regex(R"(&amp;)"), "&");
    text = std::regex_replace(text, std::regex(R"(&lt;)"), "<");
    text = std::regex_replace(text, std::regex(R"(&gt;)"), ">");

    // 4. Collapse multiple spaces and newlines
    text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
    
    // 5. Trim
    const auto first = text.find_first_not_of(' ');
    if (std::string::npos == first) return "";
    const auto last = text.find_last_not_of(' ');
    return text.substr(first, (last - first + 1));
}

nlohmann::json WebScrapeTool::execute(const nlohmann::json& input) const {
    std::string url = input.at("url");

    const char* mock_env = std::getenv("THOTH_MOCK_SCRAPE");
    if (mock_env && std::string(mock_env) == "true") {
        return {
            {"status", "success"},
            {"data", {
                {"url", url},
                {"content", "Thoth is an experimental AI agent architecture designed for experience-guided reasoning. It uses a parallel execution engine and a hybrid memory model."},
                {"length", 150}
            }},
            {"error_message", nullptr}
        };
    }

    std::string readBuffer;
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "Failed to initialize CURL."}
        };
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "ThothAgent/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // 10 second timeout

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    }

    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "CURL error: " + std::string(curl_easy_strerror(res))}
        };
    }

    if (http_code != 200) {
        return {
            {"status", "error"},
            {"data", nlohmann::json::object()},
            {"error_message", "HTTP error: " + std::to_string(http_code)}
        };
    }

    std::string cleaned = cleanHtml(readBuffer);

    return {
        {"status", "success"},
        {"data", {
            {"url", url},
            {"content", cleaned},
            {"length", cleaned.length()}
        }},
        {"error_message", nullptr}
    };
}
