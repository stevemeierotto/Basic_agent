#include "../include/gmail_read_labels_tool.h"
#include <iostream>
#include <sstream>

using json = nlohmann::json;

GmailReadLabelsTool::GmailReadLabelsTool() {
    // libcurl initialization is usually global, but we can ensure it here if needed
}

GmailReadLabelsTool::~GmailReadLabelsTool() = default;

json GmailReadLabelsTool::input_schema() const {
    return R"({
        "type": "object",
        "properties": {
            "credential_id": { 
                "type": "string",
                "description": "Resolved OAuth2 access token for Gmail API."
            },
            "max_results": { 
                "type": "integer", 
                "default": 100,
                "minimum": 1,
                "maximum": 500
            },
            "confirmed": { "type": "boolean", "description": "Must be true to proceed." }
        },
        "required": ["credential_id"],
        "additionalProperties": false
    })"_json;
}

bool GmailReadLabelsTool::requires_confirmation() const {
    return true;
}

size_t GmailReadLabelsTool::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

json GmailReadLabelsTool::execute(const json& input) const {
    // 1. Validate mandatory credential_id presence (runtime usually does this)
    if (!input.contains("credential_id") || !input["credential_id"].is_string()) {
        return {
            {"status", "error"},
            {"data", json::object()},
            {"error_message", "Missing or invalid credential_id in input."}
        };
    }

    std::string accessToken = input["credential_id"].get<std::string>();
    int maxResults = input.value("max_results", 100);

    // 2. Prepare libcurl request
    CURL* curl = curl_easy_init();
    if (!curl) {
        return {
            {"status", "error"},
            {"data", json::object()},
            {"error_message", "Failed to initialize CURL."}
        };
    }

    std::string readBuffer;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string authHeader = "Authorization: Bearer " + accessToken;
    headers = curl_slist_append(headers, authHeader.c_str());

    std::string url = "https://gmail.googleapis.com/gmail/v1/users/me/labels";
    // We could append maxResults if the API supported it on this endpoint, 
    // but users/me/labels returns all labels usually.

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // 10 second timeout

    // 3. Perform the request
    CURLcode res = curl_easy_perform(curl);
    
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    // Clean up CURL immediately
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // 4. Handle CURL and HTTP errors
    if (res != CURLE_OK) {
        return {
            {"status", "error"},
            {"data", json::object()},
            {"error_message", std::string("CURL request failed: ") + curl_easy_strerror(res)}
        };
    }

    if (httpCode != 200) {
        std::string errMsg = "Gmail API returned HTTP " + std::to_string(httpCode);
        try {
            auto errorJson = json::parse(readBuffer);
            if (errorJson.contains("error") && errorJson["error"].contains("message")) {
                errMsg += ": " + errorJson["error"]["message"].get<std::string>();
            }
        } catch (...) {
            // If body is not JSON, use the raw response if short
            if (!readBuffer.empty() && readBuffer.size() < 100) {
                errMsg += " - " + readBuffer;
            }
        }

        return {
            {"status", "error"},
            {"data", json::object()},
            {"error_message", errMsg}
        };
    }

    // 5. Parse and return success
    try {
        auto resultJson = json::parse(readBuffer);
        
        // Gmail labels response is { "labels": [ { "id": "...", "name": "...", ... }, ... ] }
        if (!resultJson.contains("labels") || !resultJson["labels"].is_array()) {
            return {
                {"status", "error"},
                {"data", json::object()},
                {"error_message", "Unexpected Gmail API response format."}
            };
        }

        // Respect max_results if requested (though labels list is usually small)
        json labels = resultJson["labels"];
        if (labels.size() > static_cast<size_t>(maxResults)) {
            json truncated;
            for (int i = 0; i < maxResults; ++i) {
                truncated.push_back(labels[i]);
            }
            labels = truncated;
        }

        return {
            {"status", "success"},
            {"data", {{"labels", labels}}},
            {"error_message", nullptr}
        };

    } catch (const std::exception& e) {
        return {
            {"status", "error"},
            {"data", json::object()},
            {"error_message", std::string("Failed to parse Gmail response: ") + e.what()}
        };
    }
}
