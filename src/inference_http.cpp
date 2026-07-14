/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — shared HTTP helpers for inference clients (Plan H)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/inference_http.h"

#include <curl/curl.h>

namespace Thoth {

namespace {

size_t writeCallback(char* contents, size_t size, size_t nmemb, void* userp) {
    const size_t total = size * nmemb;
    static_cast<std::string*>(userp)->append(contents, total);
    return total;
}

InferenceHttpResponse performRequest(const std::string& url,
                                     const char* method,
                                     const std::string* body,
                                     long timeout_seconds) {
    InferenceHttpResponse response;
    CURL* curl = curl_easy_init();
    if (!curl) {
        response.error = "CURL initialization failed";
        return response;
    }

    struct curl_slist* headers = curl_slist_append(nullptr, "Content-Type: application/json");
    std::string read_buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);

    if (method && std::string(method) == "POST" && body) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body->c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body->size());
    } else {
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    }

    const CURLcode result = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (result != CURLE_OK) {
        response.error = curl_easy_strerror(result);
        return response;
    }

    response.body = std::move(read_buffer);
    response.ok = response.status_code >= 200 && response.status_code < 300;
    if (!response.ok && response.error.empty()) {
        response.error = "HTTP status " + std::to_string(response.status_code);
    }
    return response;
}

} // namespace

InferenceHttpResponse inferenceHttpGet(const std::string& url, long timeout_seconds) {
    return performRequest(url, "GET", nullptr, timeout_seconds);
}

InferenceHttpResponse inferenceHttpPost(const std::string& url,
                                        const std::string& json_body,
                                        long timeout_seconds) {
    return performRequest(url, "POST", &json_body, timeout_seconds);
}

} // namespace Thoth
