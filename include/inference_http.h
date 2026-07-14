/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — shared HTTP helpers for inference clients (Plan H)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#pragma once

#include <string>

namespace Thoth {

struct InferenceHttpResponse {
    bool ok = false;
    long status_code = 0;
    std::string body;
    std::string error;
};

InferenceHttpResponse inferenceHttpGet(const std::string& url, long timeout_seconds = 10);
InferenceHttpResponse inferenceHttpPost(const std::string& url,
                                        const std::string& json_body,
                                        long timeout_seconds = 600);

} // namespace Thoth
