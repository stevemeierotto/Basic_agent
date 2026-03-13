#pragma once

#include "itool.h"
#include <algorithm>
#include <sstream>
#include <vector>

/**
 * @brief Tool for summarizing text.
 * Implemented according to TOOLS.md v1.0
 */
class SummarizeTextTool : public ITool {
public:
    std::string name() const override { return "summarize_text"; }
    
    std::string description() const override { 
        return "Summarize input text into a short set of key sentences. Use when a concise summary is needed."; 
    }

    nlohmann::json input_schema() const override {
        return R"({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "num_sentences": { "type": "integer", "minimum": 1, "maximum": 20 }
            },
            "required": ["text"],
            "additionalProperties": false
        })"_json;
    }

    bool requires_confirmation() const override { return false; }

    nlohmann::json execute(const nlohmann::json& input) const override {
        try {
            if (!input.contains("text") || !input["text"].is_string()) {
                return {
                    {"status", "error"},
                    {"data", nlohmann::json::object()},
                    {"error_message", "Field 'text' is required and must be a string."}
                };
            }

            int numSentences = input.value("num_sentences", 3);
            std::string text = input["text"].get<std::string>();

            auto sentences = splitSentences(text);
            if (sentences.empty()) {
                return {
                    {"status", "success"},
                    {"data", {
                        {"summary", ""},
                        {"used_sentences", 0},
                        {"source_sentences", 0}
                    }},
                    {"error_message", nullptr}
                };
            }

            int used = std::min<int>(numSentences, static_cast<int>(sentences.size()));
            std::ostringstream summary;
            for (int i = 0; i < used; ++i) {
                if (i > 0) summary << " ";
                summary << sentences[i];
            }

            return {
                {"status", "success"},
                {"data", {
                    {"summary", summary.str()},
                    {"used_sentences", used},
                    {"source_sentences", static_cast<int>(sentences.size())}
                }},
                {"error_message", nullptr}
            };
        } catch (const std::exception& e) {
            return {
                {"status", "error"},
                {"data", nlohmann::json::object()},
                {"error_message", std::string("Internal error: ") + e.what()}
            };
        }
    }

private:
    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        auto end = s.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return s.substr(start, end - start + 1);
    }

    static std::vector<std::string> splitSentences(const std::string& text) {
        std::vector<std::string> sentences;
        std::string current;
        for (char c : text) {
            current += c;
            if (c == '.' || c == '!' || c == '?') {
                std::string s = trim(current);
                if (!s.empty()) sentences.push_back(s);
                current.clear();
            }
        }
        std::string s = trim(current);
        if (!s.empty()) sentences.push_back(s);
        return sentences;
    }
};
