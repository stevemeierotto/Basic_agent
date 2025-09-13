/*
 * Copyright (c) 2025 Steve Meierotto
 * 
 * basic_agent - AI Agent with Memory and RAG Capabilities
 * uses either Ollama lacal models or OpenAI API
 *
 * Licensed under the MIT License
 * See LICENSE file in the project root for full license text
 */

#pragma once
#include "tools.h"

class WebScraperTools : public Tools {
private:
    std::string apiKey;
    std::string cseId;

    // --- CURL helper ---
    std::string performCurlRequest(const std::string& url, const std::string& userAgent);

public:
    WebScraperTools();
    WebScraperTools(const std::string& apiKey, const std::string& cseId);
    std::string handleScrape(const std::string& query, int results=3, int snippets=3);

    // Implement/override
    std::string summarizeText(const std::string& text, int numSentences = 3) override;

    std::vector<std::string> webSearch(const std::string& query, int maxResults = 5);
    std::string fetchAndExtract(const std::string& url);
    std::string searchAndSummarize(const std::string& query, int numResults = 5, int numSentences = 3);
    bool writeSummary(const std::string& summary, const std::string& filename);
    // --- Starter API Methods ---
    
    // Reddit API: fetch top posts from a subreddit
    std::string fetchRedditPosts(const std::string& subreddit, int limit=5);

    // News API: fetch top articles for a query
    std::string fetchNewsArticles(const std::string& query, int limit=5);

    // YouTube API: fetch video titles and descriptions for a search query
    std::string fetchYouTubeVideos(const std::string& query, int limit=5);

    // Google Custom Search API (already partially used)
    std::string fetchGoogleCSEResults(const std::string& query, int numResults=5);
};
