/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — C2 Phase 2 conversational retrieval boosts
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/chat_retrieval_boost.h"
#include "../include/chat_retrieval_config.h"
#include "../include/index_manager.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace Thoth {
namespace ChatRetrieval {

namespace {

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string fileBasename(const std::string& path) {
    try {
        return fs::path(path).filename().string();
    } catch (...) {
        return path;
    }
}

std::string stemBasename(const std::string& path) {
    try {
        return fs::path(path).stem().string();
    } catch (...) {
        return path;
    }
}

bool isStopWord(const std::string& token) {
    static const std::unordered_set<std::string> kStop = {
        "a", "an", "the", "is", "are", "what", "how", "do", "i", "use", "me", "tell",
        "about", "explain", "describe", "define", "quote", "first", "sentence", "of",
        "in", "to", "for", "and", "or", "be", "does", "mean", "please", "can", "you",
    };
    return kStop.count(lowerCopy(token)) > 0;
}

void syncBreakdownOrder(GragDiagnostics& diagnostics,
                        const std::vector<std::pair<CodeChunk, float>>& ranked) {
    std::unordered_map<std::string, ScoreBreakdown> byCode;
    byCode.reserve(diagnostics.breakdowns.size());
    for (const auto& breakdown : diagnostics.breakdowns) {
        byCode[breakdown.code_text] = breakdown;
    }

    diagnostics.breakdowns.clear();
    diagnostics.final_scores.clear();
    for (const auto& [chunk, score] : ranked) {
        ScoreBreakdown sb;
        auto it = byCode.find(chunk.code);
        if (it != byCode.end()) {
            sb = it->second;
            sb.final_score = score;
        } else {
            sb.file_name = chunk.fileName;
            sb.symbol = chunk.symbolName;
            sb.code_text = chunk.code;
            sb.final_score = score;
        }
        diagnostics.breakdowns.push_back(sb);
        diagnostics.final_scores.push_back(score);
    }
}

} // namespace

bool isDefinitionalQuery(const std::string& query) {
    const std::string q = lowerCopy(query);
    return q.find("explain") != std::string::npos || q.find("what is") != std::string::npos ||
           q.find("what are") != std::string::npos || q.find("define") != std::string::npos ||
           q.find("describe") != std::string::npos || q.find("quote") != std::string::npos;
}

std::vector<std::string> extractFilenameTokens(const std::string& query) {
    std::vector<std::string> tokens;

    static const std::regex mdFile(R"(([A-Za-z0-9_-]+)\.md\b)", std::regex::icase);
    for (auto it = std::sregex_iterator(query.begin(), query.end(), mdFile);
         it != std::sregex_iterator(); ++it) {
        tokens.push_back((*it)[1].str());
    }

    std::string normalized;
    normalized.reserve(query.size());
    for (char c : query) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            normalized.push_back(c);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            normalized.push_back(' ');
        }
    }

    std::istringstream stream(normalized);
    std::string token;
    while (stream >> token) {
        if (token.size() < 2 || isStopWord(token)) {
            continue;
        }
        tokens.push_back(token);
    }

    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

bool isQuoteQuery(const std::string& query) {
    const std::string q = lowerCopy(query);
    return q.find("quote") != std::string::npos || q.find("first sentence") != std::string::npos;
}

bool isUsageQuery(const std::string& query) {
    const std::string q = lowerCopy(query);
    const bool asksHow = q.find("how do i") != std::string::npos || q.find("how to") != std::string::npos;
    return asksHow && q.find("use") != std::string::npos;
}

void ensureFilenameCoverage(IndexManager* indexManager,
                            const std::vector<std::string>& tokens,
                            const std::string& query,
                            std::vector<std::pair<CodeChunk, float>>& ragResults,
                            int minPerFile) {
    if (!indexManager || tokens.empty() || minPerFile <= 0) {
        return;
    }

    const bool quoteQuery = isQuoteQuery(query);
    const auto& allChunks = indexManager->getChunks();
    std::unordered_map<std::string, std::vector<const CodeChunk*>> matchedByStem;

    for (const auto& chunk : allChunks) {
        for (const auto& token : tokens) {
            if (filenameMatchesToken(chunk.fileName, token)) {
                matchedByStem[lowerCopy(stemBasename(chunk.fileName))].push_back(&chunk);
                break;
            }
        }
    }

    for (const auto& [stem, fileChunks] : matchedByStem) {
        auto sorted = fileChunks;
        if (quoteQuery) {
            std::sort(sorted.begin(), sorted.end(), [](const CodeChunk* a, const CodeChunk* b) {
                if (a->startLine != b->startLine) {
                    return a->startLine < b->startLine;
                }
                return a->code.size() > b->code.size();
            });
        } else {
            std::sort(sorted.begin(), sorted.end(), [](const CodeChunk* a, const CodeChunk* b) {
                return a->code.size() > b->code.size();
            });
        }

        int added = 0;
        for (const CodeChunk* candidate : sorted) {
            if (candidate->code.size() < kMinChunkChars) {
                continue;
            }
            const bool alreadyPresent = std::any_of(
                ragResults.begin(), ragResults.end(),
                [&](const std::pair<CodeChunk, float>& entry) { return entry.first.code == candidate->code; });
            if (alreadyPresent) {
                continue;
            }
            ragResults.push_back({*candidate, 0.55f});
            if (++added >= minPerFile) {
                break;
            }
        }
    }
}

bool filenameMatchesToken(const std::string& filePath, const std::string& token) {
    const std::string stem = lowerCopy(stemBasename(filePath));
    const std::string needle = lowerCopy(token);
    if (stem.empty() || needle.empty()) {
        return false;
    }
    return stem == needle || stem.find(needle) != std::string::npos ||
           needle.find(stem) != std::string::npos;
}

void applyConversationalBoosts(std::vector<std::pair<CodeChunk, float>>& ranked,
                               const std::string& query,
                               GragDiagnostics& diagnostics) {
    if (ranked.empty()) {
        return;
    }

    const auto tokens = extractFilenameTokens(query);
    const bool definitional = isDefinitionalQuery(query);

    for (auto& [chunk, score] : ranked) {
        if (chunk.code.size() < kMinChunkChars) {
            score *= kTinyChunkScoreFactor;
        }

        for (const auto& token : tokens) {
            if (filenameMatchesToken(chunk.fileName, token)) {
                score += kFilenameMatchBoost;
                break;
            }
        }

        if (definitional && chunk.code.size() >= kSubstantiveChunkChars) {
            score += kSubstantiveChunkBoost;
        }

        if (isUsageQuery(query) && lowerCopy(stemBasename(chunk.fileName)) == "howto") {
            score += kUsageDocBoost;
        }
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    syncBreakdownOrder(diagnostics, ranked);
    diagnostics.chunks_reranked = static_cast<int>(ranked.size());
}

std::vector<std::pair<CodeChunk, float>> selectTopKForInjection(
    const std::vector<std::pair<CodeChunk, float>>& ranked,
    int topK,
    std::size_t minChunkChars,
    GragDiagnostics& diagnostics) {
    if (topK <= 0) {
        return {};
    }

    std::vector<std::pair<CodeChunk, float>> selected;
    selected.reserve(static_cast<std::size_t>(topK));

    for (const auto& entry : ranked) {
        if (entry.first.code.size() < minChunkChars) {
            continue;
        }
        selected.push_back(entry);
        if (static_cast<int>(selected.size()) >= topK) {
            break;
        }
    }

    if (static_cast<int>(selected.size()) < topK) {
        for (const auto& entry : ranked) {
            if (static_cast<int>(selected.size()) >= topK) {
                break;
            }
            if (entry.first.code.size() >= minChunkChars) {
                continue;
            }
            selected.push_back(entry);
        }
    }

    syncBreakdownOrder(diagnostics, selected);
    diagnostics.chunks_retrieved = static_cast<int>(selected.size());
    return selected;
}

std::string formatChunkForPrompt(const CodeChunk& chunk) {
    std::ostringstream oss;
    oss << "Document: " << fileBasename(chunk.fileName) << '\n';
    if (chunk.startLine > 0) {
        oss << "Lines: " << chunk.startLine;
        if (chunk.endLine > chunk.startLine) {
            oss << '-' << chunk.endLine;
        }
        oss << '\n';
    }
    oss << chunk.code;
    return oss.str();
}

} // namespace ChatRetrieval
} // namespace Thoth
