// src/similarity.cpp
#include "../include/similarity.h"
#include <cmath>
#include <algorithm>
#include <unordered_set>

// Cosine
float CosineSimilarity::operator()(const std::vector<float>& a,
                                   const std::vector<float>& b) const {
    if (a.empty() || b.empty()) return 0.0f;
    size_t len = std::min(a.size(), b.size());

    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < len; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        normA += static_cast<double>(a[i]) * a[i];
        normB += static_cast<double>(b[i]) * b[i];
    }

    if (normA == 0.0 || normB == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(normA) * std::sqrt(normB)));
}

// Euclidean
float EuclideanSimilarity::operator()(const std::vector<float>& a,
                                      const std::vector<float>& b) const {
    if (a.empty() || b.empty()) return 0.0f;
    size_t len = std::min(a.size(), b.size());

    double sumSq = 0.0;
    for (size_t i = 0; i < len; ++i) {
        double diff = static_cast<double>(a[i]) - b[i];
        sumSq += diff * diff;
    }
    return 1.0f / (1.0f + std::sqrt(sumSq)); // normalized similarity
}

// Dot Product
float DotProductSimilarity::operator()(const std::vector<float>& a,
                                       const std::vector<float>& b) const {
    if (a.empty() || b.empty()) return 0.0f;
    size_t len = std::min(a.size(), b.size());

    double dot = 0.0;
    for (size_t i = 0; i < len; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(dot);
}

// Jaccard (treats nonzero entries as set membership)
float JaccardSimilarity::operator()(const std::vector<float>& a,
                                    const std::vector<float>& b) const {
    if (a.empty() || b.empty()) return 0.0f;
    size_t len = std::min(a.size(), b.size());

    size_t intersection = 0, unionCount = 0;
    for (size_t i = 0; i < len; ++i) {
        bool inA = (a[i] != 0.0f);
        bool inB = (b[i] != 0.0f);
        if (inA || inB) {
            unionCount++;
            if (inA && inB) intersection++;
        }
    }
    return (unionCount > 0) ? static_cast<float>(intersection) / unionCount : 0.0f;
}

