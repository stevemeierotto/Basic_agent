// include/similarity.h
#ifndef SIMILARITY_H
#define SIMILARITY_H

#include <vector>

class ISimilarity {
public:
    virtual ~ISimilarity() = default;
    virtual float operator()(const std::vector<float>& a,
                             const std::vector<float>& b) const = 0;
};

class CosineSimilarity : public ISimilarity {
public:
    float operator()(const std::vector<float>& a,
                     const std::vector<float>& b) const override;
};

class EuclideanSimilarity : public ISimilarity {
public:
    float operator()(const std::vector<float>& a,
                     const std::vector<float>& b) const override;
};

class DotProductSimilarity : public ISimilarity {
public:
    float operator()(const std::vector<float>& a,
                     const std::vector<float>& b) const override;
};

class JaccardSimilarity : public ISimilarity {
public:
    float operator()(const std::vector<float>& a,
                     const std::vector<float>& b) const override;
};

#endif // SIMILARITY_H

