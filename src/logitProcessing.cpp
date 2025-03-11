#include "logitProcessing.h"
#include "note.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <random>

// numerically stable softmax ; reduces overflows, but costs slightly more
// substract by max logit
// low logits will become 0
// high logits will have better precision
void stableSoftmaxRange(const SearchArgs& args, const Range* ranges, size_t nbRanges)
{
    for (int32_t b = 0; b < args.nbBatches; ++b)
    {
        float* batchLogits = args.logitsTensor + b * args.vocabSize;
        int batchLogitsSize = args.vocabSize;
        float sum = 0;

        float maxLogit = *std::max_element(batchLogits, batchLogits + batchLogitsSize);

        for (int32_t rangeIndex = 0; rangeIndex < nbRanges; rangeIndex++)
        {
            for (int32_t token = ranges[rangeIndex].min; token <= ranges[rangeIndex].max; token++)
            {
                float currentLogit = batchLogits[token];

                currentLogit = std::exp(currentLogit - maxLogit);
                sum += currentLogit;

                batchLogits[token] = currentLogit;
            }
        }
        
        for (int32_t rangeIndex = 0; rangeIndex < nbRanges; rangeIndex++)
        {
            for (int32_t token = ranges[rangeIndex].min; token <= ranges[rangeIndex].max; token++)
            {
                batchLogits[token] /= sum;
            }
        }
    }
}

void stableSoftmaxRange(const SearchArgs* args, const Range* ranges, size_t nbRanges)
{
    stableSoftmaxRange(*args, ranges, nbRanges);
}

void softmaxRange(const SearchArgs& args, const Range* ranges, size_t nbRanges)
{
    for (int32_t b = 0; b < args.nbBatches; ++b)
    {
        float* batchLogits = args.logitsTensor + b * args.vocabSize;
        float sum = 0;

        for (int32_t rangeIndex = 0; rangeIndex < nbRanges; rangeIndex++)
        {
            for (int32_t token = ranges[rangeIndex].min; token <= ranges[rangeIndex].max; token++)
            {
                float currentLogit = batchLogits[token];

                currentLogit = std::exp(currentLogit);
                sum += currentLogit;

                batchLogits[token] = currentLogit;
            }
        }

        for (int32_t rangeIndex = 0; rangeIndex < nbRanges; rangeIndex++)
        {
            for (int32_t token = ranges[rangeIndex].min; token <= ranges[rangeIndex].max; token++)
            {
                batchLogits[token] /= sum;
            }
        }
    }
}

void softmaxRange(const SearchArgs* args, const Range* ranges, size_t nbRanges)
{
    softmaxRange(*args, ranges, nbRanges);
}

void stableSoftmax(float* logits, int32_t* indicesBegin, int32_t* indicesEnd)
{
    float sum = 0;

    int32_t maxLogitIndex = *std::max_element(indicesBegin, indicesEnd, [logits](int32_t a, int32_t b)
    {
        return logits[a] < logits[b];
    });
    float maxLogit = logits[maxLogitIndex];

    for (int32_t* indicesIt = indicesBegin; indicesIt < indicesEnd; ++indicesIt)
    {
        const int32_t token = *indicesIt;

        float currentLogit = logits[token];

        currentLogit = std::exp(currentLogit - maxLogit);
        sum += currentLogit;

        logits[token] = currentLogit;
    }

    for (int32_t* indicesIt = indicesBegin; indicesIt < indicesEnd; ++indicesIt)
    {
        const int32_t token = *indicesIt;
        logits[token] /= sum;
    }
}
void softmax(float* logits, int32_t* indicesBegin, int32_t* indicesEnd)
{
    float sum = 0;

    for (int32_t* indicesIt = indicesBegin; indicesIt < indicesEnd; ++indicesIt)
    {
        const int32_t token = *indicesIt;

        float currentLogit = logits[token];

        currentLogit = std::exp(currentLogit);
        sum += currentLogit;

        logits[token] = currentLogit;
    }

    for (int32_t* indicesIt = indicesBegin; indicesIt < indicesEnd; ++indicesIt)
    {
        const int32_t token = *indicesIt;
        logits[token] /= sum;
    }
}

void sortLogits(float* logits, int32_t* indicesStart, int32_t* indicesEnd, int32_t nbLogitsToSort)
{
    auto comp = [logits](int32_t i, int32_t j) { return logits[i] > logits[j]; };
    std::nth_element(indicesStart, indicesStart + nbLogitsToSort, indicesEnd, comp);
    std::sort(indicesStart, indicesStart + nbLogitsToSort, comp);
}

class WeightedContainer
{
private:
    float* weights;
    int32_t* indicesStart;
    int32_t* indicesEnd;

public:
    explicit WeightedContainer(float* inWeights, int32_t* inIndicesStart, int32_t* inIndicesEnd) 
        : weights(inWeights), indicesStart(inIndicesStart), indicesEnd(inIndicesEnd) {}

    // Custom iterator that only iterates over weights
    class WeightIterator
    {
    private:
        float* weights;
        int32_t* it;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = float;
        using difference_type = std::ptrdiff_t;
        using pointer = const float*;
        using reference = const float&;

        explicit WeightIterator(float* inWeights, int32_t* inIt) : weights(inWeights), it(inIt) {}

        reference operator*() const { return weights[*it]; }
        WeightIterator& operator++() { ++it; return *this; }
        bool operator!=(const WeightIterator& other) const { return it != other.it; }
    };

    WeightIterator begin() { return WeightIterator(weights, indicesStart); }
    WeightIterator end() { return WeightIterator(weights, indicesEnd); }
};

std::mt19937& getRandGenerator()
{
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    return gen;
}

int32_t randomSampling(float* logits, int32_t* indicesStart, int32_t* indicesEnd)
{
    std::uniform_int_distribution<int32_t> distribution(*indicesStart, *indicesEnd);
    return distribution(getRandGenerator());
}

int32_t topKSampling(float* logits, int32_t* indicesStart, int32_t* indicesEnd)
{
    // std::discrete_distribution doesn't need normalized probabilities
    WeightedContainer wrappedLogits(logits, indicesStart, indicesEnd);
    std::discrete_distribution<int32_t> distribution(wrappedLogits.begin(), wrappedLogits.end());
    return indicesStart[distribution(getRandGenerator())];
}

int32_t* topPSamplingFindCutoffIt(float* logits, int32_t* indicesStart, int32_t* indicesEnd, float cutoff)
{
    // get truncating index
    float cumulativeProb = 0.0;
    int32_t* cutoffIndexIt = indicesEnd;
    for (int32_t* it = indicesStart; it != indicesEnd; ++it)
    {
        const int32_t token = *it;
        cumulativeProb += logits[token];
        if (cumulativeProb >= cutoff) 
        {
            ++it;
            cutoffIndexIt = it;
            break; 
        }
    }
    return cutoffIndexIt;
}

int32_t topPSampling(float* logits, int32_t* indicesStart, int32_t* indicesEnd, float cutoff)
{
    int32_t* cutoffIndexIt = topPSamplingFindCutoffIt(logits, indicesStart, indicesEnd, cutoff);
    return topKSampling(logits, indicesStart, cutoffIndexIt);
}

