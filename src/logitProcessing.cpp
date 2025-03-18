#include "logitProcessing.h"
#include "logitProcessing.hpp"
#include "note.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <random>
#include <cassert>
#include "midiTokenizer.hpp"
#include "generationHistory.hpp"

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

void temperatureTransform(float* logits, const Range* ranges, size_t nbRanges, float temperature)
{
    for (int32_t rangeIndex = 0; rangeIndex < nbRanges; rangeIndex++)
    {
        for (int32_t token = ranges[rangeIndex].min; token <= ranges[rangeIndex].max; token++)
        {
            logits[token] /= temperature;
        }
    }
}

template<typename F>
inline void customPenaltyTransformTemplated(float* logits, const Range* ranges, size_t nbRanges, F&& penaltyFunctor)
{
    for (int32_t rangeIndex = 0; rangeIndex < nbRanges; rangeIndex++)
    {
        for (int32_t token = ranges[rangeIndex].min; token <= ranges[rangeIndex].max; token++)
        {
            float penalty;
            if (penaltyFunctor(token, &penalty))
            {
                const float logit = logits[token];
                if (logit > 0.0)
                {
                    logits[token] /= penalty;
                }
                else
                {
                    logits[token] *= penalty;
                }
            }
        }
    }
}

void customPenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, const void* data, bool (*penaltyFunctor)(const void* data, const int32_t token, float* outPenalty))
{
    customPenaltyTransformTemplated(logits, ranges, nbRanges, [data, penaltyFunctor](const int32_t token, float* outPenalty) -> bool
    {
        return penaltyFunctor(data, token, outPenalty);
    });
}

void repetitionPenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, float penalty, GenerationHistory* history, int32_t maxAge)
{
    auto penaltyFunctor = [penalty, maxAge, history](const int32_t token, float* outPenalty) -> bool
    {
        if (!history->hadDecodedTokenRecently(token, 250))
        {
            return false;
        }

        *outPenalty = penalty;
        return true;
    };

    customPenaltyTransformTemplated(logits, ranges, nbRanges, penaltyFunctor);
}

void specialPenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, GenerationHistory* history, const SpecialPenaltyTransformArgs* args)
{
    assert(args != nullptr);
    specialPenaltyTransform(logits, ranges, nbRanges, history, *args);
}

void specialPenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, GenerationHistory* history, const SpecialPenaltyTransformArgs& args)
{
    auto penaltyFunctor = [history, &args](const int32_t token, float* outPenalty) -> bool
    {
        *outPenalty = 1.0;

        const MidiTokenizer& tokenizer = history->getTokenizer();
        const int32_t* decodedTokensBegin;
        const int32_t* decodedTokensEnd;
        tokenizer.decodeTokenFast(token, decodedTokensBegin, decodedTokensEnd);

        for (const int32_t* it = decodedTokensBegin; it != decodedTokensEnd; ++it)
        {
            const int32_t decodedToken = *it;
            int32_t age;
            if (tokenizer.isPitchFast(decodedToken) && history->getDecodedTokensHistory().findMostRecentAge(token, age))
            {
                *outPenalty += args.pitchMaxAdditivePenalty * (1.f - float(age) / args.pitchWindowSize);
            }
        }

        return true;
    };

    customPenaltyTransformTemplated(logits, ranges, nbRanges, penaltyFunctor);
}

void musicalScalePenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, const int32_t* pitches, int32_t nbPitches, float penaltyPerOutOfScalePitch, MidiTokenizerHandle tokenizer)
{
    constexpr int32_t nbPitchesPerOctave = 12;
    assert(nbPitches > 0 && nbPitches <= nbPitchesPerOctave);
    assert((pitches + nbPitches) == std::find_if(pitches, pitches+nbPitches, [nbPitchesPerOctave](int32_t pitch) 
        { return pitch > nbPitchesPerOctave; }));

    auto penaltyFunctor = [pitches, nbPitches, nbPitchesPerOctave, penaltyPerOutOfScalePitch, tokenizer](const int32_t token, float* outPenalty) -> bool
    {
        *outPenalty = 1.0;

        const int32_t* decodedTokensBegin;
        const int32_t* decodedTokensEnd;
        tokenizer->decodeTokenFast(token, decodedTokensBegin, decodedTokensEnd);

        for (const int32_t* it = decodedTokensBegin; it != decodedTokensEnd; ++it)
        {
            const int32_t decodedToken = *it;
            if (tokenizer->isPitchFast(decodedToken))
            {
                const int32_t pitch = tokenizer->getPitchValueFast(decodedToken) % nbPitchesPerOctave;
                const int32_t* pitchesEnd = pitches+nbPitches;
                // linear search faster than dichotomy for small arrays (and nbPitches <= 12)
                if (std::find(pitches, pitchesEnd, pitch) == pitchesEnd) // if not in scale, then penalty
                {
                    *outPenalty += penaltyPerOutOfScalePitch;
                }
            }
        }

        return true;
    };

    customPenaltyTransformTemplated(logits, ranges, nbRanges, penaltyFunctor);
}

void pitchRangePenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, const int32_t minPitch, const int32_t maxPitch, float penaltyPerOutOfRangePitch, MidiTokenizerHandle tokenizer)
{
    auto penaltyFunctor = [tokenizer, minPitch, maxPitch, penaltyPerOutOfRangePitch](const int32_t token, float* outPenalty) -> bool
    {
        *outPenalty = 1.0;

        const int32_t* decodedTokensBegin;
        const int32_t* decodedTokensEnd;
        tokenizer->decodeTokenFast(token, decodedTokensBegin, decodedTokensEnd);

        for (const int32_t* it = decodedTokensBegin; it != decodedTokensEnd; ++it)
        {
            const int32_t decodedToken = *it;
            if (tokenizer->isPitchFast(decodedToken))
            {
                const int32_t pitch = tokenizer->getPitchValueFast(decodedToken);
                if (pitch < minPitch || pitch > maxPitch)
                {
                    *outPenalty += penaltyPerOutOfRangePitch;
                }
            }
        }

        return true;
    };

    customPenaltyTransformTemplated(logits, ranges, nbRanges, penaltyFunctor);
}

constexpr int32_t ionianCMajor[] = {60%12, 62%12, 64%12, 65%12, 67%12, 69%12, 71%12, 72%12};
constexpr int32_t aeolianCNatural[] = {60%12, 62%12, 63%12, 65%12, 67%12, 68%12, 70%12, 72%12};
constexpr int32_t harmonicCMinor[] = {60%12, 62%12, 63%12, 65%12, 67%12, 68%12, 71%12, 72%12};
constexpr int32_t ascendingMelodicCMinor[] = {60%12, 62%12, 63%12, 65%12, 67%12, 69%12, 71%12, 72%12};

template<typename T, int32_t N>
constexpr std::array<T, N>& modArray(std::array<int, 8> arr, int32_t modValue)
{
    for (std::size_t i = 0; i < arr.size(); ++i) 
    {
        arr[i] = i % modValue;
    }
    return arr;
}

namespace Scales::Ionian::CMajor
{
constexpr std::array<int, 8> IonianCMajor() 
{
    constexpr std::array<int, 8> arr = {60, 62, 64, 65, 67, 69, 71, 72};
    return modArray<int, 8>(arr, 12);
}

constexpr const int32_t* get()
{
    return IonianCMajor().data();
}
constexpr int32_t size()
{
    return int32_t(IonianCMajor().size());
}
}