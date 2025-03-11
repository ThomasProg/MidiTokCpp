#pragma once

#include "fwd.h"
#include "range.h"

extern "C" 
{
    API_EXPORT int32_t greedySearch(const SearchArgs* args, const Range* ranges, size_t nbRanges);

    API_EXPORT void stableSoftmaxRange(const SearchArgs* args, const Range* ranges, size_t nbRanges);
    API_EXPORT void softmaxRange(const SearchArgs* args, const Range* ranges, size_t nbRanges);

    API_EXPORT void stableSoftmax(float* logits, int32_t* indicesBegin, int32_t* indicesEnd);
    API_EXPORT void softmax(float* logits, int32_t* indicesBegin, int32_t* indicesEnd);

    API_EXPORT void sortLogits(float* logits, int32_t* indicesStart, int32_t* indicesEnd, int32_t nbLogitsToSort);

    // Stochastic Sampling: requires normalization (softmax) before
    API_EXPORT int32_t randomSampling(float* logits, int32_t* indicesStart, int32_t* indicesEnd);

    // logits should be sorted
    API_EXPORT int32_t topKSampling(float* logits, int32_t* indicesStart, int32_t* indicesEnd);
    API_EXPORT int32_t* topPSamplingFindCutoffIt(float* logits, int32_t* indicesStart, int32_t* indicesEnd, float cutoff);
    API_EXPORT int32_t topPSampling(float* logits, int32_t* indicesStart, int32_t* indicesEnd, float cutoff);

    API_EXPORT void temperatureTransform(const SearchArgs* args, const Range* ranges, size_t nbRanges);

}

