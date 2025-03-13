#pragma once

#include "logitProcessing.h"

API_EXPORT void stableSoftmaxRange(const SearchArgs& args, const Range* ranges, size_t nbRanges);
API_EXPORT void softmaxRange(const SearchArgs& args, const Range* ranges, size_t nbRanges);

API_EXPORT void specialPenaltyTransform(float* logits, const Range* ranges, size_t nbRanges, GenerationHistory* history, const SpecialPenaltyTransformArgs& args);