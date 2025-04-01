#pragma once

#include "fwd.h"
#include <memory>
#include "range.hpp"

class Inf
{
public:
    EnvHandle Env;
    IAutoRegressivePipeline* ARPipeline;
    std::unique_ptr<MidiTokenizer> Tokenizer;
    AutoRegressiveBatchHandle Batch2;

    RangeGroup SearchedRangeGroup;

    int32_t LineNbMaxToken = 512;
    int32_t NbTokensToGenerate = 300;

    std::vector<int32_t> EncodedTokens;

    void load(const char* folderPath, bool printLogs);
    void runInference();
};

class TokenHistoryTest
{
public:
    static void TestTokenHistory(const class TokenHistory& tokenHistory);
};