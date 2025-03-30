#pragma once

#include "fwd.h"
#include <memory>

class Inf
{
public:
    EnvHandle Env;
    IAutoRegressivePipeline* ARPipeline;
    std::unique_ptr<MidiTokenizer> Tokenizer;
    AutoRegressiveBatchHandle Batch2;

    int32_t LineNbMaxToken = 512;
    int32_t NbTokensToGenerate = 300;

    void runInference(const char* folderPath, bool printLogs);
};

class TokenHistoryTest
{
public:
    static void TestTokenHistory(const class TokenHistory& tokenHistory);
};