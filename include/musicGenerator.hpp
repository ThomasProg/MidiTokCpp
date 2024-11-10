#pragma once

#include <map>
#include <vector>
#include <list>
#include <onnxruntime_cxx_api.h>
#include "midiTokenizer.hpp"
#include "fwd.h"

struct Batch
{
public:
    using DataType = std::int32_t;

    // Values
    std::vector<DataType> inputIds; // encoded tokens
    std::vector<DataType> attentionMask;
    std::vector<DataType> positionIds;

public:
    void push(DataType inInputId, DataType inMask, DataType inPositionId);
    void push(DataType inInputId);
    void pop();

    void set(const std::vector<DataType>& inTokens, std::int32_t fromPos = 0);

    size_t size() const;
};

struct RunInstance
{
    using DataType = Batch::DataType;

    // Values
    std::vector<Batch*> batches;

    std::int32_t nbCache = 6;
    std::vector<Ort::Value> cachedValues; // "present" / "past" values

    // Model Info & Tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions cacheAllocator;
    std::vector<Ort::Value> inputDataTensors;

public:
    void updateInputTensors(const struct ModelInfo& info);
};

struct ModelInfo
{
    // @TODO : load from config
    int64_t num_attention_heads; // n_head
    int64_t hidden_size; // n_embd
    int64_t num_layer; // n_layer

    // Labels
    std::string inputIdLabel;
    std::string attentionMaskLabel;
    std::string positionIdLabel;
    std::vector<std::string> pastLabels;
};

class MusicGenerator
{
private:
    std::unique_ptr<Ort::Session> session;
public:
    ModelInfo modelInfo;

public:
    static std::unique_ptr<Ort::Env> createOnnxEnv(bool useLogging = false);
    void loadOnnxModel(const Ort::Env& env, const std::string& modelPath);
    void generate(RunInstance& input);

    RunInstance generateInput(std::vector<RunInstance::DataType>&& inputTokens);
};

