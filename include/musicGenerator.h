#pragma once

#include <map>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "midiTokenizer.hpp"

struct Input
{
    using DataType = int32_t;

    std::vector<std::string> inputNames;

    // [type, value]
    std::vector<std::vector<DataType>> inputData;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputDataTensors;
};

class MusicGenerator
{
public:
    std::unique_ptr<Ort::Session> session;



    // @TODO : load from config
    int64_t num_attention_heads = 8;
    int64_t hidden_size = 512;
    int64_t num_layer = 8;

    int64_t batchSize = 1;




public:
    static std::unique_ptr<Ort::Env> createOnnxEnv(bool useLogging = false);
    void loadOnnxModel(const Ort::Env& env, const std::string& modelPath);
    void generate(Input& input);

    Input generateInput(std::vector<Input::DataType>&& inputTokens);

    static void updateInputTensors(Input& input);
};

