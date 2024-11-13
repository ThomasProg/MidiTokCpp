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

    DataType lastGeneratedToken = 0;

public:
    void push(DataType inInputId, DataType inMask, DataType inPositionId);
    void push(DataType inInputId);
    void pop();

    void set(const std::vector<DataType>& inTokens, std::int32_t fromPos = 0);

    size_t size() const;
};

struct ModelInfo;

struct RunInstance
{
    using DataType = Batch::DataType;

    // Values
    std::vector<Batch*> batches;

    // Model Info & Tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions defaultAllocator;




    Ort::IoBinding io_binding = Ort::IoBinding(nullptr);

    // Inputs
    Ort::Value inputIdsTensor = Ort::Value(nullptr);
    Ort::Value positionIdsTensor = Ort::Value(nullptr);
    Ort::Value attentionMaskTensor = Ort::Value(nullptr);
    std::vector<Ort::Value> presentTensors; // cache

    // Outputs
    Ort::Value logitsTensor = Ort::Value(nullptr);
    std::vector<Ort::Value> pastTensors; // cache

    std::int64_t seqLength = 0;
    std::int64_t maxInputLength = 255; // ideally, the amount of tokens the model can process 
    std::int64_t subsequentGenerationIndex = 0;

public:
    void updateInputIdsTensor(const ModelInfo& info);
    void updatePositionIdsTensor(const ModelInfo& info);
    void updateAttentionMaskTensor(const ModelInfo& info);

    void updateAttentionMaskTensorCache(const ModelInfo& info, std::int64_t seqLength);
    void updateInputIdsTensorCache(const ModelInfo& info, const std::vector<RunInstance::DataType>& nextIds);

    // void updateInputTensors(const ModelInfo& info);

    void createInputIdsTensor(const ModelInfo& info);
    void createInputIdsTensorCache(const ModelInfo& info);
    void createPositionIdsTensor(const ModelInfo& info);
    void createPositionIdsTensorCache(const ModelInfo& info);
    void createAttentionMaskTensor(const ModelInfo& info, std::int64_t seqLength);
    void createPresentTensors(const ModelInfo& info, std::int64_t seqLength = 0); // if 0, isn't using present tensors

    void createLogitsTensor(const ModelInfo& info, std::int64_t seqLength);
    void createPastTensors(const ModelInfo& info, std::int64_t seqLength);

    // void createInputTensors(const ModelInfo& info, std::int64_t seqLength);
    // void createOutputTensors(const ModelInfo& info, std::int64_t seqLength);
    void bindInputs(const ModelInfo& modelInfo);
    void bindOutputs(const ModelInfo& modelInfo);
    void bind(const ModelInfo& modelInfo);

    std::int64_t getNbBatches() const
    {
        return std::int64_t(batches.size());
    }

    void reset();

    void copyAndShiftPresentIntoNextPast(const float* presentData, float* pastData, int64_t presentShape[], int64_t pastShape[]);

    void printInputTensors();
    void printOutputTensors();
};

struct ModelInfo
{
    // @TODO : load from config
    int64_t num_attention_heads; // n_head
    int64_t hidden_size; // n_embd
    int64_t num_layer; // n_layer
    int64_t vocab_size; // vocab size
    int64_t nbMaxPositions; // n_positions

    // Input Labels
    std::string inputIdLabel;
    std::string attentionMaskLabel;
    std::string positionIdLabel;
    std::vector<std::string> pastLabels;

    // Output Labels
    std::string logitsLabel;
    std::vector<std::string> presentLabels;
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

    void getNextTokens_greedy(const Ort::Value& logitsTensor, std::vector<RunInstance::DataType>& outNextTokens);
    void getNextTokens(const Ort::Value& logitsTensor, std::vector<RunInstance::DataType>& outNextTokens);

    void preGenerate(RunInstance& input);
    void generate(RunInstance& input);
    void postGenerate(RunInstance& input);

    // Generate the next token given the input.
    // Internally calls preRun() -> run() -> postRun()
    // Call them manually if you want to access correct values for present and past tensors inbetween.
    void generateNextToken(RunInstance& input);

    RunInstance* createRunInstance();
    RunInstance generateInput(std::vector<RunInstance::DataType>&& inputTokens);


    void printInputsInfo();
};

