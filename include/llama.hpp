#pragma once

#include "abstractPipeline.hpp"
#include "modelBuilderManager.hpp"
#include "causalLMHelpers.hpp"
#include "generationHistory.hpp"
#include <onnxruntime_cxx_api.h>

namespace Llama
{
struct Batch
{
public:
    using DataType = std::int32_t;
    using TensorDataType = std::int64_t;

    // Values
    std::vector<DataType> inputIds; // encoded tokens
    std::vector<DataType> attentionMask;

    DataType lastGeneratedToken = 0;

// public:
//     void push(DataType inInputId, DataType inMask, DataType inPositionId);
//     void push(DataType inInputId);
//     void pop();

    void set(const DataType* inTokens, int32_t inNbTokens);
    size_t size() const;
};

struct ModelInfo
{
    // @TODO : load from config
    int64_t num_attention_heads; // n_head
    int64_t ctx; // n_embd
    int64_t hidden_size; // n_embd
    int64_t num_layer; // n_layer
    int64_t vocab_size; // vocab size
    int64_t nbMaxPositions; // n_positions
    
    CStr model_type;
    CStr torch_dtype;

    // Input Labels
    std::string inputIdLabel;
    std::string attentionMaskLabel;
    // std::string positionIdLabel;
    std::vector<std::string> pastLabels;

    // Output Labels
    std::string logitsLabel;
    std::vector<std::string> presentLabels;
};

class LlamaModel final : public AOnnxModel
{
public:
    ModelInfo modelInfo;

public:
    LlamaModel() = default;
    LlamaModel(const ModelLoadingParams& loadingData);

    // BEGIN - AModel
    virtual IAutoRegressivePipeline* createPipeline() override;
    // END - AModel

    // BEGIN - AOnnxModel
    virtual CResult onPostOnnxLoad() override;
    // END - AOnnxModel
};


// only one batch ?
class LlamaPipeline final : public IAutoRegressivePipeline, private IIOHandler
{
private:
    LlamaModel* model = nullptr;
    std::unique_ptr<GenerationHistory> history;


    void* searchStrategyData = nullptr;
    using TSearchStrategy = void (*)(const struct SearchArgs& args, void* searchStrategyData);
    TSearchStrategy searchStrategy = nullptr;

    std::vector<std::unique_ptr<Batch>> batches;

    std::int64_t seqLength = 0;
    std::int64_t subsequentGenerationIndex = 0;
    std::int64_t maxInputLength = 0;
    std::vector<Batch::DataType> nextTokens;

    Ort::IoBinding ioBinding = Ort::IoBinding(nullptr);
    OrtAllocator* allocator = nullptr;

public:
    LlamaPipeline(LlamaModel* inModel);

    // BEGIN - IPipeline
    virtual void preGenerate(CppResult& outResult) override;
    virtual void generate(CppResult& outResult) override;
    virtual void postGenerate(CppResult& outResult) override;
    virtual LlamaModel* getModel() const override;
    virtual void reset() override;
    // END - IPipeline

    // BEGIN - IAutoRegressivePipeline
    virtual void setSearchStrategyData(void* searchStrategyData) override;
    virtual void setSearchStrategy(TSearchStrategy searchStrategy) override;
    virtual int32_t getNbBatches() const override;

    // returns the index of the new batch
    virtual AutoRegressiveBatchHandle addBatch() override;
    virtual void removeAllBatches() override;

    // If the model has to be updated, for example RNN state being reset if resetting the batch
    virtual int32_t batchGetLastGeneratedToken(AutoRegressiveBatchHandle batch) override;
    virtual void batchSet(AutoRegressiveBatchHandle batch, DataType* inputTokens, std::int32_t nbTokens, std::int32_t fromPos) override;
    virtual void setMaxInputLength(int32_t newMaxInputLength) override;

    virtual void createHistory(const MidiTokenizer& tokenizer) override
    {
        history = std::make_unique<GenerationHistory>(tokenizer);
    }
    virtual GenerationHistory* getHistory(AutoRegressiveBatchHandle batchHandle) const override
    {
        return history.get(); // @TODO : add multiple batches support
    }
    // END - IAutoRegressivePipeline

private:
    // Inputs
    Ort::Value inputIdsTensor = Ort::Value(nullptr);
    Ort::Value attentionMaskTensor = Ort::Value(nullptr);
    std::vector<Ort::Value> presentTensors; // cache

    // Outputs
    Ort::Value logitsTensor = Ort::Value(nullptr);
    std::vector<Ort::Value> pastTensors; // cache

public:
    // BEGIN - IIOHandler
    virtual void createInputTensor(Ort::Value* tensor) override;
    virtual void createPresentTensors(int64_t presentLength);
    virtual void createPastTensors(int64_t pastLength) override;

    void getPastTensorShape(std::array<std::int64_t, 4>& outPastShape) const;
    void getPresentTensorShape(std::array<std::int64_t, 4>& outPresentShape) const;

    virtual void createPositionIdsTensor() override {}
    
    virtual void updateInputIdsTensor() override;
    virtual void updatePositionIdsTensor() override {}
    virtual void updateAttentionMaskTensor() override;

    void updateAttentionMaskTensorCache(std::int64_t seqLength);
    void updateInputIdsTensorCache(const std::vector<DataType>& nextIds);

    // Bind Inputs
    virtual void bindInputIds() override;
    virtual void bindPositionIds() override {}
    virtual void bindAttentionMask() override;
    virtual void bindPasts(CppResult& outResult) override;

    // Bind Outputs
    virtual void bindPresents(CppResult& outResult) override;
    virtual void bindLogits() override;

    virtual int32_t getInputLength() const override { return static_cast<int32_t>(batches.front()->size()); }
    virtual int32_t getVocabSize() const override { return static_cast<int32_t>(model->modelInfo.vocab_size); }
    virtual int32_t getNbAttentionHeads() const override { return static_cast<int32_t>(model->modelInfo.num_attention_heads); }
    virtual int32_t getHiddenSize() const override { return static_cast<int32_t>(model->modelInfo.hidden_size); }
    virtual int32_t getNbLayers() const override { return static_cast<int32_t>(model->modelInfo.num_layer); }

    virtual Ort::Value* getInputIdsTensor() { return &inputIdsTensor; }
    virtual Ort::Value* getAttentionMaskTensor() { return &attentionMaskTensor; }
    virtual std::vector<Ort::Value>* getPresentTensors() { return &presentTensors; }
    virtual Ort::Value* getLogitsTensor() { return &logitsTensor; }
    virtual std::vector<Ort::Value>* getPastTensors() { return &pastTensors; }
    virtual OrtAllocator* getAllocator() { return allocator; }
    // END - IIOHandler

    void createInputIdsTensorCache();
    void createPositionIdsTensorCache();
};


class LlamaBuilder final : public OnnxModelBuilder
{
public:
    virtual class LlamaModel* loadModel(const ModelLoadingParams& loadingData) const override;
};
}