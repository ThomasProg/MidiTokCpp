#pragma once

#include <map>
#include <vector>
#include <list>
#include <onnxruntime_cxx_api.h>
#include "midiTokenizer.hpp"
#include "fwd.h"
#include "utilities.hpp"

template<typename T>
void PrintTensorContent(const Ort::Value& value) {
    // Get the tensor's shape
    auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
    
    // Get the number of elements in the tensor
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }

    // Get the tensor's data (assuming it's of type float)
    const T* tensor_data = value.GetTensorData<T>();

    // Print the shape
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    // Print the tensor data
    std::cout << "Tensor data: ";
    for (size_t i = 0; i < num_elements; ++i) {
        std::cout << tensor_data[i] << " ";
        if ((i + 1) % 10 == 0) {  // Print 10 elements per line for better readability
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}


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

std::int64_t computeMultiDimIdx(std::int64_t* shape, std::int64_t* indices);

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

    std::vector<RunInstance::DataType> nextTokens;

public:
    void* searchStrategyData = nullptr;
    using TSearchStrategy = void (*)(const struct SearchArgs& args, void* searchStrategyData);

    // static inline constexpr TSearchStrategy defaultSearchStrategy = [](const struct SearchArgs& args, void* searchStrategyData) { return MusicGenerator::getNextTokens_greedy(args); };
    TSearchStrategy searchStrategy = nullptr;//defaultSearchStrategy;

public:
    RunInstance()
    {
        resetSearchStrategy();
    }
    void resetSearchStrategy();

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

    void getPastTensorShape(const ModelInfo& modelInfo, std::array<std::int64_t, 5>& outPastShape) const;
    void getPresentTensorShape(const ModelInfo& modelInfo, std::array<std::int64_t, 5>& outPresentShape) const;

    // Bind Inputs
    void bindInputIds(const ModelInfo& modelInfo);
    void bindPositionIds(const ModelInfo& modelIngetfo);
    void bindAttentionMask(const ModelInfo& modelInfo);
    void bindPasts(const ModelInfo& modelInfo, CppResult& outResult);

    // Bind Outputs
    void bindPresents(const ModelInfo& modelInfo, CppResult& outResult);
    void bindLogits(const ModelInfo& modelInfo);

    void bindInputs(const ModelInfo& modelInfo, CppResult& outResult);
    void bindOutputs(const ModelInfo& modelInfo, CppResult& outResult);
    void bind(const ModelInfo& modelInfo, CppResult& outResult);

    std::int64_t getNbBatches() const;

    void reset();

    void copyAndShiftPresentIntoNextPast(const float* presentData, float* pastData, int64_t presentShape[], int64_t pastShape[]);

    void printInputTensors();
    void printOutputTensors();
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
    std::string positionIdLabel;
    std::vector<std::string> pastLabels;

    // Output Labels
    std::string logitsLabel;
    std::vector<std::string> presentLabels;
};

#include "abstractPipeline.hpp"
#include "modelBuilderManager.hpp"

class MusicGenerator : public AOnnxModel
{
public:
    ModelInfo modelInfo;

public:
    MusicGenerator() = default;
    MusicGenerator(const ModelLoadingParams& jsonData);

    static std::unique_ptr<Ort::Env> createOnnxEnv(bool useLogging = false);
    // void loadOnnxModel(const Ort::Env& env, const std::string& modelPath);
    virtual CResult onPostOnnxLoad() override;

    // static void getNextTokens_greedy(const Ort::Value& logitsTensor, std::vector<RunInstance::DataType>& outNextTokens);
    static void getNextTokens_greedyFiltered(const struct SearchArgs& args, bool (*filter)(std::int32_t token, void* data), void* data);
    static void getNextTokens_greedyPreFiltered(const struct SearchArgs& args, std::int32_t* availableTokens, std::int32_t nbAvailableToken);
    static void getNextTokens_greedy(const struct SearchArgs& args);
    void getNextTokens(RunInstance& runInstance, const Ort::Value& logitsTensor, std::vector<RunInstance::DataType>& outNextTokens);

    void preGenerate(RunInstance& input, CppResult& outResult);
    void generate(RunInstance& input, CppResult& outResult);
    void postGenerate(RunInstance& input, CppResult& outResult);

    // Generate the next token given the input.
    // Internally calls preRun() -> run() -> postRun()
    // Call them manually if you want to access correct values for present and past tensors inbetween.
    void generateNextToken(RunInstance& input);

    RunInstance* createRunInstance();
    RunInstance generateInput(std::vector<RunInstance::DataType>&& inputTokens);

    void printInputsInfo();

    // BEGIN - AModel
    virtual IAutoRegressivePipeline* createPipeline();
    // END - AModel
};

// Necessary if a single model can be infered in different threads at the same time.
// Also helps splitting the model from the data.
class MusicGeneratorPipeline : public IAutoRegressivePipeline
{
private:
    MusicGeneratorHandle musicGenerator = nullptr;
    RunInstanceHandle runInstance = nullptr;

public:
    MusicGeneratorPipeline(MusicGeneratorHandle newMusicGenerator, RunInstanceHandle newRunInstance);
    virtual void preGenerate(CppResult& outResult) override;
    virtual void generate(CppResult& outResult) override;
    virtual void postGenerate(CppResult& outResult) override;
    virtual AModel* getModel() const override;




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
    virtual void reset() override;
};

class MusicGeneratorBuilder : public OnnxModelBuilder
{
public:
    virtual class MusicGenerator* loadModel(const ModelLoadingParams& jsonData) const override;
};
