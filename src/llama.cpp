#include "llama.hpp"
#include "modelLoadingParams.hpp"
#include "searchArgs.h"
#include "llama.h"

#include <iostream>

namespace Llama
{
// =================================== //
// ============== BATCH ============== // 
// =================================== //

void Batch::set(const DataType* inTokens, int32_t inNbTokens)
{
    inputIds.resize(inNbTokens);
    std::copy(inTokens, inTokens + inNbTokens, inputIds.data());
    attentionMask.resize(inNbTokens, 1);
}

size_t Batch::size() const
{
    return inputIds.size();
}

// =================================== //
// ========== LlamaModel =========== //
// =================================== //

LlamaModel::LlamaModel(const ModelLoadingParams& loadingData)
{
    modelInfo.ctx = loadingData.json["max_position_embeddings"];
    modelInfo.hidden_size = loadingData.json["hidden_size"];
    modelInfo.num_attention_heads = loadingData.json["num_attention_heads"];
    modelInfo.num_layer = loadingData.json["num_hidden_layers"];
    modelInfo.nbMaxPositions = loadingData.json["max_position_embeddings"];
    modelInfo.vocab_size = loadingData.json["vocab_size"];

    modelInfo.model_type = MakeCStr(loadingData.json["model_type"].template get<std::string>().c_str()); // "gpt2"
    modelInfo.torch_dtype = MakeCStr(loadingData.json["torch_dtype"].template get<std::string>().c_str()); // "float32"
}

IAutoRegressivePipeline* LlamaModel::createPipeline()
{
    return new LlamaPipeline(this);
}

CResult LlamaModel::onPostOnnxLoad()
{
    // Input Labels
    modelInfo.inputIdLabel = "input_ids";
    modelInfo.attentionMaskLabel = "attention_mask";
    // modelInfo.positionIdLabel = "position_ids";

    modelInfo.presentLabels.resize(modelInfo.num_layer*2);
    for (std::int64_t i = 0; i < modelInfo.num_layer; i++)
    {
        modelInfo.presentLabels[i] = std::string("present_key_") + std::to_string(i);
        modelInfo.presentLabels[i+modelInfo.num_layer] = std::string("present_value_") + std::to_string(i);
    }
    
    // Output Labels
    modelInfo.logitsLabel = "logits";

    modelInfo.pastLabels.resize(modelInfo.num_layer * 2);
    for (std::int64_t i = 0; i < modelInfo.num_layer; i++)
    {
        modelInfo.pastLabels[i] = std::string("past_key_") + std::to_string(i);
        modelInfo.pastLabels[i+modelInfo.num_layer] = std::string("past_value_") + std::to_string(i);
    }

    return CResult();
}

// =================================== //
// ========= LlamaPipeline ========= //
// =================================== //

LlamaPipeline::LlamaPipeline(LlamaModel* inModel) : model(inModel)
{
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api->GetAllocatorWithDefaultOptions(&allocator);
    ioBinding = Ort::IoBinding(*inModel->getSession());
    maxInputLength = inModel->modelInfo.nbMaxPositions;
}

void LlamaPipeline::updateInputIdsTensor()
{
    Batch::TensorDataType* data = inputIdsTensor.GetTensorMutableData<Batch::TensorDataType>();
    for (const std::unique_ptr<Batch>& batch : batches)
    {
        auto& inputIds = batch->inputIds; 
        assert(!batch->inputIds.empty());
        std::vector<Batch::TensorDataType> inputIdsExt(batch->inputIds.size());
        std::copy(batch->inputIds.begin(), batch->inputIds.end(), inputIdsExt.data());
        std::copy(inputIdsExt.begin(), inputIdsExt.end(), data);
        data += batch->inputIds.size();
    }
}
void LlamaPipeline::updateAttentionMaskTensor()
{
    Batch::TensorDataType* data = attentionMaskTensor.GetTensorMutableData<Batch::TensorDataType>();
    size_t i = 0;
    for (const std::unique_ptr<Batch>& batch : batches)
    {
        assert(!batch->attentionMask.empty());
        for (std::size_t j = 0; j < batch->attentionMask.size(); j++)
        {
            data[i] = batch->attentionMask[j];
            i++;
        }
    }
}

void LlamaPipeline::updateAttentionMaskTensorCache(std::int64_t seqLength)
{
    Batch::TensorDataType* data = attentionMaskTensor.GetTensorMutableData<Batch::TensorDataType>();
    size_t i = 0;
    for (std::unique_ptr<Batch>& batch : batches)
    {
        for (std::int64_t j = 0; j < seqLength; j++)
        {
            data[i] = 1;
            i++;
        }
    }
}

void LlamaPipeline::updateInputIdsTensorCache(const std::vector<DataType>& nextIds)
{
    Batch::TensorDataType* data = inputIdsTensor.GetTensorMutableData<Batch::TensorDataType>();
    size_t i = 0;
    for (std::unique_ptr<Batch>& batch : batches)
    {
        for (const DataType& id : nextIds)
        {
            data[i] = id;
            i++;
        }
    }
}

void LlamaPipeline::bindInputIds()
{
    ioBinding.BindInput(model->modelInfo.inputIdLabel.c_str(), inputIdsTensor);
}
void LlamaPipeline::bindAttentionMask()
{
    ioBinding.BindInput(model->modelInfo.attentionMaskLabel.c_str(), attentionMaskTensor);
}
void LlamaPipeline::bindPasts(CppResult& outResult)
{
    const ModelInfo& modelInfo = model->modelInfo;
    if (modelInfo.pastLabels.size() != modelInfo.num_layer*2 || pastTensors.size() != modelInfo.num_layer*2)
    {
        outResult = CppResult("(modelInfo.pastLabels.size() != modelInfo.num_layer*2 || pastTensors.size() != modelInfo.num_layer*2); make sure the setup is correct");
    }

    for (std::int32_t i = 0; i < modelInfo.num_layer*2; i++)
    {
        if (i < pastTensors.size())
        ioBinding.BindInput(modelInfo.pastLabels[i].c_str(), pastTensors[i]);
    }
}

void LlamaPipeline::bindPresents(CppResult& outResult)
{
    const ModelInfo& modelInfo = model->modelInfo;
    if (modelInfo.presentLabels.size() < size_t(modelInfo.num_layer*2) || presentTensors.size() != size_t(modelInfo.num_layer*2))
    {
        outResult = CppResult("(modelInfo.presentLabels.size() < modelInfo.num_layer*2 || presentTensors.size() != modelInfo.num_layer*2); make sure the setup is correct");
    }

    for (std::int32_t i = 0; i < modelInfo.num_layer*2; i++)
    {
        ioBinding.BindOutput(modelInfo.presentLabels[i].c_str(), presentTensors[i]);
    }
}
void LlamaPipeline::bindLogits()
{
    ioBinding.BindOutput(model->modelInfo.logitsLabel.c_str(), logitsTensor);
}

void LlamaPipeline::createInputTensor(Ort::Value* tensor) 
{
    assert(tensor != nullptr);
    const std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), static_cast<std::int64_t>(getInputLength())};
    *tensor = Ort::Value::CreateTensor<Batch::TensorDataType>(getAllocator(), inputShape.data(), inputShape.size());
}

void LlamaPipeline::createPresentTensors(int64_t presentLength)
{
    int32_t nbAttentionHeads = getNbAttentionHeads();
    int32_t nbLayers = getNbLayers();
    const std::array<std::int64_t, 4> presentShape = {std::int64_t(getNbBatches()), nbAttentionHeads, presentLength, getHiddenSize() / nbAttentionHeads};

    std::vector<Ort::Value>* presentTensors = getPresentTensors();
    assert(presentTensors != nullptr);
    presentTensors->clear();
    presentTensors->reserve(nbLayers*2);
    for (std::int32_t i = 0; i < nbLayers*2; i++)
    {
        presentTensors->push_back(Ort::Value::CreateTensor<float>(getAllocator(), presentShape.data(), presentShape.size()));
    }
}

void LlamaPipeline::createPastTensors(int64_t pastLength)
{
    int32_t nbAttentionHeads = getNbAttentionHeads();
    int32_t nbLayers = getNbLayers();
    const std::array<std::int64_t, 4> pastShape = {std::int64_t(getNbBatches()), nbAttentionHeads, pastLength, getHiddenSize() / nbAttentionHeads};

    std::vector<Ort::Value>* pastTensors = getPastTensors();
    assert(pastTensors != nullptr);
    pastTensors->clear();
    pastTensors->reserve(nbLayers*2);
    for (std::int32_t i = 0; i < nbLayers*2; i++)
    {
        pastTensors->push_back(Ort::Value::CreateTensor<float>(getAllocator(), pastShape.data(), pastShape.size()));
    }
}

void LlamaPipeline::getPastTensorShape(std::array<std::int64_t, 4>& outPastShape) const
{
    const ModelInfo& modelInfo = model->modelInfo;
    outPastShape = {getNbBatches(), modelInfo.num_attention_heads, maxInputLength-1, modelInfo.hidden_size / modelInfo.num_attention_heads};
}
void LlamaPipeline::getPresentTensorShape(std::array<std::int64_t, 4>& outPresentShape) const
{
    const ModelInfo& modelInfo = model->modelInfo;
    outPresentShape = {getNbBatches(), modelInfo.num_attention_heads, seqLength, modelInfo.hidden_size / modelInfo.num_attention_heads};
}

void LlamaPipeline::preGenerate(CppResult& outResult)
{
    if (subsequentGenerationIndex == 0)
    {
        if (seqLength > maxInputLength)
        {
            outResult = CppResult("assert(seqLength <= maxInputLength) is false"); 
        }
        IIOHandler::createFirstTimeTensors(outResult);
    }
}

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

void LlamaPipeline::createInputIdsTensorCache()
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), 1};
    inputIdsTensor = Ort::Value::CreateTensor<Batch::TensorDataType>(allocator, inputShape.data(), inputShape.size());
}
void LlamaPipeline::createPositionIdsTensorCache()
{

}

void LlamaPipeline::generate(CppResult& outResult)
{
    // PrintTensorContent<int64_t>(inputIdsTensor);
    model->generate(ioBinding, outResult);
}
void LlamaPipeline::postGenerate(CppResult& outResult)
{
    std::int64_t nbBatches = getNbBatches(); 

    SearchArgs searchArgs = createSearchArgs(logitsTensor, nextTokens);
    (*searchStrategy)(searchArgs, searchStrategyData); // filter and sample next tokens

    // Update next inputs
    for (int64_t b = 0; b < nbBatches; ++b) 
    {
        Batch& batch = *batches[b];
        batch.lastGeneratedToken = nextTokens[b];

        batch.attentionMask.resize(1, 1);
        batch.inputIds.resize(1, batch.lastGeneratedToken);
    }

    // Create IoBinding optimized inputs
    if (subsequentGenerationIndex == 0)
    {
        createInputIdsTensorCache();
        createPositionIdsTensorCache();
        createAttentionMaskTensor();
        updateAttentionMaskTensorCache(1);

        createLogitsTensor();

        bindInputIds();
        bindPositionIds();
        bindAttentionMask();
        bindLogits();
    }

    if (seqLength < maxInputLength)
    {
        seqLength += 1;

        pastTensors = std::move(presentTensors);
        createPresentTensors(seqLength+1);

        bindPasts(outResult);
        bindPresents(outResult);
    }
    else
    {
        // if first time, the past is empty; have to initialize
        // we can easily get here if we brute force with reset() every generation,
        // or by having a starting context big enough
        if (subsequentGenerationIndex == 0)
        {
            // @TODO : createPastTensors() function instead 
            std::swap(presentTensors, pastTensors);
            createPresentTensors(maxInputLength-1);
            std::swap(presentTensors, pastTensors);
            bindPasts(outResult);
        }

        // remove the oldest "dim" of pastTensors when reaching 512 to prevent overloading  
        // Copy previous "past" values

        std::array<std::int64_t, 4> pastShape;
        getPastTensorShape(pastShape);
        std::array<std::int64_t, 5> pastShapeIncr = {1, pastShape[0], pastShape[1], pastShape[2], pastShape[3]};

        std::array<std::int64_t, 4> presentShape;
        getPresentTensorShape(presentShape);
        std::array<std::int64_t, 5> presentShapeIncr = {1, presentShape[0], presentShape[1], presentShape[2], presentShape[3]};


        for (size_t i = 0; i < presentTensors.size(); i++)
        {
            const float* presentData = presentTensors[i].GetTensorData<float>();
            float* pastData = pastTensors[i].GetTensorMutableData<float>();

            // Ensure sizes are correct
            // #ifndef NDEBUG
            // auto tensor_info2 = presentTensors[i].GetTensorTypeAndShapeInfo();
            // std::vector<int64_t> presentShape2 = tensor_info2.GetShape();

            // auto tensor_info = pastTensors[i].GetTensorTypeAndShapeInfo();
            // std::vector<int64_t> pastShape2 = tensor_info.GetShape();

            // for (size_t i = 0; i < presentShape2.size(); i++)
            // {
            //     assert(presentShape[i] == presentShape2[i]);
            // }

            // for (size_t i = 0; i < pastShape2.size(); i++)
            // {
            //     assert(pastShape[i] == pastShape2[i]);
            // }

            // #endif
            
            IIOHandler::copyAndShiftPresentIntoNextPast(presentData, pastData, presentShapeIncr.data(), pastShapeIncr.data());
        }
    }

    updateInputIdsTensorCache(nextTokens);
    updatePositionIdsTensor();

    subsequentGenerationIndex += 1;
}
LlamaModel* LlamaPipeline::getModel() const
{
    return model;
}
void LlamaPipeline::reset()
{
    subsequentGenerationIndex = 0;
    seqLength = 0;
}





void LlamaPipeline::setSearchStrategyData(void* searchStrategyData)
{
    this->searchStrategyData = searchStrategyData;
}
void LlamaPipeline::setSearchStrategy(TSearchStrategy searchStrategy)
{
    this->searchStrategy = searchStrategy;
}
int32_t LlamaPipeline::getNbBatches() const
{
    return static_cast<int32_t>(batches.size());
}

// returns the index of the new batch
AutoRegressiveBatchHandle LlamaPipeline::addBatch()
{
    batches.emplace_back(std::make_unique<Batch>());
    return int32_t(batches.size()) - 1;
}
void LlamaPipeline::removeAllBatches()
{
    batches.clear();
}

// If the model has to be updated, for example RNN state being reset if resetting the batch
int32_t LlamaPipeline::batchGetLastGeneratedToken(AutoRegressiveBatchHandle batch)
{
    return batches[batch]->lastGeneratedToken;
}
void LlamaPipeline::batchSet(AutoRegressiveBatchHandle batch, Batch::DataType* inputTokens, std::int32_t nbTokens, std::int32_t fromPos)
{
    batches[batch]->set(inputTokens, nbTokens);
}
void LlamaPipeline::setMaxInputLength(int32_t newMaxInputLength)
{
    maxInputLength = newMaxInputLength;
}









// =================================== //
// ============ REGISTER ============= //
// =================================== //

LlamaModel* LlamaBuilder::loadModel(const ModelLoadingParams& loadingData) const
{
    LlamaModel* gen = new LlamaModel(loadingData);

    const std::string path = std::string(loadingData.modelPath.Str()) + "/model.onnx";
    gen->loadOnnxModel(*env, path.c_str());

    return gen;
}
}

void registerLlamaModelBuilder()
{
    getModelBuilderManager().registerModelBuilder("llama", new Llama::LlamaBuilder());
}