#include "llama.hpp"
#include "modelLoadingParams.hpp"
#include "searchArgs.h"

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
    modelInfo.ctx = loadingData.json["n_ctx"];
    modelInfo.hidden_size = loadingData.json["n_embd"];
    modelInfo.num_attention_heads = loadingData.json["n_head"];
    modelInfo.num_layer = loadingData.json["n_layer"];
    modelInfo.nbMaxPositions = loadingData.json["n_positions"];
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

    modelInfo.presentLabels.resize(modelInfo.num_layer);
    for (std::int64_t i = 0; i < modelInfo.num_layer; i++)
    {
        modelInfo.presentLabels[i] = std::string("present_") + std::to_string(i);
    }
    
    // Output Labels
    modelInfo.logitsLabel = "logits";

    modelInfo.pastLabels.resize(modelInfo.num_layer);
    for (std::int64_t i = 0; i < modelInfo.num_layer; i++)
    {
        modelInfo.pastLabels[i] = std::string("past_") + std::to_string(i);
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
}

void LlamaPipeline::updateInputIdsTensor()
{
    DataType* data = inputIdsTensor.GetTensorMutableData<DataType>();
    for (const std::unique_ptr<Batch>& batch : batches)
    {
        auto& inputIds = batch->inputIds; 
        assert(!batch->inputIds.empty());
        std::copy(batch->inputIds.begin(), batch->inputIds.end(), data);
        data += batch->inputIds.size();
    }
}
void LlamaPipeline::updateAttentionMaskTensor()
{
    DataType* data = attentionMaskTensor.GetTensorMutableData<DataType>();
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
    if (modelInfo.pastLabels.size() != modelInfo.num_layer || pastTensors.size() != modelInfo.num_layer)
    {
        outResult = CppResult("(modelInfo.pastLabels.size() != modelInfo.num_layer || pastTensors.size() != modelInfo.num_layer); make sure the setup is correct");
    }

    for (std::int32_t i = 0; i < modelInfo.num_layer; i++)
    {
        if (i < pastTensors.size())
        ioBinding.BindInput(modelInfo.pastLabels[i].c_str(), pastTensors[i]);
    }
}

void LlamaPipeline::bindPresents(CppResult& outResult)
{
    const ModelInfo& modelInfo = model->modelInfo;
    if (modelInfo.presentLabels.size() < size_t(modelInfo.num_layer) || presentTensors.size() != size_t(modelInfo.num_layer))
    {
        outResult = CppResult("(modelInfo.presentLabels.size() < modelInfo.num_layer || presentTensors.size() != modelInfo.num_layer); make sure the setup is correct");
    }

    for (std::int32_t i = 0; i < modelInfo.num_layer; i++)
    {
        ioBinding.BindOutput(modelInfo.presentLabels[i].c_str(), presentTensors[i]);
    }
}
void LlamaPipeline::bindLogits()
{
    ioBinding.BindOutput(model->modelInfo.logitsLabel.c_str(), logitsTensor);
}

void LlamaPipeline::getPastTensorShape(std::array<std::int64_t, 5>& outPastShape) const
{
    const ModelInfo& modelInfo = model->modelInfo;
    outPastShape = {2, getNbBatches(), modelInfo.num_attention_heads, maxInputLength-1, modelInfo.hidden_size / modelInfo.num_attention_heads};
}
void LlamaPipeline::getPresentTensorShape(std::array<std::int64_t, 5>& outPresentShape) const
{
    const ModelInfo& modelInfo = model->modelInfo;
    outPresentShape = {2, getNbBatches(), modelInfo.num_attention_heads, seqLength, modelInfo.hidden_size / modelInfo.num_attention_heads};
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
void LlamaPipeline::generate(CppResult& outResult)
{
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
        createInputIdsTensor();
        createPositionIdsTensor();
        createAttentionMaskTensor();
        updateAttentionMaskTensor();

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
        createPresentTensors(getInputLength());

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

        std::array<std::int64_t, 5> pastShape;
        getPastTensorShape(pastShape);

        std::array<std::int64_t, 5> presentShape;
        getPresentTensorShape(presentShape);

        for (size_t i = 0; i < presentTensors.size(); i++)
        {
            const float* presentData = presentTensors[i].GetTensorData<float>();
            float* pastData = pastTensors[i].GetTensorMutableData<float>();

            // Ensure sizes are correct
            #ifndef NDEBUG
            auto tensor_info2 = presentTensors[i].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> presentShape2 = tensor_info2.GetShape();

            auto tensor_info = pastTensors[i].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> pastShape2 = tensor_info.GetShape();

            for (size_t i = 0; i < presentShape2.size(); i++)
            {
                assert(presentShape[i] == presentShape2[i]);
            }

            for (size_t i = 0; i < pastShape2.size(); i++)
            {
                assert(pastShape[i] == pastShape2[i]);
            }

            #endif
            
            IIOHandler::copyAndShiftPresentIntoNextPast(presentData, pastData, presentShape.data(), pastShape.data());
        }
    }

    updateInputIdsTensor();
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
void LlamaPipeline::batchSet(AutoRegressiveBatchHandle batch, DataType* inputTokens, std::int32_t nbTokens, std::int32_t fromPos)
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

inline auto _ = []() -> int
{
    getModelBuilderManager().registerModelBuilder("Llama", new Llama::LlamaBuilder());

    return 0;
}();