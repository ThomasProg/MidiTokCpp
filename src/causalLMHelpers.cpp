#include "causalLMHelpers.hpp"
#include <array>
#include <cassert>
#include <onnxruntime_cxx_api.h>
#include "searchArgs.h"

void IIOHandler::bindInputs(CppResult& outResult)
{
    bindInputIds();
    bindPositionIds();
    bindAttentionMask();
    bindPasts(outResult);
}
void IIOHandler::bindOutputs(CppResult& outResult)
{
    bindLogits();
    bindPresents(outResult);
}
void IIOHandler::bind(CppResult& outResult)
{
    bindInputs(outResult);
    bindOutputs(outResult);
}

void IIOHandler::createInputTensor(Ort::Value* tensor)
{
    assert(tensor != nullptr);
    const std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), static_cast<std::int64_t>(getInputLength())};
    *tensor = Ort::Value::CreateTensor<DataType>(getAllocator(), inputShape.data(), inputShape.size());
}

void IIOHandler::createInputIdsTensor()
{
    createInputTensor(getInputIdsTensor());
}
void IIOHandler::createPositionIdsTensor()
{
    createInputTensor(getPositionIdsTensor());
}
void IIOHandler::createAttentionMaskTensor()
{
    createInputTensor(getAttentionMaskTensor());
}

void IIOHandler::createLogitsTensor()
{
    Ort::Value* tensor = getLogitsTensor();
    assert(tensor != nullptr);
    const std::array<std::int64_t, 3> inputShape = {std::int64_t(getNbBatches()), getInputLength(), getVocabSize()};
    *tensor = Ort::Value::CreateTensor<float>(getAllocator(), inputShape.data(), inputShape.size());
}

void IIOHandler::createLogitsTensorCache()
{
    Ort::Value* tensor = getLogitsTensor();
    assert(tensor != nullptr);
    const std::array<std::int64_t, 3> inputShape = {std::int64_t(getNbBatches()), 1, getVocabSize()};
    *tensor = Ort::Value::CreateTensor<float>(getAllocator(), inputShape.data(), inputShape.size());
}

void IIOHandler::createPresentTensors(int64_t presentLength)
{
    int32_t nbAttentionHeads = getNbAttentionHeads();
    int32_t nbLayers = getNbLayers();
    const std::array<std::int64_t, 5> presentShape = {2, std::int64_t(getNbBatches()), nbAttentionHeads, presentLength, getHiddenSize() / nbAttentionHeads};

    std::vector<Ort::Value>* presentTensors = getPresentTensors();
    assert(presentTensors != nullptr);
    presentTensors->clear();
    presentTensors->reserve(nbLayers);
    for (std::int32_t i = 0; i < nbLayers; i++)
    {
        presentTensors->push_back(Ort::Value::CreateTensor<float>(getAllocator(), presentShape.data(), presentShape.size()));
    }
}

void IIOHandler::createPastTensors(int64_t pastLength)
{
    int32_t nbAttentionHeads = getNbAttentionHeads();
    int32_t nbLayers = getNbLayers();
    const std::array<std::int64_t, 5> pastShape = {2, std::int64_t(getNbBatches()), nbAttentionHeads, pastLength, getHiddenSize() / nbAttentionHeads};

    std::vector<Ort::Value>* pastTensors = getPastTensors();
    assert(pastTensors != nullptr);
    pastTensors->clear();
    pastTensors->reserve(nbLayers);
    for (std::int32_t i = 0; i < nbLayers; i++)
    {
        pastTensors->push_back(Ort::Value::CreateTensor<float>(getAllocator(), pastShape.data(), pastShape.size()));
    }
}

void IIOHandler::createFirstTimeTensors(CppResult& outResult)
{
    try
    {
        createInputIdsTensor();
        createPositionIdsTensor();
        createAttentionMaskTensor();
        createPastTensors(0);

        updateInputIdsTensor();
        updatePositionIdsTensor();
        updateAttentionMaskTensor();
        
        createLogitsTensor();
        createPresentTensors(getInputLength());

        bind(outResult);
    }
    catch (const std::exception& e)
    {
        outResult = CppResult(e.what());
    }
}

SearchArgs IIOHandler::createSearchArgs(Ort::Value& logitsTensor, std::vector<DataType>& outNextTokens)
{
    Ort::TensorTypeAndShapeInfo tensorInfo = logitsTensor.GetTensorTypeAndShapeInfo();
    assert(tensorInfo.GetDimensionsCount() == 3);
    std::vector<int64_t> shape = tensorInfo.GetShape(); // @TODO : optimize, remove allocation
    SearchArgs args;
    args.nbBatches = static_cast<std::int32_t>(shape[0]);
    args.nbSequences = static_cast<std::int32_t>(shape[1]);
    args.vocabSize = static_cast<std::int32_t>(shape[2]);
    args.logitsTensor = logitsTensor.GetTensorMutableData<float>();
    outNextTokens.resize(args.nbBatches);
    args.outNextTokens = outNextTokens.data();
    return args;
}

void IIOHandler::copyAndShiftPresentIntoNextPast(const float* presentData, float* pastData, int64_t presentShape[], int64_t pastShape[])
{
    int64_t presentId2 = presentShape[4];
    int64_t presentIdEnd2 = presentShape[3] * presentShape[4];
    int64_t pastId2 = 0;

    const int64_t presentOffset = presentShape[3] * presentShape[4];
    const int64_t pastOffset = pastShape[3] * pastShape[4];

    const int64_t limit = pastShape[0] * pastShape[1] * pastShape[2];

    assert((pastShape[3] + 1) == presentShape[3]);
    for (int64_t i = 0; i < limit; i++)
    {
        std::copy(presentData + presentId2, presentData + presentIdEnd2, pastData + pastId2);

        presentId2 += presentOffset;
        presentIdEnd2 += presentOffset;
        pastId2 += pastOffset;
    }
}
