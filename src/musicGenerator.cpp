#include "musicGenerator.hpp"

#include <sstream>
#include "note.h"

// #define PRINT_TENSOR_UPDATE
// #define PRINT_TENSOR_SHAPE


void Batch::push(DataType inInputId, DataType inMask, DataType inPositionId)
{
    inputIds.push_back(inInputId);
    attentionMask.push_back(inMask);
    positionIds.push_back(inPositionId);
}

void Batch::push(DataType inInputId)
{
    DataType&& pos = positionIds.empty() ? 0 : (positionIds.back() + 1);  
    push(inInputId, 1, std::move(pos));
}

void Batch::pop()
{
    inputIds.pop_back();
    attentionMask.pop_back();
    positionIds.pop_back();
}

void Batch::set(const std::vector<DataType>& inTokens, std::int32_t fromPos)
{
    inputIds = inTokens;
    attentionMask.resize(inputIds.size(), 1);
    positionIds.resize(inputIds.size());

    for (std::int32_t i = 0; i < positionIds.size(); i++)
    {
        positionIds[i] = (i + fromPos) ; // @TODO : currently using circular encoding ; use relative encoding instead?
    }
}

size_t Batch::size() const
{
    assert(inputIds.size() == attentionMask.size());
    assert(inputIds.size() == positionIds.size());

    return inputIds.size();
}


std::wstring widen( const std::string& str )
{
    std::wostringstream wstm ;
    const std::ctype<wchar_t>& ctfacet = std::use_facet<std::ctype<wchar_t>>(wstm.getloc()) ;
    for( size_t i=0 ; i<str.size() ; ++i ) 
              wstm << ctfacet.widen( str[i] ) ;
    return wstm.str() ;
}

std::unique_ptr<Ort::Env> MusicGenerator::createOnnxEnv(bool useLogging)
{
    if (useLogging)
        return std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "MusicGenerator");
    else
        return std::make_unique<Ort::Env>();
}

void MusicGenerator::loadOnnxModel(const Ort::Env& env, const std::string& modelPath)
{
    // Create session options and enable optimization
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try 
    {
        session = std::make_unique<Ort::Session>(env, widen(modelPath).c_str(), session_options);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }

    // Load Config
    // @TODO : load from config

    // modelInfo.num_attention_heads = 8;
    // modelInfo.hidden_size = 512;
    // modelInfo.num_layer = 8;

    modelInfo.num_attention_heads = 4;
    modelInfo.hidden_size = 256;
    modelInfo.num_layer = 6;
    modelInfo.vocab_size = 10000;
    modelInfo.nbMaxPositions = 512;

    // Input Labels
    modelInfo.inputIdLabel = "input_ids";
    modelInfo.attentionMaskLabel = "attention_mask";
    modelInfo.positionIdLabel = "position_ids";

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
}

void RunInstance::resetSearchStrategy()
{
    searchStrategy = [](const struct SearchArgs& args, void* searchStrategyData) { return MusicGenerator::getNextTokens_greedy(args); };
}


void RunInstance::createInputIdsTensor(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), static_cast<std::int64_t>(batches.front()->size())};
    inputIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "InputIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createInputIdsTensorCache(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), 1};
    inputIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "InputIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createPositionIdsTensor(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), static_cast<std::int64_t>(batches.front()->size())};
    positionIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "PositionIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createPositionIdsTensorCache(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), 1};
    positionIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "PositionIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createAttentionMaskTensor(const struct ModelInfo& info, std::int64_t seqLength)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(getNbBatches()), seqLength};
    attentionMaskTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "AttentionMask : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createPresentTensors(const struct ModelInfo& info, std::int64_t seqLength)
{
    const std::array<std::int64_t, 5> emptyPresentShape = {2, std::int64_t(getNbBatches()), info.num_attention_heads, seqLength, info.hidden_size / info.num_attention_heads};
    presentTensors.clear();
    presentTensors.reserve(info.num_layer);
    for (std::int32_t i = 0; i < info.num_layer; i++)
    {
        presentTensors.push_back(Ort::Value::CreateTensor<float>(defaultAllocator, emptyPresentShape.data(), emptyPresentShape.size()));
    }

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "Present : " << emptyPresentShape[0] << " / " << emptyPresentShape[1] << 
    " / " << emptyPresentShape[2] << 
    " / " << emptyPresentShape[3] << 
    " / " << emptyPresentShape[4] << std::endl;
#endif
}

void RunInstance::createLogitsTensor(const struct ModelInfo& info, std::int64_t seqLength)
{
    const std::array<std::int64_t, 3> inputShape = {std::int64_t(getNbBatches()), seqLength, info.vocab_size};
    logitsTensor = Ort::Value::CreateTensor<float>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "Logits : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}

void RunInstance::createPastTensors(const struct ModelInfo& info, std::int64_t seqLength)
{
    const std::array<std::int64_t, 5> pastShape = {2, std::int64_t(getNbBatches()), info.num_attention_heads, seqLength, info.hidden_size / info.num_attention_heads};
    pastTensors.clear();
    pastTensors.reserve(info.num_layer);
    for (std::int32_t i = 0; i < info.num_layer; i++)
    {
        pastTensors.push_back(Ort::Value::CreateTensor<float>(defaultAllocator, pastShape.data(), pastShape.size()));
    }

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "Past : " << pastShape[0] << " / " << pastShape[1] << 
    " / " << pastShape[2] << 
    " / " << pastShape[3] << 
    " / " << pastShape[4] << std::endl;
#endif
}

void RunInstance::updateInputIdsTensor(const struct ModelInfo& info)
{
    DataType* data = inputIdsTensor.GetTensorMutableData<DataType>();
    size_t i = 0;
    for (Batch* batch : batches)
    {
        for (std::size_t j = 0; j < batch->inputIds.size(); j++)
        {
            data[i] = batch->inputIds[j];
            i++;
        }
    }
}

void RunInstance::updatePositionIdsTensor(const struct ModelInfo& info)
{
    DataType* data = positionIdsTensor.GetTensorMutableData<DataType>();
    size_t i = 0;
    for (Batch* batch : batches)
    {
        for (std::size_t j = 0; j < batch->positionIds.size(); j++)
        {
            data[i] = batch->positionIds[j] % info.nbMaxPositions;
            i++;
        }
    }
}

void RunInstance::updateAttentionMaskTensor(const struct ModelInfo& info)
{
    DataType* data = attentionMaskTensor.GetTensorMutableData<DataType>();
    size_t i = 0;
    for (Batch* batch : batches)
    {
        for (std::size_t j = 0; j < batch->attentionMask.size(); j++)
        {
            data[i] = batch->attentionMask[j];
            i++;
        }
    }
}

void RunInstance::updateAttentionMaskTensorCache(const ModelInfo& info, std::int64_t seqLength)
{
    DataType* data = attentionMaskTensor.GetTensorMutableData<DataType>();
    size_t i = 0;
    for (Batch* batch : batches)
    {
        for (std::int64_t j = 0; j < seqLength; j++)
        {
            data[i] = 1;
            i++;
        }
    }
}

void RunInstance::updateInputIdsTensorCache(const ModelInfo& info, const std::vector<RunInstance::DataType>& nextIds)
{
    DataType* data = inputIdsTensor.GetTensorMutableData<DataType>();
    size_t i = 0;
    for (Batch* batch : batches)
    {
        for (const RunInstance::DataType& id : nextIds)
        {
            data[i] = id;
            i++;
        }
    }
}

void RunInstance::bindInputIds(const ModelInfo& modelInfo)
{
    io_binding.BindInput(modelInfo.inputIdLabel.c_str(), inputIdsTensor);
}
void RunInstance::bindPositionIds(const ModelInfo& modelInfo)
{
    io_binding.BindInput(modelInfo.positionIdLabel.c_str(), positionIdsTensor);
}
void RunInstance::bindAttentionMask(const ModelInfo& modelInfo)
{
    io_binding.BindInput(modelInfo.attentionMaskLabel.c_str(), attentionMaskTensor);
}
void RunInstance::bindPasts(const ModelInfo& modelInfo)
{
    for (std::int32_t i = 0; i < modelInfo.num_layer; i++)
    {
        if (i < pastTensors.size())
        io_binding.BindInput(modelInfo.pastLabels[i].c_str(), pastTensors[i]);
    }
}

void RunInstance::bindPresents(const ModelInfo& modelInfo)
{
    for (std::int32_t i = 0; i < modelInfo.num_layer; i++)
    {
        io_binding.BindOutput(modelInfo.presentLabels[i].c_str(), presentTensors[i]);
    }
}
void RunInstance::bindLogits(const ModelInfo& modelInfo)
{
    io_binding.BindOutput(modelInfo.logitsLabel.c_str(), logitsTensor);
}

void RunInstance::bindInputs(const ModelInfo& modelInfo)
{
    bindInputIds(modelInfo);
    bindPositionIds(modelInfo);
    bindAttentionMask(modelInfo);
    bindPasts(modelInfo);
}
void RunInstance::bindOutputs(const ModelInfo& modelInfo)
{
    bindLogits(modelInfo);
    bindPresents(modelInfo);
}
void RunInstance::bind(const ModelInfo& modelInfo)
{
    bindInputs(modelInfo);
    bindOutputs(modelInfo);
}

void RunInstance::reset()
{
    subsequentGenerationIndex = 0;
    seqLength = 0;
}


void MusicGenerator::preGenerate(RunInstance& input)
{
    if (input.subsequentGenerationIndex == 0)
    {
        input.seqLength = input.batches[0]->inputIds.size();
        assert(input.seqLength <= input.maxInputLength);

        input.createInputIdsTensor(modelInfo);
        input.createPositionIdsTensor(modelInfo);
        input.createAttentionMaskTensor(modelInfo, input.seqLength);
        input.createPastTensors(modelInfo, 0);

        input.updateInputIdsTensor(modelInfo);
        input.updatePositionIdsTensor(modelInfo);
        input.updateAttentionMaskTensor(modelInfo);
        
        input.createLogitsTensor(modelInfo, input.seqLength);
        input.createPresentTensors(modelInfo, input.seqLength);

        input.bind(modelInfo);
    }
}
bool MusicGenerator::generate(RunInstance& input, const char*& outError)
{
    try 
    {
        // std::cout << "=============" << std::endl;
        session->Run(Ort::RunOptions{nullptr}, input.io_binding);
        outError = nullptr;
        return true;
    }
    catch(const Ort::Exception& e)
    {
        std::string errorMsg;
        errorMsg += "Error occurred: " + std::string(e.what());
        errorMsg += "Error code: " + std::to_string(e.GetOrtErrorCode());
        std::cout << errorMsg << std::endl;
        outError = errorMsg.c_str();
        return false;
    }
}

void MusicGenerator::getNextTokens_greedyFiltered(const struct SearchArgs& args, bool (*filter)(std::int32_t token, void* data), void* data)
{
    // Get the last token's logits for each sequence in the batch
    for(std::int32_t b = 0; b < args.nbBatches; ++b) 
    {
        // Pointer to the logits for the last token
        const float* last_logits = args.logitsTensor + (b * args.nbSequences + (args.nbSequences - 1)) * args.vocabSize;
        
        // Find the index with the maximum logit
        float max_logit = last_logits[0];
        std::int32_t max_index = 0;
        for(std::int32_t token = 1; token < args.vocabSize; token++) 
        {
            if(last_logits[token] > max_logit && (*filter)(token, data)) 
            {
                max_logit = last_logits[token];
                max_index = token;
            }
        }
        args.outNextTokens[b] = max_index;
    }
}

void MusicGenerator::getNextTokens_greedyPreFiltered(const SearchArgs& args, std::int32_t* availableTokens, std::int32_t nbAvailableToken)
{    
    assert(args.vocabSize >= nbAvailableToken);

    // Get the last token's logits for each sequence in the batch
    for(std::int32_t b = 0; b < args.nbBatches; b++) 
    {
        // Pointer to the logits for the last token
        const float* last_logits = args.logitsTensor + (b * args.nbSequences + (args.nbSequences - 1)) * args.vocabSize;
        
        // Find the index with the maximum logit
        float max_logit = last_logits[0];
        std::int32_t max_index = 0;
        for(std::int32_t i = 1; i < nbAvailableToken; i++) 
        {
            std::int32_t token = availableTokens[i];
            if(last_logits[token] > max_logit) 
            {
                max_logit = last_logits[token];
                max_index = token;
            }
        }
        args.outNextTokens[b] = max_index;
    }
}


void MusicGenerator::getNextTokens_greedy(const SearchArgs& args)
{    
    // Get the last token's logits for each sequence in the batch
    for(std::int32_t b = 0; b < args.nbBatches; ++b) 
    {
        // Pointer to the logits for the last token
        const float* last_logits = args.logitsTensor + (b * args.nbSequences + (args.nbSequences - 1)) * args.vocabSize;
        
        // Find the index with the maximum logit
        float max_logit = last_logits[0];
        std::int32_t max_index = 0;
        for(std::int32_t token = 1; token < args.vocabSize; token++) 
        {
            if(last_logits[token] > max_logit) 
            {
                max_logit = last_logits[token];
                max_index = token;
            }
        }
        args.outNextTokens[b] = max_index;
    }
}

// void MusicGenerator::getNextTokens_greedy(const Ort::Value& logitsTensor, std::vector<RunInstance::DataType>& outNextTokens)
// {
//     const Ort::Value& output_tensor = logitsTensor; // logits
//     const float* output_data = output_tensor.GetTensorData<float>();
//     Ort::TensorTypeAndShapeInfo tensorInfo = output_tensor.GetTensorTypeAndShapeInfo();
//     std::vector<int64_t> shape = tensorInfo.GetShape();

//     int64_t batchSize = shape[0];
//     int64_t vocab_size = shape[2];

//     // Greedy
//     // @TODO : optimize with custom search (?)
    
//     // Get the last token's logits for each sequence in the batch
//     for(int64_t b = 0; b < batchSize; ++b) {
//         // Pointer to the logits for the last token
//         const float* last_logits = output_data + (b * shape[1] + (shape[1] - 1)) * vocab_size;
        
//         // Find the index with the maximum logit
//         float max_logit = last_logits[0];
//         int max_index = 0;
//         for(int v = 1; v < vocab_size; ++v) {
//             if(last_logits[v] > max_logit) {
//                 max_logit = last_logits[v];
//                 max_index = v;
//             }
//         }
//         outNextTokens[b] = max_index;
//     }
// }

void MusicGenerator::getNextTokens(RunInstance& runInstance, const Ort::Value& logitsTensor, std::vector<RunInstance::DataType>& outNextTokens)
{
    // getNextTokens_greedy(logitsTensor, outNextTokens);



    Ort::TensorTypeAndShapeInfo tensorInfo = logitsTensor.GetTensorTypeAndShapeInfo();
    assert(tensorInfo.GetDimensionsCount() == 3);
    std::vector<int64_t> shape = tensorInfo.GetShape();
    SearchArgs args;
    args.logitsTensor = logitsTensor.GetTensorData<float>();
    args.outNextTokens = outNextTokens.data();
    args.nbBatches = static_cast<std::int32_t>(shape[0]);
    args.nbSequences = static_cast<std::int32_t>(shape[1]);
    args.vocabSize = static_cast<std::int32_t>(shape[2]);

    (*runInstance.searchStrategy)(args, runInstance.searchStrategyData);
    // getNextTokens_greedy(args);
}

void RunInstance::copyAndShiftPresentIntoNextPast(const float* presentData, float* pastData, int64_t presentShape[], int64_t pastShape[])
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

void RunInstance::getPastTensorShape(const ModelInfo& modelInfo, std::array<std::int64_t, 5>& outPastShape) const
{
    outPastShape = {2, getNbBatches(), modelInfo.num_attention_heads, maxInputLength-1, modelInfo.hidden_size / modelInfo.num_attention_heads};
}
void RunInstance::getPresentTensorShape(const ModelInfo& modelInfo, std::array<std::int64_t, 5>& outPresentShape) const
{
    outPresentShape = {2, getNbBatches(), modelInfo.num_attention_heads, seqLength, modelInfo.hidden_size / modelInfo.num_attention_heads};
}

std::int64_t computeMultiDimIdx(std::int64_t* shape, std::int64_t* indices)
{
    return indices[4]
        + indices[3] * shape[4]
        + indices[2] * shape[4] * shape[3]
        + indices[1] * shape[4] * shape[3] * shape[2]
        + indices[0] * shape[4] * shape[3] * shape[2] * shape[1];
}

void MusicGenerator::postGenerate(RunInstance& input)
{
    std::int64_t nbBatches = input.getNbBatches(); 

    input.nextTokens.resize(nbBatches);
    getNextTokens(input, input.logitsTensor, input.nextTokens);

    // Update next inputs
    for (int64_t b = 0; b < nbBatches; ++b) 
    {
        DataType lastElem = input.batches[b]->positionIds.back();
        input.batches[b]->positionIds.clear();
        input.batches[b]->positionIds.push_back(lastElem + 1); // @TODO : Modulo? or already handled by the model?

        input.batches[b]->lastGeneratedToken = input.nextTokens[b];
    }

    // // Update has_eos flags
    // std::vector<bool> has_eos(batchSize, false);
    // int64_t eos_token_id = 99999999999; // @TODO : update eos token id
    // for(int b = 0; b < batchSize; ++b) {
    //     if(next_tokens[b] == eos_token_id) {
    //         has_eos[b] = true;
    //     }
    // }

    // // Replace tokens with EOS where necessary
    // for(int b = 0; b < batchSize; ++b) {
    //     if(has_eos[b]) {
    //         next_tokens[b] = eos_token_id;
    //     }
    // }

    // Create IoBinding optimized inputs
    if (input.subsequentGenerationIndex == 0)
    {
        input.createInputIdsTensorCache(modelInfo);
        input.createPositionIdsTensorCache(modelInfo);
        input.createAttentionMaskTensor(modelInfo, 1);
        input.updateAttentionMaskTensorCache(modelInfo, 1);

        input.createLogitsTensor(modelInfo, 1);

        input.bindInputIds(modelInfo);
        input.bindPositionIds(modelInfo);
        input.bindAttentionMask(modelInfo);
        input.bindLogits(modelInfo);
    }

    if (input.seqLength < input.maxInputLength)
    {
        input.seqLength += 1;

        input.pastTensors = std::move(input.presentTensors);
        input.createPresentTensors(modelInfo, input.seqLength);

        input.bindPasts(modelInfo);
        input.bindPresents(modelInfo);
    }
    else
    {
        // if first time, the past is empty; have to initialize
        // we can easily get here if we brute force with reset() every generation,
        // or by having a starting context big enough
        if (input.subsequentGenerationIndex == 0)
        {
            // @TODO : createPastTensors() function instead 
            std::swap(input.presentTensors, input.pastTensors);
            input.createPresentTensors(modelInfo, input.maxInputLength-1);
            std::swap(input.presentTensors, input.pastTensors);
            input.bindPasts(modelInfo);
        }

        // remove the oldest "dim" of input.pastTensors when reaching 512 to prevent overloading  
        // Copy previous "past" values

        std::array<std::int64_t, 5> pastShape;
        input.getPastTensorShape(modelInfo, pastShape);

        std::array<std::int64_t, 5> presentShape;
        input.getPresentTensorShape(modelInfo, presentShape);

        for (size_t i = 0; i < input.presentTensors.size(); i++)
        {
            const float* presentData = input.presentTensors[i].GetTensorData<float>();
            float* pastData = input.pastTensors[i].GetTensorMutableData<float>();

            // Ensure sizes are correct
            #ifndef NDEBUG
            auto tensor_info2 = input.presentTensors[i].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> presentShape2 = tensor_info2.GetShape();

            auto tensor_info = input.pastTensors[i].GetTensorTypeAndShapeInfo();
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
            
            input.copyAndShiftPresentIntoNextPast(presentData, pastData, presentShape.data(), pastShape.data());
        }
    }

    input.updateInputIdsTensorCache(modelInfo, input.nextTokens);
    input.updatePositionIdsTensor(modelInfo); // @TODO : Cache version?

    input.subsequentGenerationIndex += 1;
}

void MusicGenerator::printInputsInfo()
{
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t j = 0; j < session->GetInputCount(); j++)
    {
        const auto& input_name = session->GetInputNameAllocated(j, allocator);
        auto type_info = session->GetInputTypeInfo(j);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << "Input " << input_name << " has shape: ";
        for (const auto dim : tensor_info.GetShape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
}

void RunInstance::printInputTensors()
{
    std::cout << "Inputs:" << std::endl;
    PrintTensorContent<int32_t>(inputIdsTensor);
    PrintTensorContent<int32_t>(positionIdsTensor);
    PrintTensorContent<int32_t>(attentionMaskTensor);

    if (!pastTensors.empty())
    {
        PrintTensorContent<int32_t>(pastTensors[0]);
    }
}

void RunInstance::printOutputTensors()
{
    std::cout << "Outputs:" << std::endl;
    PrintTensorContent<int32_t>(logitsTensor);

    if (!presentTensors.empty())
    {
        PrintTensorContent<int32_t>(presentTensors[0]);
    }
}

void MusicGenerator::generateNextToken(RunInstance& input)
{
    preGenerate(input);

    const char* errorMsg;
    generate(input, errorMsg);

    postGenerate(input);
}


RunInstance MusicGenerator::generateInput(std::vector<RunInstance::DataType>&& inputTokens)
{
    RunInstance input;

    input.batches.emplace_back();
    input.batches.front()->set(inputTokens);
    // input.updateInputTensors(modelInfo);

    return input;
}

RunInstance* MusicGenerator::createRunInstance()
{
    RunInstance* runInstance = new RunInstance();

    runInstance->io_binding = Ort::IoBinding(*session);

    return runInstance;
}