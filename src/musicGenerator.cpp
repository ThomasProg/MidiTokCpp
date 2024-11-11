#include "musicGenerator.hpp"

#include <sstream>

#define PRINT_TENSOR_UPDATE
#define PRINT_TENSOR_SHAPE


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

// void RunInstance::updateInputTensors(const ModelInfo& info)
// {
//     bool usePast = !presentTensors.empty();

//     // @TODO : support multiple batches
//     assert(batches.size() == 1);

//     auto pushTensorSingle = [&](DataType& data, std::int32_t index)
//     {
//         std::vector<int64_t> input_shape = {int64_t(batches.size()), 1};
//         inputDataTensors.push_back(Ort::Value::CreateTensor<DataType>(memory_info, &data, sizeof(DataType), input_shape.data(), input_shape.size()));
//     };

//     auto pushTensor = [&](std::vector<DataType>& data, std::int32_t index)
//     {
//         std::vector<int64_t> input_shape = {int64_t(batches.size()), static_cast<int64_t>(batches.front()->size())};
//         inputDataTensors.push_back(Ort::Value::CreateTensor<DataType>(memory_info, data.data(), data.size() * sizeof(DataType), input_shape.data(), input_shape.size()));
//     };

//     inputDataTensors.clear();

//     if (usePast)
//     {
//         // @TODO : support multiple batches
//         assert(batches.size() == 1);
//         pushTensorSingle(batches.front()->inputIds.back(), 0);
//         pushTensor(batches.front()->attentionMask, 1);
//         pushTensorSingle(batches.front()->positionIds.back(), 2);
//     }
//     else 
//     {
//         // @TODO : support multiple batches
//         assert(batches.size() == 1);
//         pushTensor(batches.front()->inputIds, 0);
//         pushTensor(batches.front()->attentionMask, 1);
//         pushTensor(batches.front()->positionIds, 2);
//     }


//     if (!usePast)
//     {
//         // Update Past Tensors
//         const std::array<int64_t, 5> past_shape = {
//             2, int64_t(batches.size()), info.num_attention_heads, 0, info.hidden_size / info.num_attention_heads
//         };

//         for (int64_t i = 0; i < info.num_layer; i++)
//         {
//             Ort::Value past_tensor = Ort::Value::CreateTensor<float>(cacheAllocator, past_shape.data(), past_shape.size());
//             inputDataTensors.push_back(std::move(past_tensor));
//         }


//         // size_t nbElements = 1;
//         // for (int64_t v : past_shape)
//         // {
//         //     nbElements *= v;
//         // }
//         // std::vector<float> past;
//         // for (size_t j = 0; j < nbElements; j++)
//         //     past.push_back(0.0f);

//         // std::vector<int64_t> past_shape_v(std::begin(past_shape), std::end(past_shape));
//         // for (int64_t i = 0; i < info.num_layer; i++)
//         // {
//         //     inputDataTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, past.data(), past.size() * sizeof(float), past_shape_v.data(), past_shape_v.size()));
//         // }
//     }
//     else
//     {
//         for (size_t i = 0; i < cachedValues.size(); i++)
//         {
//             inputDataTensors.emplace_back(std::move(cachedValues[i]));
//         }
//         cachedValues.clear();
//     }
// }



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
    modelInfo.vocab_size = 30000;
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

void RunInstance::createInputIdsTensor(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(batches.size()), static_cast<std::int64_t>(batches.front()->size())};
    inputIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "InputIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createInputIdsTensorCache(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(batches.size()), 1};
    inputIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "InputIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createPositionIdsTensor(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(batches.size()), static_cast<std::int64_t>(batches.front()->size())};
    positionIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "PositionIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createPositionIdsTensorCache(const struct ModelInfo& info)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(batches.size()), 1};
    positionIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "PositionIds : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createAttentionMaskTensor(const struct ModelInfo& info, std::int64_t seqLength)
{
    std::array<std::int64_t, 2> inputShape = {std::int64_t(batches.size()), seqLength};
    attentionMaskTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "AttentionMask : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}
void RunInstance::createPresentTensors(const struct ModelInfo& info, std::int64_t seqLength)
{
    const std::array<std::int64_t, 5> emptyPresentShape = {2, std::int64_t(batches.size()), info.num_attention_heads, seqLength, info.hidden_size / info.num_attention_heads};
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

// void RunInstance::createInputTensors(const struct ModelInfo& info, std::int64_t seqLength)
// {
//     std::array<std::int64_t, 2> inputShape = {std::int64_t(batches.size()), static_cast<std::int64_t>(batches.front()->size())};
//     inputIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());
//     positionIdsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());
//     attentionMaskTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

//     const std::array<std::int64_t, 5> emptyPresentShape = {2, std::int64_t(batches.size()), info.num_attention_heads, 0, info.hidden_size / info.num_attention_heads};
//     for (std::int32_t i = 0; i < info.num_layer; i++)
//     {
//         presentTensors[i] = Ort::Value::CreateTensor<DataType>(defaultAllocator, emptyPresentShape.data(), emptyPresentShape.size());
//     }
// }

void RunInstance::createLogitsTensor(const struct ModelInfo& info, std::int64_t seqLength)
{
    const std::array<std::int64_t, 3> inputShape = {std::int64_t(batches.size()), seqLength, info.vocab_size};
    logitsTensor = Ort::Value::CreateTensor<float>(defaultAllocator, inputShape.data(), inputShape.size());

#ifdef PRINT_TENSOR_SHAPE
    std::cout << "Logits : " << inputShape[0] << " / " << inputShape[1] << std::endl;
#endif
}

void RunInstance::createPastTensors(const struct ModelInfo& info, std::int64_t seqLength)
{
    const std::array<std::int64_t, 5> pastShape = {2, std::int64_t(batches.size()), info.num_attention_heads, seqLength, info.hidden_size / info.num_attention_heads};
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
        for (std::int64_t j = 0; j < batch->inputIds.size(); j++)
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
        for (std::int64_t j = 0; j < batch->positionIds.size(); j++)
        {
            data[i] = batch->positionIds[j];
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
        for (std::int64_t j = 0; j < batch->attentionMask.size(); j++)
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

// void RunInstance::createOutputTensors(const struct ModelInfo& info, std::int64_t seqLength)
// {
//     const std::array<std::int64_t, 3> inputShape = {std::int64_t(batches.size()), seqLength, info.vocab_size};
//     logitsTensor = Ort::Value::CreateTensor<DataType>(defaultAllocator, inputShape.data(), inputShape.size());

//     const std::array<std::int64_t, 5> emptyPresentShape = {2, std::int64_t(batches.size()), info.num_attention_heads, seqLength, info.hidden_size / info.num_attention_heads};
//     for (std::int32_t i = 0; i < info.num_layer; i++)
//     {
//         presentTensors[i] = Ort::Value::CreateTensor<DataType>(defaultAllocator, emptyPresentShape.data(), emptyPresentShape.size());
//     }
// }

void RunInstance::bindInputs(const ModelInfo& modelInfo)
{
    // Bind inputs
    io_binding.BindInput(modelInfo.inputIdLabel.c_str(), inputIdsTensor);
    io_binding.BindInput(modelInfo.positionIdLabel.c_str(), positionIdsTensor);
    io_binding.BindInput(modelInfo.attentionMaskLabel.c_str(), attentionMaskTensor);

    for (std::int32_t i = 0; i < modelInfo.num_layer; i++)
    {
        if (i < pastTensors.size())
        io_binding.BindInput(modelInfo.pastLabels[i].c_str(), pastTensors[i]);
    }
}
void RunInstance::bindOutputs(const ModelInfo& modelInfo)
{
    // Bind outputs
    io_binding.BindOutput(modelInfo.logitsLabel.c_str(), logitsTensor);

    for (std::int32_t i = 0; i < modelInfo.num_layer; i++)
    {
        io_binding.BindOutput(modelInfo.presentLabels[i].c_str(), presentTensors[i]);
    }
}
void RunInstance::bind(const ModelInfo& modelInfo)
{
    bindInputs(modelInfo);
    bindOutputs(modelInfo);
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


void MusicGenerator::generate(RunInstance& input)
{
    if (input.isFirstRun)
    {
        input.seqLength = input.batches[0]->inputIds.size();
        input.createInputIdsTensor(modelInfo);
        input.createPositionIdsTensor(modelInfo);
        input.createAttentionMaskTensor(modelInfo, input.seqLength);
        input.createPastTensors(modelInfo, 0);

        input.updateInputIdsTensor(modelInfo);
        input.updatePositionIdsTensor(modelInfo);
        input.updateAttentionMaskTensor(modelInfo);
        
        input.bindInputs(modelInfo);

        // input.createOutputTensors(modelInfo, input.seqLength);
        input.createLogitsTensor(modelInfo, input.seqLength);
        input.createPresentTensors(modelInfo, input.seqLength);

        input.bindOutputs(modelInfo);
    }
    // else 
    // {
    //     input.createPresentTensors(modelInfo, 30);

    //     input.bindOutputs(modelInfo);
    // }

    // Ort::AllocatorWithDefaultOptions allocator;
    // for (size_t j = 0; j < session->GetInputCount(); j++)
    // {
    //     const auto& input_name = session->GetInputNameAllocated(j, allocator);
    //     auto type_info = session->GetInputTypeInfo(j);
    //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    //     std::cout << "Input " << input_name << " has shape: ";
    //     for (const auto dim : tensor_info.GetShape()) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << std::endl;
    // }

    #ifdef PRINT_TENSOR_UPDATE

    // std::cout << "Inputs:" << std::endl;
    // PrintTensorContent<int32_t>(input.inputIdsTensor);
    // PrintTensorContent<int32_t>(input.positionIdsTensor);
    // PrintTensorContent<int32_t>(input.attentionMaskTensor);

    // if (!input.presentTensors.empty())
    // {
    //     PrintTensorContent<int32_t>(input.presentTensors[0]);
    // }

    // if (!input.pastTensors.empty())
    // {
    //     PrintTensorContent<int32_t>(input.pastTensors[0]);
    // }

    #endif

    try 
    {
        // std::cout << "=============" << std::endl;
        session->Run(Ort::RunOptions{nullptr}, input.io_binding);


        // if (useIOBindings)
        // {

        // }
        // else 
        // {
        //     output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNamesCStr.data(), input.inputDataTensors.data(), input.inputDataTensors.size(), outputNames.data(), outputNames.size());
        // }
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }

    #ifdef PRINT_TENSOR_UPDATE

    // std::cout << "Outputs:" << std::endl;
    // PrintTensorContent<int32_t>(input.inputIdsTensor);
    // PrintTensorContent<int32_t>(input.positionIdsTensor);
    // PrintTensorContent<int32_t>(input.attentionMaskTensor);

    // if (!input.presentTensors.empty())
    // {
    //     PrintTensorContent<int32_t>(input.presentTensors[0]);
    // }

    // if (!input.pastTensors.empty())
    // {
    //     PrintTensorContent<int32_t>(input.pastTensors[0]);
    // }

    #endif


    // input.presentTensors.swap(std::move(input.pastTensors));

    // input.presentTensors.swap(input.pastTensors);

    // @TODO for each batch
    const Ort::Value& output_tensor = input.logitsTensor; // logits
    const float* output_data = output_tensor.GetTensorData<float>();
    Ort::TensorTypeAndShapeInfo tensorInfo = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensorInfo.GetShape();


    // for (int64_t batchIndex = 0; batchIndex < shape[0]; batchIndex++)
    // {
    //     // Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
    //     next_tokens = torch.argmax(next_token_logits, dim=-1)
    // }
    
    // next_token_logits = torch_output[0][:, -1, :]


    int64_t batchSize = shape[0];
    int64_t vocab_size = shape[2];


    // Greedy
    // @TODO : optimize with custom search
    
    // Get the last token's logits for each sequence in the batch
    std::vector<RunInstance::DataType> next_tokens(batchSize);
    for(int b = 0; b < batchSize; ++b) {
        // Pointer to the logits for the last token
        const float* last_logits = output_data + (b * shape[1] + (shape[1] - 1)) * vocab_size;
        
        // Find the index with the maximum logit
        float max_logit = last_logits[0];
        int max_index = 0;
        for(int v = 1; v < vocab_size; ++v) {
            if(last_logits[v] > max_logit) {
                max_logit = last_logits[v];
                max_index = v;
            }
        }
        next_tokens[b] = max_index;
    }

    for (int64_t b = 0; b < batchSize; ++b) 
    {
        DataType lastElem = input.batches[b]->positionIds.back();
        input.batches[b]->positionIds.clear();
        input.batches[b]->positionIds.push_back(lastElem + 1); // @TODO : Modulo? or already handled by the model?

        input.batches[b]->lastGeneratedToken = next_tokens[b];
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

    // std::int32_t batchIndex = 0;
    // for (auto& batch : input.batches)
    // {
    //     batch->push(next_tokens[batchIndex]);
    //     batchIndex++;
    // }
    // input.updateInputTensors(modelInfo);
    
    if (input.isFirstRun)
    {
        input.createInputIdsTensorCache(modelInfo);
        input.createPositionIdsTensorCache(modelInfo);
        input.createAttentionMaskTensor(modelInfo, 1);
        input.updateAttentionMaskTensorCache(modelInfo, 1);

        input.createLogitsTensor(modelInfo, 1);
        // input.createPresentTensors(modelInfo, 1);
    }

    if (input.pastSequenceLength < modelInfo.nbMaxPositions)
    {
        input.pastSequenceLength += 1;
    }


    if (input.seqLength < modelInfo.nbMaxPositions)
    {
        input.seqLength += 1;

        input.pastTensors = std::move(input.presentTensors);
        input.createPresentTensors(modelInfo, input.seqLength);




        // if (input.seqLength < modelInfo.nbMaxPositions)
        // {
        //     input.pastTensors = std::move(input.presentTensors);
        //     input.createPresentTensors(modelInfo, input.seqLength);
        // }
        // else
        // {
            
        // }
    }
    else
    {
        // @TODO : remove the oldest "dim" of input.pastTensors when reaching 512 to prevent overloading  
        // Copy previous "past" values

        // input.createPresentTensors(modelInfo, input.seqLength);
        // input.createPastTensors(modelInfo, input.seqLength-1);

        for (size_t i = 0; i < input.presentTensors.size(); i++)
        {
            const float* presentData = input.presentTensors[i].GetTensorData<float>();

            float* pastData = input.pastTensors[i].GetTensorMutableData<float>();

            auto tensor_info2 = input.presentTensors[i].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape2 = tensor_info2.GetShape();

            auto tensor_info = input.pastTensors[i].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();

            // size_t offset = shape.back();

            // for (int j = 0; j < shape[0]; j++)
            // {
            //     for (int64_t batch = 0; batch < shape[1]; ++batch) 
            //     {
            //         for (int64_t head = 0; head < shape[2]; ++head) 
            //         {
            //             for (int64_t seq = 0; seq < shape[3]; ++seq) // Skip first element 
            //             {
            //                 int64_t presentSeq = seq+1;
            //                 int64_t pastSeq = seq;
            //                 for (int64_t embed = 0; embed < shape[4]; ++embed) 
            //                 { 
            //                     int64_t presentId = embed 
            //                                     + presentSeq * shape[4] 
            //                                     + head * (shape[3]+1) * shape[4]
            //                                     + batch * shape[2] * (shape[3]+1) * shape[4]
            //                                     + j * shape[1] * shape[2] * (shape[3]+1) * shape[4];

            //                     int64_t pastId = embed 
            //                                     + pastSeq * shape[4] 
            //                                     + head * shape[3] * shape[4]
            //                                     + batch * shape[2] * shape[3] * shape[4]
            //                                     + j * shape[1] * shape[2] * shape[3] * shape[4];

            //                     pastData[pastId] = presentData[presentId];

            //                 // std::copy(
            //                 //     original_data + ((batch * shape[1] + head) * seq_len + seq) * elements_per_seq,
            //                 //     original_data + ((batch * shape[1] + head) * seq_len + seq + 1) * elements_per_seq,
            //                 //     sliced_data.begin() + ((batch * shape[1] + head) * (seq_len - 1) + (seq - 1)) * elements_per_seq
            //                 // );
            //                 }
            //             }
            //         }
            //     }
            // }

            std::vector<bool> r;
            r.resize(shape[0] * shape[1] * shape[2] * shape[3] * shape[4]);

            for (int j = 0; j < shape[0]; j++)
            {
                for (int64_t batch = 0; batch < shape[1]; ++batch) 
                {
                    for (int64_t head = 0; head < shape[2]; ++head) 
                    {
                        for (int64_t seq = 0; seq < shape[3]; ++seq) // Skip first element 
                        {
                            int64_t presentSeq = seq+1;
                            int64_t pastSeq = seq;
                            for (int64_t embed = 0; embed < shape[4]; ++embed) 
                            { 
                                int64_t presentId = embed 
                                                + presentSeq * shape[4] 
                                                + head * (shape[3]+1) * shape[4]
                                                + batch * shape[2] * (shape[3]+1) * shape[4]
                                                + j * shape[1] * shape[2] * (shape[3]+1) * shape[4];

                                int64_t pastId = embed 
                                                + pastSeq * shape[4] 
                                                + head * shape[3] * shape[4]
                                                + batch * shape[2] * shape[3] * shape[4]
                                                + j * shape[1] * shape[2] * shape[3] * shape[4];

                                pastData[pastId] = presentData[presentId];
                                r[pastId] = true;

                            // std::copy(
                            //     original_data + ((batch * shape[1] + head) * seq_len + seq) * elements_per_seq,
                            //     original_data + ((batch * shape[1] + head) * seq_len + seq + 1) * elements_per_seq,
                            //     sliced_data.begin() + ((batch * shape[1] + head) * (seq_len - 1) + (seq - 1)) * elements_per_seq
                            // );
                            }
                        }
                    }
                }
            }


            for (int i = 0; i < r.size(); i++)
            {
                assert(r[i]);
            }










            // for (int j = 0; j < shape[0]; ++j) {
            //     int64_t past_batch_offset_j = j * shape[1] * shape[2] * shape[3] * shape[4];
            //     int64_t present_batch_offset_j = j * shape[1] * shape[2] * (shape[3] + 1) * shape[4];

            //     for (int64_t batch = 0; batch < shape[1]; ++batch) {
            //         int64_t past_batch_offset = batch * shape[2] * shape[3] * shape[4] + past_batch_offset_j;
            //         int64_t present_batch_offset = batch * shape[2] * (shape[3] + 1) * shape[4] + present_batch_offset_j;

            //         for (int64_t head = 0; head < shape[2]; ++head) {
            //             int64_t past_head_offset = head * shape[3] * shape[4] + past_batch_offset;
            //             int64_t present_head_offset = head * (shape[3] + 1) * shape[4] + present_batch_offset;

            //             for (int64_t seq = 0; seq < shape[3]; ++seq) { // Skip first element
            //                 int64_t past_offset = seq * shape[4] + past_head_offset;
            //                 int64_t present_offset = (seq + 1) * shape[4] + present_head_offset;

            //                 // Use std::copy to handle entire embedding dimension at once
            //                 std::copy(
            //                     presentData + present_offset,
            //                     presentData + present_offset + shape[4],
            //                     pastData + past_offset
            //                 );
            //             }
            //         }
            //     }
            // }




            // int64_t batch_size = shape[0];
            // int64_t num_heads = shape[1];
            // int64_t seq_len = shape[2];
            // int64_t embedding_dim = shape[3];

            // // Calculate the offset to skip the first sequence element in presentData
            // int64_t offset = 1 * embedding_dim; // Skip the first element along sequence dim

            // for (int64_t x = 0; x < 2; ++x) {
            //     for (int64_t b = 0; b < batch_size; ++b) {
            //         for (int64_t h = 0; h < num_heads; ++h) {
            //             // Calculate the base address for the current batch and head
            //             float* past_ptr = pastData + ((x * batch_size + b) * num_heads + h) * seq_len * embedding_dim;
            //             const float* present_ptr = presentData + ((x * batch_size + b) * num_heads + h) * (seq_len + 1) * embedding_dim + offset;

            //             // Copy all elements from the current `present` (skipping first sequence element) to `past`
            //             std::copy(present_ptr, present_ptr + seq_len * embedding_dim, past_ptr);
            //         }
            //     }
            // }



        }
    }

    // else if (input.seqLength < modelInfo.nbMaxPositions)
    // else
    // {
    //     input.createPresentTensors(modelInfo, input.seqLength);
    // }

    input.updateInputIdsTensorCache(modelInfo, next_tokens);
    input.updatePositionIdsTensor(modelInfo); // @TODO : Cache version?

    input.bind(modelInfo);

    input.isFirstRun = false;
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