#include "musicGenerator.hpp"

#include <sstream>


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

void RunInstance::updateInputTensors(const ModelInfo& info)
{
    bool usePast = !cachedValues.empty();

    // @TODO : support multiple batches
    assert(batches.size() == 1);

    auto pushTensorSingle = [&](DataType& data, std::int32_t index)
    {
        std::vector<int64_t> input_shape = {int64_t(batches.size()), 1};
        inputDataTensors.push_back(Ort::Value::CreateTensor<DataType>(memory_info, &data, sizeof(DataType), input_shape.data(), input_shape.size()));
    };

    auto pushTensor = [&](std::vector<DataType>& data, std::int32_t index)
    {
        std::vector<int64_t> input_shape = {int64_t(batches.size()), static_cast<int64_t>(batches.front()->size())};
        inputDataTensors.push_back(Ort::Value::CreateTensor<DataType>(memory_info, data.data(), data.size() * sizeof(DataType), input_shape.data(), input_shape.size()));
    };

    inputDataTensors.clear();

    if (usePast)
    {
        // @TODO : support multiple batches
        assert(batches.size() == 1);
        pushTensorSingle(batches.front()->inputIds.back(), 0);
        pushTensor(batches.front()->attentionMask, 1);
        pushTensorSingle(batches.front()->positionIds.back(), 2);
    }
    else 
    {
        // @TODO : support multiple batches
        assert(batches.size() == 1);
        pushTensor(batches.front()->inputIds, 0);
        pushTensor(batches.front()->attentionMask, 1);
        pushTensor(batches.front()->positionIds, 2);
    }


    if (!usePast)
    {
        // Update Past Tensors
        const std::array<int64_t, 5> past_shape = {
            2, int64_t(batches.size()), info.num_attention_heads, 0, info.hidden_size / info.num_attention_heads
        };

        for (int64_t i = 0; i < info.num_layer; i++)
        {
            Ort::Value past_tensor = Ort::Value::CreateTensor<float>(cacheAllocator, past_shape.data(), past_shape.size());
            inputDataTensors.push_back(std::move(past_tensor));
        }


        // size_t nbElements = 1;
        // for (int64_t v : past_shape)
        // {
        //     nbElements *= v;
        // }
        // std::vector<float> past;
        // for (size_t j = 0; j < nbElements; j++)
        //     past.push_back(0.0f);

        // std::vector<int64_t> past_shape_v(std::begin(past_shape), std::end(past_shape));
        // for (int64_t i = 0; i < info.num_layer; i++)
        // {
        //     inputDataTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, past.data(), past.size() * sizeof(float), past_shape_v.data(), past_shape_v.size()));
        // }
    }
    else
    {
        for (size_t i = 0; i < cachedValues.size(); i++)
        {
            inputDataTensors.emplace_back(std::move(cachedValues[i]));
        }
        cachedValues.clear();
    }
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

    // Labels
    modelInfo.inputIdLabel = "input_ids";
    modelInfo.attentionMaskLabel = "attention_mask";
    modelInfo.positionIdLabel = "position_ids";

    for (int64_t i = 0; i < modelInfo.num_layer; i++)
    {
        modelInfo.pastLabels.push_back(std::string("past_") + std::to_string(i));
    }
    
    std::vector<std::string> pastLabels;
}

void MusicGenerator::generate(RunInstance& input)
{
    input.updateInputTensors(modelInfo);

    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatorWithDefaultOptions> outputAllocators;
    outputAllocators.resize(1 + modelInfo.num_layer);
    std::vector<Ort::AllocatedStringPtr> outputAllocNames;
    for (std::int32_t i = 0; i < 1 + modelInfo.num_layer; i ++)
    {
        outputAllocNames.emplace_back(session->GetOutputNameAllocated(i, outputAllocators[i]));
        outputNames.push_back(outputAllocNames.back().get());
    }

    std::vector<const char*> inputNamesCStr;
    inputNamesCStr.push_back(modelInfo.inputIdLabel.c_str());
    inputNamesCStr.push_back(modelInfo.attentionMaskLabel.c_str());
    inputNamesCStr.push_back(modelInfo.positionIdLabel.c_str());

    for (auto& inputName : modelInfo.pastLabels)
    {
        inputNamesCStr.push_back(inputName.c_str());
    }

    std::vector<Ort::Value> output_tensors;
    try 
    {
        if (useIOBindings)
        {
            // Ort::IoBinding io_binding(*session);

            // // Initial binding, only done once
            // Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(input.memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());;
            // Ort::Value output_tensor = Ort::Value::CreateTensor<int64_t>(input.memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());;
            // io_binding.BindInput("input_ids", input_tensor);
            // io_binding.BindOutput("logits", output_tensor);
            // session->Run(Ort::RunOptions{nullptr}, io_binding);
        }
        else 
        {
            output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNamesCStr.data(), input.inputDataTensors.data(), input.inputDataTensors.size(), outputNames.data(), outputNames.size());
        }
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }

    input.cachedValues.clear();
    for (std::int32_t i = 1; i < 1+modelInfo.num_layer; i++)
    {
        // Ort::TensorTypeAndShapeInfo info = output_tensors[i].GetTensorTypeAndShapeInfo();
        // std::vector<int64_t> sh = info.GetShape();

        // std::cout << "Output Shape " << i << ":\n";
        // for (int j = 0; j < sh.size(); j++)
        //     std::cout << " - " << sh[j] << '\n';

        input.cachedValues.emplace_back(std::move(output_tensors[i]));
    }

    // @TODO for each batch
    const Ort::Value& output_tensor = output_tensors[0]; // logits
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

    std::int32_t batchIndex = 0;
    for (auto& batch : input.batches)
    {
        batch->push(next_tokens[batchIndex]);
        batchIndex++;
    }
    // input.updateInputTensors(modelInfo);
}


RunInstance MusicGenerator::generateInput(std::vector<RunInstance::DataType>&& inputTokens)
{
    RunInstance input;

    input.batches.emplace_back();
    input.batches.front()->set(inputTokens);
    input.updateInputTensors(modelInfo);

    return input;
}