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

    for (std::int32_t i = fromPos; i < positionIds.size() + fromPos; i++)
    {
        positionIds[i] = i;
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
    // @TODO : support multiple batches
    assert(batches.size() == 1);
    std::vector<int64_t> input_shape = {int64_t(batches.size()), static_cast<int64_t>(batches.front()->size())};

    auto pushTensor = [&](std::vector<DataType>& data, std::int32_t index)
    {
        inputDataTensors.push_back(Ort::Value::CreateTensor<DataType>(memory_info, data.data(), data.size() * sizeof(DataType), input_shape.data(), input_shape.size()));
    };

    inputDataTensors.clear();

    // @TODO : support multiple batches
    assert(batches.size() == 1);
    pushTensor(batches.front()->inputIds, 0);
    pushTensor(batches.front()->attentionMask, 1);
    pushTensor(batches.front()->positionIds, 2);





    // Update Past Tensors
    int64_t past_shape[] = {
        2, int64_t(batches.size()), info.num_attention_heads, 0, info.hidden_size / info.num_attention_heads
    };
    size_t nbElements = 1;
    for (int64_t v : past_shape)
    {
        nbElements *= v;
    }
    std::vector<float> past;
    for (size_t j = 0; j < nbElements; j++)
        past.push_back(0.0f);

    std::vector<int64_t> past_shape_v(std::begin(past_shape), std::end(past_shape));
    for (int64_t i = 0; i < info.num_layer; i++)
    {
        inputDataTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, past.data(), past.size() * sizeof(float), past_shape_v.data(), past_shape_v.size()));
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

    modelInfo.num_attention_heads = 8;
    modelInfo.hidden_size = 512;
    modelInfo.num_layer = 8;

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

    Ort::AllocatorWithDefaultOptions outputAllocator;
    Ort::AllocatedStringPtr outputStr = session->GetOutputNameAllocated(0, outputAllocator);
    const char* output_names[] = {outputStr.get()};

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
        output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNamesCStr.data(), input.inputDataTensors.data(), input.inputDataTensors.size(), output_names, 1);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }

    // @TODO for each batch
    const Ort::Value& output_tensor = output_tensors[0];
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