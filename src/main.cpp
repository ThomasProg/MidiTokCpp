// based on compTest.py from MidiTokMusicGen

#include "midiTokenizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_cxx_api.h>

#include <sstream>
std::wstring widen( const std::string& str )
{
    std::wostringstream wstm ;
    const std::ctype<wchar_t>& ctfacet = std::use_facet<std::ctype<wchar_t>>(wstm.getloc()) ;
    for( size_t i=0 ; i<str.size() ; ++i ) 
              wstm << ctfacet.widen( str[i] ) ;
    return wstm.str() ;
}

struct Input
{
    using DataType = int32_t;

    std::vector<std::string> inputNames;

    // [type, value]
    std::vector<std::vector<DataType>> inputData;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputDataTensors;
};

class MusicGenerator
{
public:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<MidiTokenizer> tokenizer;

public:
    void createOnnxEnv();
    void loadOnnxModel(const std::string& modelPath);
    void loadTokenizerModel(const std::string& tokenizerPath);

    void generate(Input& input);


};

void MusicGenerator::createOnnxEnv()
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "MusicGenerator");
}

void MusicGenerator::loadOnnxModel(const std::string& modelPath)
{
    // Create session options and enable optimization
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try 
    {
        session = std::make_unique<Ort::Session>(*env.get(), widen(modelPath).c_str(), session_options);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }
}

void MusicGenerator::loadTokenizerModel(const std::string& tokenizerPath)
{
    tokenizer = std::make_unique<MidiTokenizer>(tokenizerPath);
}

void updateInputTensors(Input& input)
{
    int64_t nbBatches = 1;
    std::vector<int64_t> input_shape = {nbBatches, static_cast<int64_t>(input.inputData.front().size())};
    for (size_t i = 0; i < input.inputData.size(); i++)
    {
        std::vector<Input::DataType>& inputData = input.inputData[i];
        auto createTensor = [&]()
        {
            return Ort::Value::CreateTensor<Input::DataType>(input.memory_info, inputData.data(), inputData.size() * sizeof(Input::DataType), input_shape.data(), input_shape.size());
        };
        if (i < input.inputDataTensors.size())
            input.inputDataTensors[i] = createTensor();
        else
            input.inputDataTensors.push_back(createTensor());
    }
}


void MusicGenerator::generate(Input& input)
{
    Ort::AllocatorWithDefaultOptions outputAllocator;
    Ort::AllocatedStringPtr outputStr = session->GetOutputNameAllocated(0, outputAllocator);
    const char* output_names[] = {outputStr.get()};

    std::vector<const char*> inputNamesCStr;
    for (auto& inputName : input.inputNames)
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
    std::vector<Input::DataType> next_tokens(batchSize);
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

    // Append tokens to all_token_ids
    for(Input::DataType b = 0; b < batchSize; ++b) {
        // input.inputData[b].push_back(next_tokens[b]);

        // @TODO : batch support
        input.inputData[0].push_back(next_tokens[b]);
        input.inputData[1].push_back(Input::DataType(input.inputData[0].size()));
        input.inputData[2].push_back(1);
    }


    updateInputTensors(input);
}

int main()
{

    // TODO : load from config
    int64_t num_attention_heads = 8;
    int64_t hidden_size = 512;
    int64_t num_layer = 8;

    int64_t batchSize = 1;




    MusicGenerator generator;
    
    std::cout << "Current working directory: " <<  WORKSPACE_PATH << std::endl;

    generator.createOnnxEnv();

    // generator.loadOnnxModel("C:/Users/thoma/Documents/Unreal Projects/MIDITokCpp/onnx/model.onnx");
    generator.loadOnnxModel(WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx");

    generator.loadTokenizerModel(WORKSPACE_PATH "/tokenizer.json");



    Input input;
    input.inputNames.push_back("input_ids");
    input.inputNames.push_back("position_ids");
    input.inputNames.push_back("attention_mask");

    Input::DataType input_ids[] = {
        942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
        12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
        2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
    };
    input.inputData.emplace_back(std::begin(input_ids), std::end(input_ids));
    input.inputData.emplace_back();
    input.inputData.emplace_back();

    const std::vector<Input::DataType>& input_ids_v_rf = input.inputData[0];
    std::vector<Input::DataType>& position_ids_v_rf = input.inputData[1];
    std::vector<Input::DataType>& attention_mask_v_rf = input.inputData[2];

    for (size_t i = 0; i < input_ids_v_rf.size(); i++)
    {
        position_ids_v_rf.push_back(Input::DataType(i));
        attention_mask_v_rf.push_back(Input::DataType(1));
    }

    // Memory info for the input tensor
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<Ort::Value> inputDataTensors;

    int64_t nbBatches = 1;
    // std::vector<int64_t> input_shape = {nbBatches, static_cast<int64_t>(input.inputData.front().size())};
    // for (size_t i = 0; i < input.inputData.size(); i++)
    // {
    //     std::vector<Input::DataType>& inputData = input.inputData[i];
    //     input.inputDataTensors.push_back(Ort::Value::CreateTensor<Input::DataType>(input.memory_info, inputData.data(), inputData.size() * sizeof(Input::DataType), input_shape.data(), input_shape.size()));
    // }

    updateInputTensors(input);


    // // Memory info for the input tensor
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<Ort::Value> inputDataTensors;

    // int64_t nbBatches = 1;
    // std::vector<int64_t> input_shape = {nbBatches, static_cast<int64_t>(input.inputData.front().size())};
    // for (std::vector<Input::DataType>& inputData : input.inputData)
    // {
    //     inputDataTensors.push_back(Ort::Value::CreateTensor<Input::DataType>(memory_info, inputData.data(), inputData.size() * sizeof(Input::DataType), input_shape.data(), input_shape.size()));
    // }






    // Update Past
    int64_t past_shape[] = {
        2, batchSize, num_attention_heads, 0, hidden_size / num_attention_heads
    };
    size_t nbElements = 1;
    for (int64_t v : past_shape)
    {
        nbElements *= v;
    }

    std::vector<int64_t> past_shape_v(std::begin(past_shape), std::end(past_shape));
    for (int64_t i = 0; i < num_layer; i++)
    {
        input.inputNames.push_back(std::string("past_") + std::to_string(i));
        std::vector<float> past;
        const std::vector<Input::DataType>& input_ids_v_rf2 = input.inputData[0];
        for (size_t j = 0; j < nbElements; j++)
            past.push_back(0.0f);

        input.inputDataTensors.push_back(Ort::Value::CreateTensor<float>(input.memory_info, past.data(), past.size() * sizeof(float), past_shape_v.data(), past_shape_v.size()));
    }



    for (int i = 0; i < 100; i++)
    {
        generator.generate(input);
    }

    for (auto& v : input.inputData[0])
    {
        std::cout << v << '\t';

    }

    // // displayModelInputsInfo(*session.get());


    // // Set sequence generation parameters
    // const int max_sequence_length = 1024;   // Set a max length to prevent infinite loop
    // // const int64_t end_token_id = 102;     // Example end token ID
    // // const int64_t start_token_id = 101;   // Example start token ID


    // Score baseScore = Score();
    // // baseScore.push_back(0);

    // // Initial input: start token
    // std::vector<int64_t> input_data = tokenizer.encode(baseScore);
    // input_data.push_back(942);

    // std::vector<int64_t> output_data;
    
    // try 
    // {
    //     output_data = generate(*session.get(), input_data, max_sequence_length);
    // }
    // catch(const Ort::Exception& e)
    // {
    //     std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
    //     std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
    //     exit(1);
    // }


    // // Print the generated sequence
    // std::cout << "Generated sequence: ";
    // for (const auto& token : output_data) {
    //     std::cout << token << " ";
    // }
    // std::cout << std::endl;

    // Score newScore = tokenizer.decode(output_data);

    // // // Cleanup dynamically allocated memory for input/output names
    // // Ort::AllocatorWithDefaultOptions().Free((void*)input_name);
    // // Ort::AllocatorWithDefaultOptions().Free((void*)output_name);

}