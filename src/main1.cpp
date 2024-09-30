#include "midiTokenizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_cxx_api.h>

// Function to generate position_ids
std::vector<int64_t> generate_position_ids(int64_t past_length, int64_t sequence_length) {
    std::vector<int64_t> position_ids(sequence_length);
    for (int64_t i = 0; i < sequence_length; ++i) {
        position_ids[i] = past_length + i;
    }
    return position_ids;
}

void displayModelInputsInfo(Ort::Session& session)
{
    // Get the number of inputs
    size_t num_inputs = session.GetInputCount();

    // Loop through each input and get its expected shape
    for (size_t i = 0; i < num_inputs; i++) {
        // Get input name (optional, for reference)
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " name: " << input_name.get() << std::endl;

        // Get the input type info
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

        // Get the input shape
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        // Print the expected shape
        std::cout << "Expected shape for input " << i << ": ";
        for (auto dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<int64_t> generate(Ort::Session& session, std::vector<int64_t> input_data, const int max_sequence_length)
{
    std::vector<int64_t> generated_sequence = {};  // Start sequence

    // Memory info for the input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Ort::AllocatorWithDefaultOptions inputAllocator;
    Ort::AllocatorWithDefaultOptions outputAllocator;
    Ort::AllocatedStringPtr outputStr = session.GetOutputNameAllocated(0, outputAllocator);

    for (int i = 0; i < session.GetInputCount(); i++)
    {
        Ort::AllocatorWithDefaultOptions inputAllocator;
        Ort::AllocatedStringPtr inputStr = session.GetInputNameAllocated(i, inputAllocator);
        std::cout << inputStr.get() << std::endl;
    }

    // Get model input/output names
    const char* input_names[] = {"input_ids", "position_ids", "attention_mask"};
    const char* output_names[] = {outputStr.get()};

    size_t inputSize = input_data.size();
    for (int i = inputSize; i < max_sequence_length; i++)
    {
        input_data.push_back(0);
    }


    std::vector<int64_t> attentionIds;
    attentionIds.resize(input_data.size());
    for (int i = 0; i < inputSize; i++)
    {
        attentionIds[i] = 1;
    }


    std::vector<int64_t> attention_shape = {1, static_cast<int64_t>(attentionIds.size())};
    


    // Generate sequence iteratively
    for (int step = 0; step < max_sequence_length; step++) {
        // Prepare input shape: [batch_size, sequence_length]
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};

        std::vector<int64_t> posIds = generate_position_ids(0, input_data.size());
        std::vector<int64_t> pos_shape = {1, static_cast<int64_t>(posIds.size())};
        

        Ort::Value inputs[] = {
            Ort::Value::CreateTensor<int64_t>(memory_info, input_data.data(), input_data.size() * sizeof(int64_t), input_shape.data(), 2), 
            Ort::Value::CreateTensor<int64_t>(memory_info, posIds.data(), posIds.size() * sizeof(int64_t), pos_shape.data(), 2),
            Ort::Value::CreateTensor<int64_t>(memory_info, attentionIds.data(), attentionIds.size() * sizeof(int64_t), attention_shape.data(), 2)
            };

        // Run the model
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, inputs, 3, output_names, 1);
        auto& output_tensor = output_tensors[0];

        // Get the predicted token (assuming model output is token ID)
        int64_t* output_data = output_tensor.GetTensorMutableData<int64_t>();
        int64_t predicted_token_id = output_data[0];  // Get the next predicted token

        // Append predicted token to the sequence
        generated_sequence.push_back(predicted_token_id);
        // input_data.push_back(predicted_token_id);  // Extend input for next iteration
        input_data[inputSize] = predicted_token_id;
        attentionIds[inputSize] = 1;
        inputSize += 1; 

        // // Break the loop if the end token is generated
        // if (predicted_token_id == end_token_id) {
        //     break;
        // }
    }

    return generated_sequence;
}


#include <sstream>
std::wstring widen( const std::string& str )
{
    std::wostringstream wstm ;
    const std::ctype<wchar_t>& ctfacet = std::use_facet<std::ctype<wchar_t>>(wstm.getloc()) ;
    for( size_t i=0 ; i<str.size() ; ++i ) 
              wstm << ctfacet.widen( str[i] ) ;
    return wstm.str() ;
}




int main()
{

    std::cout << "Current working directory: " <<  WORKSPACE_PATH "t.json" << std::endl;
    // MidiTokenizer tokenizer("tokenizer.json");
    MidiTokenizer tokenizer(WORKSPACE_PATH "/tokenizer.json");




    // Initialize ONNX Runtime environment
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sequence_generation");
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "sequence_generation");

    // Create session options and enable optimization
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load the model
    const std::string model_path = "C:/Users/thoma/Documents/Unreal Projects/MIDITokCpp/onnx/model.onnx";  // Path to your ONNX model
    // const ORTCHAR_T* modelPathStr = (const ORTCHAR_T*) model_path.c_str();

    std::unique_ptr<Ort::Session> session;
    try 
    {
        session = std::make_unique<Ort::Session>(env, widen(model_path).c_str(), session_options);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }


    // displayModelInputsInfo(*session.get());


    // Set sequence generation parameters
    const int max_sequence_length = 1024;   // Set a max length to prevent infinite loop
    // const int64_t end_token_id = 102;     // Example end token ID
    // const int64_t start_token_id = 101;   // Example start token ID


    Score baseScore = Score();
    // baseScore.push_back(0);

    // Initial input: start token
    std::vector<int64_t> input_data = tokenizer.encode(baseScore);
    input_data.push_back(942);

    std::vector<int64_t> output_data;
    
    try 
    {
        output_data = generate(*session.get(), input_data, max_sequence_length);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }


    // Print the generated sequence
    std::cout << "Generated sequence: ";
    for (const auto& token : output_data) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    Score newScore = tokenizer.decode(output_data);

    // // Cleanup dynamically allocated memory for input/output names
    // Ort::AllocatorWithDefaultOptions().Free((void*)input_name);
    // Ort::AllocatorWithDefaultOptions().Free((void*)output_name);

}