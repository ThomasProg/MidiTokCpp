#include "midiTokenizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_cxx_api.h>

const char* input_name = "input_ids";//session.GetInputName(0, Ort::AllocatorWithDefaultOptions());
const char* output_name = "outputs";//session.GetOutputName(0, Ort::AllocatorWithDefaultOptions());

std::vector<int64_t> generate(Ort::Session& session, std::vector<int64_t> input_data, const int max_sequence_length)
{
    std::vector<int64_t> generated_sequence = {};  // Start sequence

    // Memory info for the input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Get model input/output names
    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    // Generate sequence iteratively
    for (int step = 1; step < max_sequence_length; step++) {
        // Prepare input shape: [batch_size, sequence_length]
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};

        // Create input tensor with the current sequence
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_data.data(), input_data.size() * sizeof(int64_t), input_shape.data(), 2);

        // Run the model
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto& output_tensor = output_tensors[0];

        // Get the predicted token (assuming model output is token ID)
        int64_t* output_data = output_tensor.GetTensorMutableData<int64_t>();
        int64_t predicted_token_id = output_data[0];  // Get the next predicted token

        // Append predicted token to the sequence
        generated_sequence.push_back(predicted_token_id);
        input_data.push_back(predicted_token_id);  // Extend input for next iteration

        // // Break the loop if the end token is generated
        // if (predicted_token_id == end_token_id) {
        //     break;
        // }
    }

    return generated_sequence;
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
    const std::string model_path = "transformer_model.onnx";  // Path to your ONNX model
    const ORTCHAR_T* modelPathStr = (const ORTCHAR_T*) model_path.c_str();

    std::unique_ptr<Ort::Session> session;
    try 
    {
        session = std::make_unique<Ort::Session>(env, modelPathStr, session_options);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }

    // Set sequence generation parameters
    const int max_sequence_length = 20;   // Set a max length to prevent infinite loop
    // const int64_t end_token_id = 102;     // Example end token ID
    // const int64_t start_token_id = 101;   // Example start token ID


    Score baseScore = Score();

    // Initial input: start token
    std::vector<int64_t> input_data = tokenizer.encode(baseScore);

    std::vector<int64_t> output_data = generate(*session.get(), input_data, max_sequence_length);

    // Print the generated sequence
    std::cout << "Generated sequence: ";
    for (const auto& token : output_data) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    Score newScore = tokenizer.decode(output_data);

    // Cleanup dynamically allocated memory for input/output names
    Ort::AllocatorWithDefaultOptions().Free((void*)input_name);
    Ort::AllocatorWithDefaultOptions().Free((void*)output_name);

}