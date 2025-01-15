// based on compTest.py from MidiTokMusicGen

#include "midiTokenizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_cxx_api.h>

#include "gen.h"
#include "musicGenerator.hpp"
#include "redirector.hpp"
#include "midiConverter.hpp"

#include <algorithm>

#include <sstream>
std::wstring widen2( const std::string& str )
{
    std::wostringstream wstm ;
    const std::ctype<wchar_t>& ctfacet = std::use_facet<std::ctype<wchar_t>>(wstm.getloc()) ;
    for( size_t i=0 ; i<str.size() ; ++i ) 
              wstm << ctfacet.widen( str[i] ) ;
    return wstm.str() ;
}

void rawGenTest()
{

    std::unique_ptr<Ort::Env> env = MusicGenerator::createOnnxEnv();
    std::unique_ptr<Ort::Session> session;

    // Create session options and enable optimization
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try 
    {
        const char* modelPath = WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx";
        session = std::make_unique<Ort::Session>(*env.get(), widen2(modelPath).c_str(), session_options);
    }
    catch(const Ort::Exception& e)
    {
        std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
        exit(1);
    }


    std::vector<Ort::Value> output_tensors;
    // try 
    {
        // @TODO : load from config
        int64_t num_attention_heads = 8;
        int64_t hidden_size = 512;
        int64_t num_layer = 8;
        
        int64_t batchSize = 1;

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



        std::vector<std::string> inputNames;
        
        inputNames.push_back("input_ids");
        inputNames.push_back("attention_mask");
        inputNames.push_back("position_ids");

        for (int64_t i = 0; i < num_layer; i++)
        {
            inputNames.push_back(std::string("past_") + std::to_string(i));
        }

        std::vector<const char*> inputNamesCStr;
        for (auto& inputName : inputNames)
        {
            inputNamesCStr.push_back(inputName.c_str());
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


        std::vector<int32_t> inputIds = 
        {
            0, 314, 372, 2833, 349, 216, 1530, 295,
        };
        std::vector<int32_t> positionIds =
        {
            0, 1, 2, 3, 4, 5, 6, 7
        };
        std::vector<int32_t> attentionMask = 
        {
            1, 1, 1, 1, 1, 1, 1, 1
        };

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(8)};

        std::vector<Ort::Value> inputDataTensors;
        inputDataTensors.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, inputIds.data(), inputIds.size() * sizeof(int32_t), input_shape.data(), input_shape.size()));
        inputDataTensors.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, positionIds.data(), positionIds.size() * sizeof(int32_t), input_shape.data(), input_shape.size()));
        inputDataTensors.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, attentionMask.data(), attentionMask.size() * sizeof(int32_t), input_shape.data(), input_shape.size()));

        std::vector<float> past;
        for (size_t j = 0; j < nbElements; j++)
            past.push_back(0.0f);

        for (int64_t i = 0; i < num_layer; i++)
        {
            inputDataTensors.push_back(Ort::Value::CreateTensor<float>(memory_info, past.data(), past.size() * sizeof(float), past_shape_v.data(), past_shape_v.size()));
        }

        Ort::AllocatorWithDefaultOptions outputAllocator;
        Ort::AllocatedStringPtr outputStr = session->GetOutputNameAllocated(0, outputAllocator);
        const char* output_names[] = {outputStr.get()};
        output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNamesCStr.data(), inputDataTensors.data(), inputDataTensors.size(), output_names, 1);
    
        const Ort::Value& output_tensor = output_tensors[0];
        const float* output_data = output_tensor.GetTensorData<float>();
        Ort::TensorTypeAndShapeInfo tensorInfo = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();

        int64_t sum = 0;
        for (int64_t t : shape)
        {
            sum += t;
        }

        std::vector<float> v;
        for (int64_t i = 0; i < sum; i++)
            v.push_back(output_data[i]);

        for (float v2 : v)
            std::cout << v2 << '\t';
        
        
    }
    // catch(const Ort::Exception& e)
    // {
    //     std::cout << "Error occurred: " << e.what() << std::endl;           // Error message
    //     std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;    // Error code
    //     exit(1);
    // }




}

#include "range.hpp"
void runRangeTest()
{
    RangeGroup group;
    group.addRange({0, 10});
    group.addRange({8, 15});
    group.addRange({17, 19});
    group.addRange({19, 25});
    group.addRange({26, 29});

    group.addRange({16, 16});

    group.addRange({-4, -1});

    const std::vector<Range>& ranges = group.getRanges();
    for (const Range& range : ranges)
    {
        std::cout << range.min << " - " << range.max << std::endl;
    }
}

int main()
{
    runRangeTest();
    return 0;

    // rawGenTest();
    // return 0;
    
    std::cout << "Current working directory: " <<  WORKSPACE_PATH << std::endl;

    std::unique_ptr<MidiTokenizer> tokenizer = std::make_unique<MidiTokenizer>(WORKSPACE_PATH "/tokenizer.json");
    // MidiTokenizer* tokenizer = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");

    std::unique_ptr<Ort::Env> env = MusicGenerator::createOnnxEnv();

    MusicGenerator generator;
    generator.loadOnnxModel(*env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model_past.onnx");


    RunInstance::DataType input_ids[] = {
        942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
        12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
        2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
    };

    RunInstance input = generator.generateInput(std::vector<RunInstance::DataType>(std::begin(input_ids), std::end(input_ids)));

    for (int i = 0; i < 50; i++)
    {
        const char* outError;
        generator.generate(input, outError);
    }

    std::cout << "Out Encoded Tokens" << std::endl;
    for (auto& v : input.batches[0]->inputIds)
    {
        std::cout << v << '\t';
    }
    std::cout << '\n';

    // return 0;

    // for (auto [k, v] : tokenizer->GetVocabBase())
    // {
    //     std::cout << k << std::endl;
    // }

    Redirector redirector;

    redirector.bindPitch(*tokenizer, "Pitch_", [](void*, std::uint8_t pitch)
    {
        std::cout << "Pitch : " << int(pitch) << std::endl;
    });

    redirector.bindBar(*tokenizer, "Bar_", [](void*, std::uint8_t barIndex, bool isBarNone)
    {
        if (isBarNone)
            std::cout << "Bar : None" << std::endl;
        else
            std::cout << "Bar : " << int(barIndex) << std::endl;
    });

    redirector.bindPosition(*tokenizer, "Position_", [](void*, std::uint8_t position)
    {
        std::cout << "Position : " << int(position) << std::endl;
    });


    std::vector<int32_t> outTokens;
    tokenizer->decodeIDs(input.batches[0]->inputIds, outTokens);

    std::cout << "Out Decoded Tokens" << std::endl;
    for (auto& v : outTokens)
    {
        std::cout << v << '\t';
    }
    std::cout << '\n';

    for (int32_t token : outTokens)
    {
        try 
        {
            redirector.tryCall(token);
        } 
        catch(const std::exception&)
        {

        }


    }






    std::cout << "======= MIDI Converter =========" << std::endl;




    std::int32_t UnplayedTokenIndex = 0;
    std::int32_t i = UnplayedTokenIndex;

    struct Args
    {
        std::int32_t& i;
        std::int32_t& unplayedTokenIndex;
    };

    Args args{ i, UnplayedTokenIndex};

    REMIConverter converter;
    converter.onNote = [](void* data, const Note& newNote)
    {
        Args& args = *(Args*)(data);

        std::cout << newNote.pitch << std::endl;
    };

    converter.tokenizerHandle = tokenizer.get();

    while (i < outTokens.size())
    {
        converter.MIDIConverter::processToken(outTokens, i, &args);
        // converterProcessToken(converter, outTokens, i, &args);
        i++;
    }

    // tokenizer->decode(input.inputData[0]);

}