// based on compTest.py from MidiTokMusicGen

#include "midiTokenizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_cxx_api.h>

#include "gen.h"
#include "musicGenerator.hpp"
#include "redirector.hpp"

int main()
{
    
    std::cout << "Current working directory: " <<  WORKSPACE_PATH << std::endl;

    std::unique_ptr<MidiTokenizer> tokenizer = std::make_unique<MidiTokenizer>(WORKSPACE_PATH "/tokenizer.json");
    // MidiTokenizer* tokenizer = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");

    std::unique_ptr<Ort::Env> env = MusicGenerator::createOnnxEnv();

    MusicGenerator generator;
    generator.loadOnnxModel(*env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx");


    Input::DataType input_ids[] = {
        942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
        12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
        2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
    };

    Input input = generator.generateInput(std::vector<Input::DataType>(std::begin(input_ids), std::end(input_ids)));

    for (int i = 0; i < 1; i++)
    {
        generator.generate(input);
    }

    // for (auto& v : input.inputData[0])
    // {
    //     std::cout << v << '\t';

    // }


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
    tokenizer->decodeIDs(input.inputData[0], outTokens);

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

    // tokenizer->decode(input.inputData[0]);

}