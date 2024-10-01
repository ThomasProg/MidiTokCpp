// based on compTest.py from MidiTokMusicGen

#include "midiTokenizer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_cxx_api.h>

#include "musicGenerator.h"

class Redirector
{
    struct Callback
    {
        void* data;
        std::function<void(void*)> callback;

        void call()
        {
            callback(data);
        }
    };

    std::map<int32_t, Callback> tokenToCallback;

public:
    void call(int32_t token)
    {
        tokenToCallback.at(token).call();
    }

    void bindPitch(const MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t pitch), void* data = nullptr)
    {
        const std::map<std::string, int32_t>& vocab = tokenizer.GetVocabBase();

        for (auto& [key, value] : vocab)
        {
            if (key.size() >= newKey.size() && key.compare(0, newKey.size(), newKey) == 0)
            {
                std::string args = key.substr(newKey.size());

                // Since it's bindPitch, we know args is just a single int
                std::uint8_t arg = std::uint8_t(std::stoi(args));

                Callback callback;
                callback.data = data;
                callback.callback = [newCallback, arg](void* data)
                {
                    newCallback(data, arg);
                };

                tokenToCallback[value] = callback;
            }
        }
    }
};

int main()
{
    
    std::cout << "Current working directory: " <<  WORKSPACE_PATH << std::endl;

    std::unique_ptr<MidiTokenizer> tokenizer = std::make_unique<MidiTokenizer>(WORKSPACE_PATH "/tokenizer.json");

    std::unique_ptr<Ort::Env> env = MusicGenerator::createOnnxEnv();

    MusicGenerator generator;
    generator.loadOnnxModel(*env.get(), WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx");


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

    for (auto& v : input.inputData[0])
    {
        std::cout << v << '\t';

    }

    Redirector redirector;

    redirector.bindPitch(*tokenizer.get(), "Pitch_", [](void*, std::uint8_t pitch)
    {
        std::cout << "Pitch : " << int(pitch) << std::endl;
    });



    std::vector<int32_t> outTokens;
    tokenizer->decodeIDs(input.inputData[0], outTokens);

    for (int32_t token : outTokens)
    {
        try 
        {
            redirector.call(token);
        } 
        catch(const std::exception&)
        {

        }


    }

    // tokenizer->decode(input.inputData[0]);
}