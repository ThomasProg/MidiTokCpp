#include "gen.h"

#include "musicGenerator.hpp"
#include "redirector.hpp"
#include "midiTokenizer.hpp"

extern "C" 
{
EnvHandle createEnv(bool useLogging)
{
    if (useLogging)
        return new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "MusicGenerator");
    else
        return new Ort::Env();
}

void destroyEnv(EnvHandle env)
{
    delete env;
}

MidiTokenizerHandle createMidiTokenizer(const char* tokenizerPath)
{
    return new MidiTokenizer(tokenizerPath);
}
void destroyMidiTokenizer(MidiTokenizerHandle tokenizer)
{
    delete tokenizer;
}

MusicGeneratorHandle createMusicGenerator()
{
    return new MusicGenerator();
}
void destroyMusicGenerator(MusicGeneratorHandle musicGen)
{
    delete musicGen;
}

void generator_loadOnnxModel(MusicGeneratorHandle generator, EnvHandle env, const char* path)
{
    generator->loadOnnxModel(*env, path);
}

InputHandle generator_generateInput(MusicGeneratorHandle generator, int32_t* inputIDs, int32_t size)
{
    std::vector<Input::DataType> in;
    in.reserve(size);
    for (int32_t i = 0; i < size; i++)
    {
        in.push_back(inputIDs[i]);
    }
    return new Input(generator->generateInput(std::move(in)));
}
void generator_generateInput_free(InputHandle input)
{
    delete input;
}

void generator_generateNextToken(MusicGeneratorHandle generator, InputHandle input)
{
    generator->generate(*input);
}

RedirectorHandle createRedirector()
{
    return new Redirector();
}
void destroyRedirector(RedirectorHandle redirector)
{
    delete redirector;
}

void redirector_bindPitch(RedirectorHandle redirector, const MidiTokenizerHandle tokenizer, const char* prefix, void* data, void(*callback)(void*, std::uint8_t pitch))
{
    redirector->bindPitch(*tokenizer, prefix, callback, data);
}

bool redirector_call(RedirectorHandle redirector, int32_t token)
{
    return redirector->tryCall(token);
}

void input_decodeIDs(InputHandle input, MidiTokenizerHandle tokenizer, std::int32_t** outputIDs, std::int32_t* outSize)
{
    tokenizer_decodeIDs(tokenizer, input->inputData[0].data(), std::int32_t(input->inputData[0].size()), outputIDs, outSize);
}
void input_decodeIDs_free(std::int32_t* outputIDs)
{
    delete[] outputIDs;
}

void tokenizer_decodeIDs(MidiTokenizerHandle tokenizer, std::int32_t* inputIDs, std::int32_t size, std::int32_t** outputIDs, std::int32_t* outSize)
{
    std::vector<std::int32_t> inTokensVec;
    inTokensVec.reserve(size);
    for (std::int32_t i = 0; i < size; i++)
        inTokensVec.push_back(inputIDs[i]);

    std::vector<std::int32_t> outTokensVec;
    tokenizer->decodeIDs(inTokensVec, outTokensVec);

    *outSize = std::int32_t(outTokensVec.size());
    *outputIDs = new std::int32_t[*outSize]();

    for (std::int32_t i = 0; i < outTokensVec.size(); i++)
    {
        (*outputIDs)[i] = outTokensVec[i];
    }
}

void tokenizer_decodeIDs_free(std::int32_t* outputIDs)
{
    delete[] outputIDs;
}













}
