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

RedirectorHandle createRedirector()
{
    return new Redirector();
}
void destroyRedirector(RedirectorHandle redirector)
{
    delete redirector;
}

void redirector_bindPitch(RedirectorHandle redirector, const MidiTokenizerHandle tokenizer, const char* prefix, void* data, void(*callback)(void*, unsigned char/*uint8*/ pitch))
{
    redirector->bindPitch(*tokenizer, prefix, callback, data);
}
}
