#pragma once

using MidiTokenizerHandle = void*;
using RedirectorHandle = void*;
using EnvHandle = void*;
using MusicGeneratorHandle = void*;

extern "C" 
{
    EnvHandle createEnv();
    void destroyEnv(EnvHandle handle);

    MidiTokenizerHandle createMidiTokenizer();
    void destroyMidiTokenizer(MidiTokenizerHandle handle);

    EnvHandle createMusicGenerator();
    void destroyMusicGenerator(MusicGeneratorHandle handle);


    RedirectorHandle createRedirector();
    void destroyRedirector(RedirectorHandle handle);

    void bindPitch(MidiTokenizerHandle handle, const char* prefix, void(*callback)(), void* data);





}



