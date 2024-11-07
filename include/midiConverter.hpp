#pragma once

#include "fwd.h"
#include <cstdint>
#include <vector>
#include "note.h"

class MIDIConverter
{
public:
    MidiTokenizerHandle tokenizerHandle;
    RedirectorHandle redirector;

    std::int32_t currentTick = 0;
    std::int32_t tickAtCurrentBar = 0;
    std::int32_t currentBar = -1;

    std::int32_t ticksPerBar = 32;
    std::int32_t ticksPerBeat = 8;
    std::int32_t ticksPerPos = 1;


    void (*onNote)(void* data, const Note&);

    void reset();
    bool processToken(const int32_t* tokens, int32_t nbTokens, std::int32_t& index, void* data = nullptr);
    bool processToken(const std::vector<int32_t>& tokens, std::int32_t& index, void* data = nullptr);
};

