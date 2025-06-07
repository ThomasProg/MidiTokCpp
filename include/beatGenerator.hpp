#pragma once

#include "fwd.h"
#include "beatGenerator.h"
#include <vector>

class GenerationHistory;

class Bar
{
    int nbTickStart = 0;
    int nbTickEnd = 0;

    int nbBeatsPerBar = 4;


    int ticksPerBar = 32;
    int ticksPerBeat = 8;

public:
    Bar() : nbTickEnd(ticksPerBar-1)
    {

    }
    void next()
    {
        nbTickStart += ticksPerBar;
        nbTickEnd += ticksPerBar;
    }
    void prev()
    {
        nbTickStart -= ticksPerBar;
        nbTickEnd -= ticksPerBar;
    }

    int getStartTick() const
    {
        return nbTickStart;
    }

    int getEndTick() const
    {
        return nbTickEnd;
    }

    // float getEndBeat() const
    // {
    //     return float(nbBeatsPerBar);
    // }
};

class BeatGenerator
{
private:
    int ticksPerBar = 32;
    int ticksPerBeat = 8;

    std::vector<BeatNote> notes;
    Bar bar;

public:
    void addKick(const Bar& bar, int tick);
    void addSnare(const Bar& bar, int tick);
    void addClosedHat(const Bar& bar, int tick);
    void addGhostSnare(const Bar& bar, int tick);
    void addFill(const Bar& bar, int tick);

    void rewind(int32_t tick);
    void refresh(const Note* melodyBegin, const Note* melodyEnd);

    int beatToTick(float beat) const
    {
        return bar.getStartTick() + lround(beat * ticksPerBeat);
    }

    const std::vector<BeatNote>& getNotes() const
    {
        return notes;
    }
};