#include "beatGenerator.hpp"
#include "beatGenerator.h"
#include "generationHistory.hpp"
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dist(0.0, 1.0);

void BeatGenerator::addKick(const Bar& bar, int tick)
{
    Note note;
    note.duration = 8;
    note.tick = tick;
    note.pitch = 36;
    note.velocity = 90;
    notes.emplace_back(BeatType::KICK, note);
}

void BeatGenerator::addSnare(const Bar& bar, int tick)
{
    Note note;
    note.duration = 8;
    note.tick = tick;
    note.pitch = 38;
    note.velocity = 90;
    notes.emplace_back(BeatType::SNARE, note);
}
void BeatGenerator::addClosedHat(const Bar& bar, int tick)
{
    Note note;
    note.duration = 8;
    note.tick = tick;
    note.pitch = 42;
    note.velocity = 90;
    notes.emplace_back(BeatType::HI_HAT, note);
}
void BeatGenerator::addGhostSnare(const Bar& bar, int tick)
{

}
void BeatGenerator::addFill(const Bar& bar, int tick)
{

}

void BeatGenerator::rewind(int32_t tick)
{
    while (bar.getStartTick() > tick)
    {
        bar.prev();
    }

    int32_t barStartTick = bar.getStartTick();

    if (notes.empty())
    {
        return;
    }

    size_t index = notes.size();
    while (index > 0 && notes[index-1].note.tick >= barStartTick)
    {
        index--;
    }
    
    if (index >= notes.size())
    {
        return;
    }
    notes.erase(notes.begin() + index, notes.end());
}

void BeatGenerator::refresh(const Note* melodyBegin, const Note* melodyEnd)
{  
    if (melodyBegin == melodyEnd)
    {
        return;
    }

    // Remove all notes from current bar
    // @TODO : it takes into account that the currently played music is not from that bar; fix that

    // rewind(bar.getStartTick());

    auto notesBeginIndex = notes.size();

    for (; bar.getEndTick() <= (melodyEnd-1)->tick; bar.next())
    {
        addKick(bar, beatToTick(0));
        if (dist(gen) < 0.25)
            addKick(bar, beatToTick(0.5));

        bool hasDoubleKick = false;
        addKick(bar, beatToTick(2));
        if (dist(gen) < 0.25)
        {
            addKick(bar, beatToTick(2.5));
            hasDoubleKick = true;
        }

        // if (isSparse(bar))
        // {
        //     addKick(bar, beatToTick(3));
        // }
        // else 
        // if (dist(gen) < 0.3)
        // {
        //     addKick(bar, beatToTick(3));
        // }

        addSnare(bar, beatToTick(1));

        addSnare(bar, beatToTick(3));
        if (!hasDoubleKick && dist(gen) < 0.2)
            addSnare(bar, beatToTick(3.125f));

        for (float beat = 0; beat < 4; beat += 0.5)
        {
            if (dist(gen) < 0.8)
            {
                addClosedHat(bar, beatToTick(beat));
            }
        }

        // if (melody.hasGap(bar, length=0.5))
        // {
        //     addGhostSnare(bar.randomSubbeat);
        // }
        // if (isPhraseEnd(bar))
        // {
        //     addFill(bar, beatToTick(bar.getEndBeat() - 0.5));
        // }
    }

    std::sort(notes.begin() + notesBeginIndex, notes.end(), [](const BeatNote& lhs, const BeatNote& rhs){return lhs.note.tick < rhs.note.tick;});
}


BeatGeneratorHandle createBeatGenerator()
{
    return new BeatGenerator();
}
void destroyBeatGenerator(BeatGeneratorHandle beatGenerator)
{
    delete beatGenerator;
}

void beatGenerator_rewind(BeatGeneratorHandle beatGenerator, int tick)
{
    beatGenerator->rewind(tick);
}

void beatGenerator_getNotes(BeatGeneratorHandle beatGenerator, const BeatNote** outNotes, int32_t* outLength)
{
    const std::vector<BeatNote>& beatNotes = beatGenerator->getNotes();
    *outNotes = beatNotes.data();
    *outLength = int32_t(beatNotes.size());
}

void beatGenerator_refreshFromGenHistory(BeatGeneratorHandle beatGenerator, const GenerationHistory* melodyGenHistory)
{
    const std::vector<Note>& notes = melodyGenHistory->getNotes();
    beatGenerator->refresh(notes.data(), notes.data() + notes.size());
}

void beatGenerator_refresh(BeatGeneratorHandle beatGenerator, const Note* melodyBegin, const Note* melodyEnd)
{
    beatGenerator->refresh(melodyBegin, melodyEnd);
}


