#include "midiConverter.hpp"
#include "midiTokenizer.hpp"
#include "redirector.hpp"

void MIDIConverter::reset()
{
    currentTick = 0;
    tickAtCurrentBar = 0;
    currentBar = -1;
}

void MIDIConverter::processToken(const int32_t* tokens, int32_t nbTokens, std::int32_t index, void* data)
{
    if (index >= nbTokens)
        return;

    MidiTokenizer& tokenizer = *tokenizerHandle;

    int32_t token = tokens[index];
    if (tokenizer.isBarNone(token))
    {
        currentBar += 1;
        if (currentBar > 0)
        {
            currentTick = tickAtCurrentBar + ticksPerBar;
        }
        tickAtCurrentBar = currentTick;
    }
    else if (tokenizer.isPosition(token))
    {
        if (currentBar == -1)
        {
            currentBar = 0;
        }
        currentTick = tickAtCurrentBar + tokenizer.getPositionValue(token) * ticksPerPos;
    }
    else if (tokenizer.isPitch(token)/* || tokenizer.isPitchDrum(token) || tokenizer.isPitchIntervalTime(token) || tokenizer.isPitchIntervalChord(token)*/)
    {
        if (tokenizer.isPitch(token))
        {
            std::int32_t pitch = tokenizer.getPitchValue(token);

            // bool useVelocities = true;
            // if (useVelocities)
            int32_t velocityTokenOffset = 1;
            int32_t durationTokenOffset = 2;
            if (index + durationTokenOffset < nbTokens && index + velocityTokenOffset < nbTokens)
            {
                int32_t velocityToken = tokens[index + velocityTokenOffset];
                int32_t durationToken = tokens[index + durationTokenOffset];

                if (tokenizer.isVelocity(velocityToken) && tokenizer.isDuration(durationToken))
                {
                    // dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                    Note newNote;
                    newNote.pitch = pitch;
                    newNote.tick = currentTick;
                    newNote.duration = tokenizer.getDurationValue(durationToken);
                    newNote.velocity = tokenizer.getVelocityValue(velocityToken);
                    onNote(data, newNote);
                }
            } 
        }
    }
}

void MIDIConverter::processToken(const std::vector<int32_t>& tokens, std::int32_t index, void* data)
{
    processToken(tokens.data(), std::int32_t(tokens.size()), index, data);
}

void tryPlay(const std::vector<int32_t>& tokens, std::int32_t& unplayedTokenIndex)
{
    std::int32_t i = unplayedTokenIndex;

    struct Args
    {
        std::int32_t& i;
        std::int32_t& unplayedTokenIndex;
    };

    Args args{i, unplayedTokenIndex};

    MIDIConverter converter;
    converter.onNote = [](void* data, const Note& newNote)
    {
        Args& args = *(Args*)(data);
        args.unplayedTokenIndex = args.i + 1;



    };


    while (i < tokens.size())
    {
        converter.processToken(tokens, i, &args);
        i++;
    }
}