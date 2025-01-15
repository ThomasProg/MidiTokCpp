#include "midiConverter.hpp"
#include "midiTokenizer.hpp"
#include "redirector.hpp"

bool MIDIConverter::processToken(const std::vector<int32_t>& tokens, std::int32_t& index, void* data)
{
    return processToken(tokens.data(), std::int32_t(tokens.size()), index, data);
}

void REMIConverter::reset()
{
    currentTick = 0;
    tickAtCurrentBar = 0;
    currentBar = -1;
}

bool REMIConverter::processToken(const int32_t* tokens, int32_t nbTokens, std::int32_t& index, void* data)
{
    if (index >= nbTokens)
        return false;

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
        index++;
        return true;
    }
    else if (tokenizer.isPosition(token))
    {
        if (currentBar == -1)
        {
            currentBar = 0;
        }
        currentTick = tickAtCurrentBar + tokenizer.getPositionValue(token) * ticksPerPos;
        index++;
        return true;
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
                    index += 3;
                    return true;
                }
            } 
        }
    }

    return false;
}




void TSDConverter::reset()
{
    currentTick = 0;
    previousNoteEnd = 0;
    ticks_per_beat = 0;
}
bool TSDConverter::processToken(const int32_t* tokens, int32_t nbTokens, std::int32_t& index, void* data)
{
    if (index >= nbTokens)
        return false;

    MidiTokenizer& tok = *tokenizerHandle;
    int32_t token = tokens[index];

    // @TODO : do at set only (so if timesig event, doesn't reset)
    ticks_per_beat = tok.time_division;


    if (tok.isTimeShift(token))
    {
        currentTick += tok._tpb_tokens_to_ticks[ticks_per_beat][tok.getTimeShiftValue(token)];
        index++;
        return true;
    }
    else if (tok.isRest(token))
    {
        currentTick = std::max(previousNoteEnd, currentTick);
        currentTick += tok._tpb_rests_to_ticks[ticks_per_beat][tok.getRestValue(token)];
        index++;
        return true;
    }
    else if (tok.isPitch(token)) // @TODO : PitchDrum, PitchIntervalTime, PitchIntervalChord
    {
        std::int32_t pitch = tok.getPitchValue(token);

        std::int32_t velocity = defaultVelocity;
        std::int32_t duration = defaultDuration;
        std::int32_t indexIncr = 1;

        if (tok.useVelocities())
        {
            std::int32_t velocityToken = tokens[index+velocityOffset];
            if (tok.isVelocity(velocityToken))
            {
                velocity = tok.getVelocityValue(velocityToken);
                indexIncr += 1;
            }
            else
            {
                return false;
            }
        }

        if (tok.useDuration())
        {
            std::int32_t durationToken = tokens[index+durationOffset];
            if (tok.isDuration(durationToken))
            {
                // if isinstance(dur, str):
                //     dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                velocity = tok.getDurationValue(durationToken);
                indexIncr += 1;
            }
            else
            {
                return false;
            }
        }

        Note newNote;
        newNote.pitch = pitch;
        newNote.tick = currentTick;
        newNote.duration = duration;
        newNote.velocity = velocity;
        onNote(data, newNote);
        index += indexIncr;
        return true;
    }
    else
    {
        if (tok.isProgram(token))
        {
            throw std::logic_error("Program not supported yet");
        }
        else if (tok.isTempo(token))
        {
            throw std::logic_error("Tempo not supported yet");
        }
        else if (tok.isTimeSig(token))
        {
            // ticks_per_beat = tok.compute_ticks_per_beat();
            throw std::logic_error("TimeSig not supported yet");
        }
        else if (tok.isPedal(token))
        {
            throw std::logic_error("Pedal not supported yet");
        }
        else if (tok.isPedalOff(token))
        {
            throw std::logic_error("PedalOff not supported yet");
        }
        else if (tok.isPitchBend(token))
        {
            throw std::logic_error("PitchBend not supported yet");
        }

        previousNoteEnd = std::max(previousNoteEnd, currentTick);
    }

    return false;
}