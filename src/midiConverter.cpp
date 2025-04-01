#include "midiConverter.hpp"
#include "midiConverter.h"
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

    const MidiTokenizer& tokenizer = *tokenizerHandle;

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
    else if (tokenizer.isPitchFast(token)/* || tokenizer.isPitchDrum(token) || tokenizer.isPitchIntervalTime(token) || tokenizer.isPitchIntervalChord(token)*/)
    {
        if (tokenizer.isPitchFast(token))
        {
            std::int32_t pitch = tokenizer.getPitchValueFast(token);

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


TSDConverter::TSDConverter()
{
    dynamicData.emplace_back();
}

void TSDConverter::reset()
{
    dynamicData.clear();
    dynamicData.emplace_back();
    ticks_per_beat = 0;
}
bool TSDConverter::processToken(const int32_t* tokens, int32_t nbTokens, std::int32_t& index, void* data)
{
    if (index >= nbTokens)
        return false;

    const MidiTokenizer& tok = *tokenizerHandle;
    int32_t token = tokens[index];

    // @TODO : do at set only (so if timesig event, doesn't reset)
    ticks_per_beat = tok.time_division;

    DynamicData current = dynamicData.back();

    if (tok.isTimeShift(token))
    {
        current.currentTick += tok._tpb_tokens_to_ticks.at(ticks_per_beat).at(tok.getTimeShiftValue(token));
        index++;
        dynamicData.push_back(current);
        return true;
    }
    else if (tok.isRest(token))
    {
        current.currentTick = std::max(current.previousNoteEnd, current.currentTick);
        current.currentTick += tok._tpb_rests_to_ticks.at(ticks_per_beat).at(tok.getRestValue(token));
        index++;
        dynamicData.push_back(current);
        return true;
    }
    else if (tok.isPitchFast(token)) // @TODO : PitchDrum, PitchIntervalTime, PitchIntervalChord
    {
        std::int32_t pitch = tok.getPitchValueFast(token);

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
        newNote.tick = current.currentTick;
        newNote.duration = duration;
        newNote.velocity = velocity;
        onNote(data, newNote);
        index += indexIncr;
        dynamicData.push_back(current);
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

        current.previousNoteEnd = std::max(current.previousNoteEnd, current.currentTick);
        dynamicData.push_back(current);
    }

    return false;
}

void TSDConverter::unwind(int32_t tick)
{
    while (dynamicData.size() > 1 && dynamicData.back().currentTick >= tick)
    {
        dynamicData.pop_back();
    }
}



MidiConverterHandle createConverterFromTokenizer(const MidiTokenizer* tokenizer)
{
    MidiConverterHandle converter = nullptr;
    std::string type = tokenizer->getTokenizationType();
    if (type == "REMI")
    {
        converter = createREMIConverter();
    }
    else if (type == "TSD")
    {
        converter = createTSDConverter();
    }

    if (converter != nullptr)
    {
        converterSetTokenizer(converter, tokenizer);
    }

    return converter;
}

MidiConverterHandle createREMIConverter()
{
    return new REMIConverter();
}
MidiConverterHandle createTSDConverter()
{
    return new TSDConverter();
}

void destroyMidiConverter(MidiConverterHandle converter)
{
    delete converter;
}

void converterSetOnNote(MidiConverterHandle converter, void (*onNote)(void* data, const Note&))
{
    converter->onNote = onNote;
}
bool converterProcessToken(MidiConverterHandle converter, const int32_t* tokens, int32_t nbTokens, std::int32_t* index, void* data)
{
    assert(index != nullptr);
    return converter->processToken(tokens, nbTokens, *index, data);
}

void converterSetTokenizer(MidiConverterHandle converter, const MidiTokenizer* tokenizer)
{
    converter->tokenizerHandle = tokenizer;
}
