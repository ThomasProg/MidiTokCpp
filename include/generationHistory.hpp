#pragma once

#include "fwd.h"
#include <unordered_map>
#include <vector>
#include "note.h"
#include "musicAdaptSequencer.hpp"

class TokenHistory
{
private:
    struct TokenData
    {
        // from oldest age to youngest
        std::vector<int32_t> births;
    };
    std::unordered_map<int32_t, TokenData> tokensData;
    std::vector<int32_t> tokens;

public:
    const std::vector<int32_t>& getTokens() const
    {
        return tokens;
    }

    const int32_t* getTokensData() const
    {
        return tokens.data();
    } 
    size_t getTokensSize() const
    {
        return tokens.size();
    } 

    int32_t getCurrentTokenTurn() const
    {
        return int32_t(tokens.size());
    }

    int32_t findMostRecentAge(const TokenData& tokenData) const
    {
        return (getCurrentTokenTurn() - tokenData.births.back());
    }

public:
    void addToken(int32_t newToken);
    bool findMostRecentAge(int32_t token, int32_t& outAge) const;
    bool hadTokenRecently(int32_t token, int32_t maxAge) const;
    void removeAfterIndex(int32_t index);

    friend class TokenHistoryTest;
};

class GenerationHistory
{
private:
    // History
    TokenHistory encodedTokensHistory;
    TokenHistory decodedTokensHistory;

    // Conversion Data
    std::vector<int32_t> decodedTokenIndexToEncodedTokenIndex;

    std::vector<Note> notes;
    std::vector<std::pair<int32_t, int32_t>> noteIndexToDecodedTokenIndex; // (begin;end) ; end excluded

    // Conversion Data: Decoded Token -> Note
    int32_t nextTokenToProcess = 0;
    int32_t nbSkips = 0;

    // Converters
    const MidiTokenizer& tokenizer;
    MIDIConverter* converter = nullptr;

    Sequencer musicAdaptSequencer;

public:
    GenerationHistory() = delete;
    GenerationHistory(const MidiTokenizer& inTokenizer) : tokenizer(inTokenizer) {}

    const std::vector<Note>& getNotes() const;
    std::vector<Note>& getNotes();

    const MidiTokenizer& getTokenizer() const
    {
        return tokenizer;
    }

    void addEncodedToken(int32_t newEncodedToken);

    bool hadEncodedTokenRecently(int32_t token, int32_t maxAge) const
    {
        return encodedTokensHistory.hadTokenRecently(token, maxAge);
    }
    bool hadDecodedTokenRecently(int32_t token, int32_t maxAge) const
    {
        return decodedTokensHistory.hadTokenRecently(token, maxAge);
    }

    // const, must not be modified directly, must go through this class to be modified
    TokenHistory& getEncodedTokensHistory()
    {
        return encodedTokensHistory;
    }
    TokenHistory& getDecodedTokensHistory()
    {
        return decodedTokensHistory;
    }

    Sequencer& getSequencer()
    {
        return musicAdaptSequencer;
    }

    void addStandaloneNote(const Note& note);
    void convert();
    size_t tickToNoteIndex(int32_t tick) const;
    size_t tickToDecodedTokenIndex(int32_t tick) const;
    size_t tickToEncodedTokenIndex(int32_t tick) const;
    void removeAfterTick(int32_t tick);
    void removeLastTimeshift();

    friend class GenerationHistoryTest;
    friend class OnAddTokensArgs;

public:
    using TOnEncodedTokenAdded = void (*)(OnAddTokensArgs* args);
    static TOnEncodedTokenAdded getDefaultOnEncodedTokenAdded();
    TOnEncodedTokenAdded onEncodedTokenAdded = getDefaultOnEncodedTokenAdded();
    void* onEncodedTokenAddedData = nullptr;

    using TOnNoteAdded = void (*)(void* userData);
    TOnNoteAdded onNoteAdded = nullptr;
    void* onNoteAddedData = nullptr;
};