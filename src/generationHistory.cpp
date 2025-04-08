#include "generationHistory.h"
#include "generationHistory.hpp"
#include <cassert>
#include "midiTokenizer.hpp"
#include "midiConverter.hpp"
#include "midiConverter.h"

void TokenHistory::addToken(int32_t newDecodedToken)
{
    TokenData& data = tokensData[newDecodedToken];
    const int32_t tokenTurn = getCurrentTokenTurn();
    data.births.push_back(tokenTurn);
    tokens.push_back(newDecodedToken);
}

bool TokenHistory::findMostRecentAge(int32_t token, int32_t& outAge) const
{
    auto findIt = tokensData.find(token);
    if (findIt == tokensData.end())
    {
        return false;
    }

    assert(!findIt->second.births.empty());
    outAge = findMostRecentAge(findIt->second); 
    return true;
}

bool TokenHistory::hadTokenRecently(int32_t token, int32_t maxAge) const
{
    assert(maxAge > 0);
    int32_t age;
    if (findMostRecentAge(token, age))
    {
        return age <= maxAge;
    }
    return false;
}

void TokenHistory::removeAfterIndex(int32_t index)
{
    for (auto rit = tokens.rbegin(); rit != tokens.rend() - index; rit++)
    {
        int32_t token = *rit;
        auto tokenDataIt = tokensData.find(token);
        tokenDataIt->second.births.pop_back();
        if (tokenDataIt->second.births.empty())
        {
            tokensData.erase(tokenDataIt);
        }
    }

    tokens.erase(tokens.begin() + index, tokens.end());
}

const std::vector<Note>& GenerationHistory::getNotes() const
{
    return notes;
}

void GenerationHistory::addEncodedToken(int32_t newEncodedToken)
{
    // static thread_local std::vector<int32_t> decodedTokens;
    // tokenizer.decodeToken(newEncodedToken, decodedTokens);

    const int32_t* outDecodedTokensBegin;
    const int32_t* outDecodedTokensEnd;
    tokenizer.decodeTokenFast(newEncodedToken, outDecodedTokensBegin, outDecodedTokensEnd);

    const int32_t encodedTokenIndex = encodedTokensHistory.getCurrentTokenTurn();
    
    while (outDecodedTokensBegin != outDecodedTokensEnd)
    {
        decodedTokensHistory.addToken(*outDecodedTokensBegin);
        decodedTokenIndexToEncodedTokenIndex.push_back(encodedTokenIndex);

        ++outDecodedTokensBegin;
    }

    encodedTokensHistory.addToken(newEncodedToken);
}

void GenerationHistory::convert()
{
    // assert(converter != nullptr);
    if (converter == nullptr)
    {
        converter = createConverterFromTokenizer(&tokenizer);
    }

    struct ConvertStruct
    {
        GenerationHistory* history;
        bool isNote;
    };

    converter->onNote = [](void* data, const Note& note)
    {
        ConvertStruct* conv = (ConvertStruct*) data;
        conv->history->notes.push_back(note);
        conv->isNote = true;
    };

    std::int32_t i = nextTokenToProcess;

    const int32_t* decodedTokens = decodedTokensHistory.getTokensData();
    size_t nbTokens = decodedTokensHistory.getTokensSize();
	while (i < nbTokens)
	{
        int32_t start = i;
	
        ConvertStruct conv{this, false};

        bool isSuccess = converter->processToken(decodedTokens, int32_t(nbTokens), i, &conv);
		if (isSuccess)
		{
            if (conv.isNote)
            {
                int32_t end = i;
                noteIndexToDecodedTokenIndex.emplace_back(start, end);
            }

			nextTokenToProcess = i;
		}
		else
		{
			i++; // ignore current token and continue
			if (i - nextTokenToProcess > 20)
			{
				i += 10; // in case there are too many errors, ignore the 10 next tokens
				nbSkips++;
			}
		}
	}
}

void GenerationHistory::removeAfterTick(int32_t tick)
{
    convert();

    if (notes.empty())
    {
        return;
    }

    // consider we want to remove an element towards the end;
    // dichotomy might be faster otherwise
    // but we consider the generation stops if there are enough tokens generated already
    auto rit = std::find_if(notes.rbegin(), notes.rend(), [tick](const Note& elem)
    {
        return elem.tick < tick;
    });

    // index of the next element, excluding the one for which the predicate is true
    size_t index = notes.rend() - rit;
    if (index >= notes.size())
    {
        return;
    }
    notes.erase(notes.begin() + index, notes.end());
    auto [decodedTokenIndexStart, decodedTokenIndexEnd] = noteIndexToDecodedTokenIndex[index];

    noteIndexToDecodedTokenIndex.erase(noteIndexToDecodedTokenIndex.begin() + index, noteIndexToDecodedTokenIndex.end());
    
    int32_t encodedTokenIndex = decodedTokenIndexToEncodedTokenIndex[decodedTokenIndexStart];
    decodedTokenIndexToEncodedTokenIndex.erase(decodedTokenIndexToEncodedTokenIndex.begin() + decodedTokenIndexStart, decodedTokenIndexToEncodedTokenIndex.end());

    encodedTokensHistory.removeAfterIndex(encodedTokenIndex);
    decodedTokensHistory.removeAfterIndex(decodedTokenIndexStart);

    nextTokenToProcess = std::min(nextTokenToProcess, int32_t(decodedTokensHistory.getTokensSize()));

    if (converter != nullptr)
    {
        converter->rewind(tick);
    }
}




void addEncodedToken(const GenerationHistoryHandle genHistory, int32_t newEncodedToken)
{
    return genHistory->addEncodedToken(newEncodedToken);
}

bool hadEncodedTokenRecently(const GenerationHistoryHandle genHistory, int32_t token, int32_t maxAge)
{
    return genHistory->hadEncodedTokenRecently(token, maxAge);
}
bool hadDecodedTokenRecently(const GenerationHistoryHandle genHistory, int32_t token, int32_t maxAge)
{
    return genHistory->hadDecodedTokenRecently(token, maxAge);
}

TokenHistoryHandle getEncodedTokensHistory(const GenerationHistoryHandle genHistory)
{
    return &genHistory->getEncodedTokensHistory();
}
TokenHistoryHandle getDecodedTokensHistory(const GenerationHistoryHandle genHistory)
{
    return &genHistory->getDecodedTokensHistory();
}

void generationHistory_removeAfterTick(const GenerationHistoryHandle genHistory, int32_t tick)
{
    genHistory->removeAfterTick(tick);
}

void generationHistory_convert(const GenerationHistoryHandle genHistory)
{
    genHistory->convert();
}

void generationHistory_getNotes(const GenerationHistoryHandle genHistory, const struct Note** outNotes, size_t* outLength)
{
    const std::vector<Note>& notes = genHistory->getNotes();
    *outNotes = notes.data();
    *outLength = notes.size();
}
 

void addToken(TokenHistoryHandle tokenHistory, int32_t newToken)
{
    return tokenHistory->addToken(newToken);
}
bool findMostRecentAge(TokenHistoryHandle tokenHistory, int32_t token, int32_t* outAge)
{
    assert(tokenHistory != nullptr && outAge != nullptr);
    return tokenHistory->findMostRecentAge(token, *outAge);
}
bool hadTokenRecently(TokenHistoryHandle tokenHistory, int32_t token, int32_t maxAge)
{
    assert(tokenHistory != nullptr);
    return tokenHistory->hadTokenRecently(token, maxAge);
}

void tokenHistory_getTokens(TokenHistoryHandle tokenHistory, const int32_t** outTokens, int32_t* outSize)
{
    assert(tokenHistory != nullptr);
    const std::vector<int32_t>& tokens = tokenHistory->getTokens();
    *outTokens = tokens.data();
    *outSize = int32_t(tokens.size());
}