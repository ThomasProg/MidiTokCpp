#pragma once

#include "fwd.h"
#include <unordered_map>
#include <vector>

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
};

class GenerationHistory
{
private:
    TokenHistory encodedTokensHistory;
    TokenHistory decodedTokensHistory;

    const MidiTokenizer& tokenizer;

public:
    GenerationHistory(const MidiTokenizer& inTokenizer) : tokenizer(inTokenizer) {}

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
    const TokenHistory& getEncodedTokensHistory() const
    {
        return encodedTokensHistory;
    }
    const TokenHistory& getDecodedTokensHistory() const
    {
        return decodedTokensHistory;
    }
};