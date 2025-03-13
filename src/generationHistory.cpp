#include "generationHistory.hpp"
#include <cassert>
#include "midiTokenizer.hpp"

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

void GenerationHistory::addEncodedToken(int32_t newEncodedToken)
{
    static thread_local std::vector<int32_t> decodedTokens;
    tokenizer.decodeToken(newEncodedToken, decodedTokens);
    for (int32_t i = 0; i < decodedTokens.size(); i++)
    {
        decodedTokensHistory.addToken(decodedTokens[i]);
    }

    encodedTokensHistory.addToken(newEncodedToken);
}