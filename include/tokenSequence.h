#pragma once

#include <vector>
#include <string>

class TokSequence
{
public:
    class MidiTokenizer* tokenizer;

    std::vector<int32_t> ids; // potentially encoded tokens
    bool are_ids_encoded;

    std::vector<std::string> tokens; // decoded tokens

    std::string bytes;

    size_t length() const
    {
        // assert(!ids.empty()); // if 0, return size of bytes or tokens or events
        return ids.size();
    }

    std::vector<int32_t>& getDecodedTokens()
    {
        return ids;
    }

    std::vector<int32_t>& getEncodedTokens()
    {
        return ids;
    }
};

// class TokenSequence
// {
// public:
//     std::vector<int32_t> encodedTokens;
//     std::vector<int32_t> decodedTokens;
        // class MidiTokenizer* tokenizer;

// public:
//     // @TODO : disable for optimizations
//     std::vector<std::string> tokens; // decoded tokens

// public:
//     size_t length() const
//     {
//         // assert(!ids.empty()); // if 0, return size of bytes or tokens or events
//         return encodedTokens.size();
//     }
// };

using TokenSequence = TokSequence;