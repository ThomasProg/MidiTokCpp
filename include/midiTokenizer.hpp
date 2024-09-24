#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include "json.hpp"

#include "tokenizers_cpp.h"

using json = nlohmann::json;

using TokSequenceInt = std::vector<int64_t>;
using Score = std::vector<int64_t>;

// https://github.com/Natooz/MidiTok/blob/main/miditok/midi_tokenizer.py
class MidiTokenizer
{
private:
    std::map<int, std::string> __vocab_base_inv;

protected:    
    std::map<std::string, int> _vocab_base;
    std::map<char, std::string> _vocab_base_byte_to_token;

    std::map<char, std::string> _vocab_base_id_to_byte;
    std::unique_ptr<tokenizers::Tokenizer> _model;

private:
    void __create_vocab_learned_bytes_to_tokens()
    {
        // @TODO
    }

protected:

    void loadFromJson(const std::string& filename);

public:
    MidiTokenizer(const std::string& filename)
    {
        // try 
        // {
            loadFromJson(filename);

        // }
        // catch (const std::exception& e)
        // {
        //     std::cout << "EXCEPTION: " << e.what() << std::endl;
        // }
            
    }

    TokSequenceInt encode(const Score& score);
    Score decode(const TokSequenceInt& tokens);
};


