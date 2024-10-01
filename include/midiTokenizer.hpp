#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <cassert>
#include "json.hpp"

#include "tokenizers_cpp.h"

using json = nlohmann::json;

using Score = std::vector<int32_t>;


class TokSequence
{
public:
    std::vector<int32_t> ids; // potentially encoded tokens
    bool are_ids_encoded;

    std::vector<std::string> tokens; // decoded tokens

    std::string bytes;

    size_t length() const
    {
        assert(!ids.empty()); // if 0, return size of bytes or tokens or events
        return ids.size();
    }
};




// https://github.com/Natooz/MidiTok/blob/main/miditok/midi_tokenizer.py
class MidiTokenizer
{
private:
    // the other way, to decode id (int) -> token (str)
    std::map<int32_t, std::string> __vocab_base_inv;

protected:    
    // vocab of prime tokens, can be viewed as unique char / bytes
    std::map<std::string, int32_t> _vocab_base;

    // byte (str) -> token (str), for basic tokens
    std::map<std::string, std::string> _vocab_base_byte_to_token;

    // id (int) -> byte (str), as this might not be chr(id) after tokenizer training
    std::map<int32_t, std::string> _vocab_base_id_to_byte;

    // Fast tokenizer model backed with 🤗tokenizers
    std::unique_ptr<tokenizers::Tokenizer> _model;


    // TODO / TO LOAD
    // byte(s) -> token(s), for faster BPE/Unigram/WordPiece decoding
    std::map<std::string, std::vector<std::string>> _vocab_learned_bytes_to_tokens;

private:
    void __create_vocab_learned_bytes_to_tokens();

protected:
    TokSequence _convert_sequence_to_tokseq(const std::vector<int32_t>& tokens);
    void loadFromJson(const std::string& filename);
    bool _are_ids_encoded(const std::vector<int32_t>& tokens) const;
    void _preprocess_tokseq_before_decoding(TokSequence& seq);

    // Convert a list of tokens (str) into their ids format (int).

    // :param tokens: list of tokens (str) to convert.
    // :return: list of corresponding ids (int).
    std::vector<int32_t> _tokens_to_ids(const std::vector<std::string>& tokens);



    // Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

    // This is an internal method called by ``self.decode``, intended to be
    // implemented by classes inheriting :class:`miditok.MusicTokenizer`.

    // :param tokens: tokens to convert. Can be either a list of
    //     :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
    // :param programs: programs of the tracks. If none is given, will default to
    //     piano, program 0. (default: ``None``)
    // :return: the ``symusic.Score`` object.
    Score _tokens_to_score(const TokSequence& seq);

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

    std::vector<int32_t> encode(const Score& score);
    Score decode(const std::vector<int32_t>& tokens);

    // custom
    void decodeIDs(const std::vector<int32_t>& tokens, std::vector<int32_t>& outTokens);

    // @TODO : optimize
    std::function<int(const std::string&)> vocab = [this](const std::string& v) -> int
    {
        return _vocab_base.at(v);
    };

    const std::map<std::string, int32_t>& GetVocabBase() const
    {
        return _vocab_base;
    }


    // Decode the ids of a :class:`miditok.TokSequence` with BPE, Unigram or WordPiece.

    // This method only modifies the ``.ids`` attribute of the input sequence(s)
    // and does not complete it. This method can be used recursively on lists of
    // :class:`miditok.TokSequence`.

    // :param seq: token sequence to decompose.
    void decode_token_ids(TokSequence& seq);



    // Complete (inplace) a :class:`miditok.TokSequence`.

    // The input sequence can have some of its attributes (``ids``, ``tokens``) not
    // initialized (i.e. ``None``). This method will initialize them from the present
    // ones. The ``events`` attribute will not be filled as it is only intended for
    // debug purpose. The ``bytes`` attribute will be created if ``complete_bytes`` is
    // provided as ``True`` and if the tokenizer has been trained.

    // :param seq: input :class:`miditok.TokSequence`, must have at least one
    //     attribute defined.
    // :param complete_bytes: will complete the bytes form of each token. This is only
    //     applicable if the tokenizer has been trained.
    void complete_sequence(TokSequence& seq, bool complete_bytes = false);

    bool is_trained() const;
};


