#include "midiTokenizer.hpp"

#include <memory>
#include <string>
#include <codecvt>

// Function to get the number of bytes in a UTF-8 character based on the first byte
size_t utf8_char_length(unsigned char c) {
    if (c == 0xC4)
    {
        return 2;
    }
    else 
    {
        return 1;
    }
}

#include <sstream>
std::string escape_invalid_characters(const std::string& str) {
    std::ostringstream oss;

    for (unsigned char c : str) {
        // Check if the character is valid (ASCII range)
        if (c < 32 || c > 126) {
            // Escape the character as hexadecimal
            oss << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
        } else {
            // Add valid characters as-is
            oss << c;
        }
    }

    return oss.str();
}

void printAsAscii(const std::string& str) {
    // Set the locale to the user's locale
    std::locale::global(std::locale("en_US.UTF-8"));


    // Convert key and value to wstring (wide string)
    std::string s = escape_invalid_characters(str);
    
    std::cout << s;
}

void MidiTokenizer::__create_vocab_learned_bytes_to_tokens()
{
    std::map<std::string, int32_t> vocab = _model->GetVocab(); 


    // for (auto& [key, value] : _vocab_learned_bytes_to_tokens)
    // {
    //     std::cout << key << ": \n";
    //     for (auto& v : value)
    //     {
    //         std::cout << " - " << v << '\n';
    //     }
    //     std::cout << '\n';
    // }

    for (auto& [key, value] : _vocab_base_byte_to_token)
    {
        printAsAscii(key);
        std::cout << ": ";
        printAsAscii(value);
        std::cout << '\n';
    }

    for (auto& [k, value] : vocab)
    {
        // @TODO : continuing_subword_prefix / end_of_word_suffix


        std::string key_ = k;

        std::string replacement = "_";
        if (key_ != replacement && key_.compare(0, replacement.length(), replacement) == 0) 
        {
            key_.erase(0, replacement.length());
        }



        std::vector<std::string> token_vector;
        token_vector.reserve(k.size());

        for (size_t i = 0; i < key_.size(); ) {
            size_t char_len = utf8_char_length(static_cast<unsigned char>(key_[i]));
            std::string utf8_char = key_.substr(i, char_len);  // Extract character

            // std::cout << "UTF-8 character: " << utf8_char << std::endl;
            token_vector.push_back(_vocab_base_byte_to_token.at(utf8_char));


            i += char_len;  // Move to the next character
        }


        // for (wchar_t byte : key_)
        // {
        //     token_vector.push_back(_vocab_base_byte_to_token.at(std::string(&byte, 1)));
        // }

        assert(!token_vector.empty());
        _vocab_learned_bytes_to_tokens[k] = std::move(token_vector);
    }


    // // Get the vocabulary size
    // size_t vocab_size = _model->GetVocabSize();

    // // Iterate over vocabulary IDs to simulate Python's get_vocab()
    // for (int32_t id = 0; id < vocab_size; ++id) {
    //     // Get the token corresponding to the vocab ID
    //     std::string k = _model->IdToToken(id);
    //     std::string key_ = k;

    //     char replacement = '_';
    //     if (key_.size() > 1 && key_[0] == replacement)
    //     {
    //         key_.erase(0);
    //         // key_.erase(0, replacement.length());
    //     }

    //     assert(!key_.empty());

    //     // For each token, fill the learned_bytes_to_tokens map
    //     std::vector<std::string> token_vector;
    //     for (char byte : key_) {
    //         // Convert each byte in the token to its corresponding learned token
    //         token_vector.push_back(_vocab_base_byte_to_token.at(std::string(&byte, 1)));
    //     }

    //     // Insert the token and its corresponding learned tokens into the map
    //     assert(!token_vector.empty());
    //     _vocab_learned_bytes_to_tokens[k] = token_vector;
    // }
}

std::string decode_utf8_to_binary(const std::string& utf8_string) {

    std::uint8_t firstByte = static_cast<std::uint8_t>(utf8_string[0]); 
    if (firstByte < 0x80)
    {
        return utf8_string;
    }

    else if (firstByte == 0xC2)
    {
        return std::string(&utf8_string[1], 1);
    }

    else if (firstByte == 0xC3)
    {
        char newChar = utf8_string[1] + (0xc0 - 0x80);
        return std::string(&newChar, 1);
    }

    return utf8_string;
}





void MidiTokenizer::loadFromJson(const std::string& filename)
{
    std::ifstream file(filename);

    if(!file)
    {
        std::cout << "Couldn't load" << std::endl;
        return;
    } 


    json data = json::parse(file);

    // Overwrite config attributes
    _vocab_base.clear();
    __vocab_base_inv.clear();
    for (auto& [key, value] : data.items())
    {
        if (key == "tokenization" || key == "miditok_version")
            continue;

        if (key == "_vocab_base")
        {
            for (auto& [vocaBaseKey, vocaBaseValue] : value.items())
            {
                _vocab_base[vocaBaseKey] = vocaBaseValue;
            }
            

            for (auto& [vocaBaseKey, vocaBaseValue] : value.items())
            {
                __vocab_base_inv[vocaBaseValue] = vocaBaseKey;
            }
            continue;
        }

        if (key == "_model")
        {
            std::string v = value.template get<std::string>();
            _model = tokenizers::Tokenizer::FromBlobJSON(v);
            continue;
        }

        if (key == "_vocab_base_byte_to_token")
        {
            for (auto& [k, v] : value.items())
            {
                assert(k.size() <= 2);
                std::string k2 = decode_utf8_to_binary(k);
                std::string vStr = v.template get<std::string>();
                // if (k2.size() == 1)
                // {
                //     std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(k2[0]); // Display each character as hex
                //     std::cout << ": " << vStr << std::endl;
                // }
                // else 
                // {
                //     std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(k2[0]); // Display each character as hex
                //     std::cout << " ";
                //     std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(k2[1]); // Display each character as hex
                //     std::cout << ": " << vStr << std::endl;
                // }


                _vocab_base_byte_to_token[k2] = std::move(vStr);
            }

            std::map<std::string, std::string> tokenToByte;
            for (auto& [k, v] : value.items())
            {
                tokenToByte[v] = k;
            }

            for (auto& [tok, i] : _vocab_base)
            {
                _vocab_base_id_to_byte[i] = tokenToByte[tok];
            }

            __create_vocab_learned_bytes_to_tokens();
        }

        if (key == "config")
        {
            if (value.contains("chord_maps"))
            {
                // @TODO
            }
            const char* beat_res_keys[] = {"beat_res", "beat_res_rest"};
            for (const char* beat_res_key : beat_res_keys)
            {
                // @TODO
            }

            // @TODO
        }

        if (key == "has_bpe")
        {
            // For config files < v3.0.3 before the attribute becomes a property
            continue;
        }


        // @TODO : cant add fields to self
        // setattr(self, key, value)
        // std::cout << key << std::endl;
    }


}

TokSequenceInt MidiTokenizer::encode(const Score& score)
{
    return TokSequenceInt();
}

TokSequence MidiTokenizer::_convert_sequence_to_tokseq(const std::vector<int32_t>& tokens)
{
    // std::vector<> seq;

    // for (const int32_t& obj : tokens)
    // {

    // }
    TokSequence tokSequence;
    tokSequence.ids = tokens;
    tokSequence.are_ids_encoded = _are_ids_encoded(tokens);
    return tokSequence;
}

bool MidiTokenizer::_are_ids_encoded(const std::vector<int32_t>& tokens) const
{
    return true;
}

void MidiTokenizer::_preprocess_tokseq_before_decoding(TokSequence& seq)
{
    if (seq.tokens.size() == 0)
    {
        if (seq.are_ids_encoded)
        {
            decode_token_ids(seq);
        }
        complete_sequence(seq);
    }
}

void MidiTokenizer::complete_sequence(TokSequence& seq, bool complete_bytes)
{
    if (seq.tokens.empty())
    {
        throw std::logic_error("Unimplemented");

        // if len(seq.events) > 0:
        //     seq.tokens = self._events_to_tokens(seq.events)
        // elif len(seq.ids) > 0:
        //     seq.tokens = self._ids_to_tokens(seq.ids)
        // elif len(seq.bytes) > 0:
        //     seq.tokens = self._bytes_to_tokens(seq.bytes)
    }

    if (seq.ids.empty())
    {
        seq.ids = _tokens_to_ids(seq.tokens);
    }

    if (complete_bytes && is_trained() && seq.bytes.empty())
    {
        throw std::logic_error("Unimplemented");
        // seq.bytes = _ids_to_bytes(seq.ids);
    }
}

std::vector<int32_t> MidiTokenizer::_tokens_to_ids(const std::vector<std::string>& tokens)
{
    if (tokens.empty())
        return std::vector<int32_t>();

    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const std::string& token : tokens)
    {
        ids.push_back(vocab(token));
    }
    return ids; 
}

void MidiTokenizer::decode_token_ids(TokSequence& seq)
{
    if (seq.are_ids_encoded)
    {
        std::vector<std::string> encoded_bytes(seq.ids.size()); 
        for (size_t i = 0; i < seq.ids.size(); i++)
        {
            auto& id_ = seq.ids[i];
            encoded_bytes[i] = _model->IdToToken(id_);
            assert(!encoded_bytes[i].empty()); // if empty, it means id hasn't been found
        } 

        std::vector<std::string> decoded_tokens;
        decoded_tokens.reserve(encoded_bytes.size()); 
        for (std::string& byte_ : encoded_bytes)
        {
            const std::vector<std::string>& decoded = _vocab_learned_bytes_to_tokens.at(byte_); 
            assert(!decoded.empty());
            decoded_tokens.insert(decoded_tokens.end(), decoded.begin(), decoded.end());
        }  

        seq.tokens = decoded_tokens;

        // @TODO : Necessary???
        seq.ids = _tokens_to_ids(decoded_tokens);
        seq.are_ids_encoded = false;
    }
}

Score MidiTokenizer::_tokens_to_score(const TokSequence& seq)
{
    // # Unsqueeze tokens in case of one_token_stream
    // if self.config.one_token_stream_for_programs:  # ie single token seq
    //     tokens = [tokens]

    // std::vector<std::string> tokens;
    // tokens.push_back(seq.tokens);

    Score score = Score();
    // score.time_division = time_division;

    // for (size_t i = 0; i < seq.length(); i++)
    // {
    //     tokens.push_back(seq[i].tokens);
    // }

    return score;
}

Score MidiTokenizer::decode(const TokSequenceInt& tokens)
{
    TokSequence tokSequence = _convert_sequence_to_tokseq(tokens);

    _preprocess_tokseq_before_decoding(tokSequence);

    Score score = _tokens_to_score(tokSequence);


    return score;
}

bool MidiTokenizer::is_trained() const
{
    return _model.get() != nullptr;
}

    // def decode(
    //     self,
    //     tokens: TokSequence | list[TokSequence] | list[int | list[int]] | np.ndarray,
    //     programs: list[tuple[int, bool]] | None = None,
    //     output_path: str | Path | None = None,
    // ) -> Score:
    //     r"""
    //     Detokenize one or several sequences of tokens into a ``symusic.Score``.

    //     You can give the tokens sequences either as :class:`miditok.TokSequence`
    //     objects, lists of integers, numpy arrays or PyTorch/Jax/Tensorflow tensors.
    //     The Score's time division will be the same as the tokenizer's:
    //     ``tokenizer.time_division``.

    //     :param tokens: tokens to convert. Can be either a list of
    //         :class:`miditok.TokSequence`, a Tensor (PyTorch and Tensorflow are
    //         supported), a numpy array or a Python list of ints. The first dimension
    //         represents tracks, unless the tokenizer handle tracks altogether as a
    //         single token sequence (``tokenizer.one_token_stream == True``).
    //     :param programs: programs of the tracks. If none is given, will default to
    //         piano, program 0. (default: ``None``)
    //     :param output_path: path to save the file. (default: ``None``)
    //     :return: the ``symusic.Score`` object.
    //     """
    //     if not isinstance(tokens, (TokSequence, list)) or (
    //         isinstance(tokens, list)
    //         and any(not isinstance(seq, TokSequence) for seq in tokens)
    //     ):
    //         tokens = self._convert_sequence_to_tokseq(tokens)

    //     # Preprocess TokSequence(s)
    //     if isinstance(tokens, TokSequence):
    //         self._preprocess_tokseq_before_decoding(tokens)
    //     else:  # list[TokSequence]
    //         for seq in tokens:
    //             self._preprocess_tokseq_before_decoding(seq)

    //     score = self._tokens_to_score(tokens, programs)

    //     # Create controls for pedals
    //     # This is required so that they are saved when the Score is dumped, as symusic
    //     # will only write the control messages.
    //     if self.config.use_sustain_pedals:
    //         for track in score.tracks:
    //             for pedal in track.pedals:
    //                 track.controls.append(ControlChange(pedal.time, 64, 127))
    //                 track.controls.append(ControlChange(pedal.end, 64, 0))
    //             if len(track.pedals) > 0:
    //                 track.controls.sort()

    //     # Set default tempo and time signatures at tick 0 if not present
    //     if len(score.tempos) == 0 or score.tempos[0].time != 0:
    //         score.tempos.insert(0, Tempo(0, self.default_tempo))
    //     if len(score.time_signatures) == 0 or score.time_signatures[0].time != 0:
    //         score.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))

    //     # Write file
    //     if output_path:
    //         output_path = Path(output_path)
    //         output_path.mkdir(parents=True, exist_ok=True)
    //         if output_path.suffix in ABC_FILES_EXTENSIONS:
    //             score.dump_abc(output_path)
    //         else:
    //             score.dump_midi(output_path)
    //     return score