#include "midiTokenizer.hpp"

#include <memory>
#include <string>
#include <codecvt>
// #include "utf8.h"
#include "range.hpp"

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

// std::vector<std::string> split_utf8_string(const std::string& input) {
//     std::vector<std::string> characters; // To hold resulting characters
//     size_t i = 0;

//     while (i < input.size()) {
//         unsigned char byte = static_cast<unsigned char>(input[i]); // Treat as unsigned

//         // Determine number of bytes in this UTF-8 character
//         size_t num_bytes = 0;
//         if ((byte & 0x80) == 0) {
//             // 1-byte character (0xxxxxxx)
//             num_bytes = 1;
//         } else if ((byte & 0xE0) == 0xC0) {
//             // 2-byte character (110xxxxx)
//             num_bytes = 2;
//         } else if ((byte & 0xF0) == 0xE0) {
//             // 3-byte character (1110xxxx)
//             num_bytes = 3;
//         } else if ((byte & 0xF8) == 0xF0) {
//             // 4-byte character (11110xxx)
//             num_bytes = 4;
//         } else {
//             // Invalid byte; skip it or handle error
//             i++;
//             continue;
//         }

//         // Collect the bytes for the character
//         if (i + num_bytes <= input.size()) {
//             std::string character = input.substr(i, num_bytes); // Get substring for the character
//             characters.push_back(character); // Store the character
//         }

//         i += num_bytes; // Move index forward by number of bytes
//     }

//     return characters;
// }

// Function to split a UTF-8 string into its individual characters
std::vector<std::string> split_utf8(const std::string& str) {
    std::vector<std::string> result;
    size_t i = 0;

    while (i < str.size()) {
        unsigned char c = str[i];
        size_t char_len = 1;

        // Determine the length of the current UTF-8 character
        if (c >= 0xF0) {        // 4-byte character
            char_len = 4;
        } else if (c >= 0xE0) { // 3-byte character
            char_len = 3;
        } else if (c >= 0xC0) { // 2-byte character
            char_len = 2;
        }

        // Extract the UTF-8 character
        result.push_back(str.substr(i, char_len));
        i += char_len;
    }

    return result;
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

    // for (auto& [key, value] : _vocab_base_byte_to_token)
    // {
    //     printAsAscii(key);
    //     std::cout << ": ";
    //     printAsAscii(value);
    //     std::cout << '\n';
    // }

    for (auto& [k, value] : vocab)
    {
        // @TODO : continuing_subword_prefix / end_of_word_suffix


        std::string key_ = k;

        std::string replacement = "\xe2\x96\x81";
        if (key_.compare(0, replacement.length(), replacement) == 0) 
        {
            key_.erase(0, replacement.length());
        }



        std::vector<std::string> token_vector;
        token_vector.reserve(k.size());

        // Split the string
        std::vector<std::string> split_str = split_utf8(key_);

        // Output the split parts
        for (const std::string& part : split_str) {
            token_vector.push_back(_vocab_base_byte_to_token.at(part));
        }

        // for (size_t i = 0; i < key_.size(); ) {
        //     size_t char_len = utf8_char_length(static_cast<unsigned char>(key_[i]));
        //     std::string utf8_char = key_.substr(i, char_len);  // Extract character

        //     // std::cout << "UTF-8 character: " << utf8_char << std::endl;
        //     token_vector.push_back(_vocab_base_byte_to_token.at(utf8_char));

        //     if (token_vector.back() == "Bar_None")
        //     {
        //         std::cout << k << std::endl;
        //     }


        //     i += char_len;  // Move to the next character
        // }


        // for (char byte : key_)
        // {
        //     token_vector.push_back(_vocab_base_byte_to_token.at(std::string(&byte, 1)));
        // }


        // // Create a UTF-8 iterator
        // utf8::iterator<std::string::iterator> it(key_.begin(), key_.begin(), key_.end());
        // utf8::iterator<std::string::iterator> end(key_.end(), key_.begin(), key_.end());

        // // Loop through the UTF-8 string
        // while (it != end) {
        //     // uint32_t codepoint = *it; // Get the current UTF-32 code point

        //     it->

        //     token_vector.push_back(_vocab_base_byte_to_token.at(std::string(&byte, 1)));

        //     // // Print the code point in hexadecimal format
        //     // std::cout << "Codepoint: " << std::hex << codepoint << std::endl;

        //     // // Print the UTF-8 character
        //     // std::cout << "Character: " << utf8::utf32to8(codepoint) << std::endl;

        //     // Advance the iterator
        //     ++it;
        // }


        // std::vector<std::string> result = split_utf8_string(key_);

        // // Output each character
        // for (const auto& character : result) {
        //     // std::cout << "Character: " << character << std::endl;
        //     token_vector.push_back(_vocab_base_byte_to_token.at(character));
        // }

        // for (size_t i = 0; i < key_.size(); ) {
        //     int len = 0;
        //     const char* s = key_.data();
        //     while (*s) len += (*s++ & 0xc0) != 0x80;

        //     // size_t char_len = utf8_char_length(static_cast<unsigned char>(key_[i]));
        //     std::string utf8_char = key_.substr(i, len);  // Extract character

        //     // std::cout << "UTF-8 character: " << utf8_char << std::endl;
        //     token_vector.push_back(_vocab_base_byte_to_token.at(utf8_char));

        //     if (token_vector.back() == "Bar_None")
        //     {
        //         std::cout << k << std::endl;
        //     }


        //     i += len;  // Move to the next character
        // }



        // assert(!token_vector.empty());
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


std::string join(const std::vector<int>& vec, const std::string& delimiter = ".") {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << delimiter;
        }
    }
    return oss.str();
}

int MidiTokenizer::_time_token_to_ticks(int beat, int pos, int res, int ticks_per_beat)
{
    return (beat * res + pos) * ticks_per_beat / res;
}

std::vector<std::pair<int, int>> MidiTokenizer::__create_time_signatures() const
{
    std::vector<std::pair<int, int>> time_signatures;

    for (auto& [beat_res, beats] : _time_signature_range)
    {
        if (beat_res <= 0)
        {
            throw std::runtime_error("The beat resolution in time signature must be a power 2.");
        }
        double v = std::log2(beat_res);
        if (v - (std::floor(v) < 0.001))
        {
            throw std::runtime_error("The beat resolution in time signature must be a power 2.");
        }

        for (int num_beats : beats)
        {
            time_signatures.emplace_back(num_beats, beat_res);
        }
    }

    return time_signatures;
}

std::map<int, int> MidiTokenizer::__create_tpb_per_ts()
{
    int max_denom = -1;
    for (auto& [num_beats, beat_res] : time_signatures)
    {
        max_denom = std::max(max_denom, beat_res);
    }

    std::map<int, int> out;
    for (const auto& denom : _time_signature_range)
    {
        out[denom.first] = max_num_pos_per_beat() * (max_denom / denom.first);
    }
    return out;
}


std::vector<std::tuple<int, int, int>> MidiTokenizer::_create_durations_tuples() const
{
    std::vector<std::tuple<int, int, int>> durations;

    for (const auto& [beat_range, beat_res] : _beat_res)
    {
        for (int beat = beat_range.first; beat < beat_range.second; beat++)
        {
            for (int pos = 0; pos < beat_res; pos++)
            {
                durations.emplace_back(beat, pos, beat_res);
            }
        }
    }

    int max = 0;
    int maxPairValue;
    for (auto& [pair, i] : _beat_res)
    {
        if (max < pair.second)
        {
            max = pair.second;
            maxPairValue = i;
        }
    }

    durations.emplace_back(max, 0, maxPairValue);

    durations.erase(durations.begin());

    return durations;
}

// Create the correspondences between times in tick and token value (str).

// These correspondences vary following the ticks/beat value, which depends on the
// time signature.

// The returned dictionary is used when decoding *Duration*/*TimeShift*/*Rest*
// tokens while taking the time signature into account.

// :param rest: will use rest values if given ``True``, otherwise durations.
//     (default: ``False``)
// :return: ticks per beat + token value to duration in tick.
std::map<int, std::map<std::string, int>> MidiTokenizer::__create_tpb_tokens_to_ticks(bool rest)
{
    std::map<int, std::map<std::string, int>> tpb_tokens_to_ticks;

    if (rest)
    {
        throw std::runtime_error("__create_tpb_tokens_to_ticks() : rest unimplemented");
    }
    // const std::vector<std::tuple<int, int, int>>& values = rests ? rest : durations;
    const std::vector<std::tuple<int, int, int>>& values = durations;

    for (const auto& tpb : _tpb_per_ts)
    {
        std::map<std::string, int> ticks;
        for (const auto& duration_tuple : values)
        {
            int beat = std::get<0>(duration_tuple);
            int pos = std::get<1>(duration_tuple);
            int res = std::get<2>(duration_tuple);

            std::vector<int> duration_vec = {beat, pos, res};
            std::string key = join(duration_vec); // Create key as "a.b.c" from duration_tuple
            int value = _time_token_to_ticks(beat, pos, res, tpb.first); // Compute value
            ticks[key] = value;
        }

        tpb_tokens_to_ticks[tpb.first] = std::move(ticks);
    } 
    return tpb_tokens_to_ticks;
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
                // assert(k.size() <= 2);
                // std::string k2 = decode_utf8_to_binary(k);
                std::string vStr = v.template get<std::string>();

                // // if (k2.size() == 1)
                // // {
                // //     std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(k2[0]); // Display each character as hex
                // //     std::cout << ": " << vStr << std::endl;
                // // }
                // // else 
                // // {
                // //     std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(k2[0]); // Display each character as hex
                // //     std::cout << " ";
                // //     std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(k2[1]); // Display each character as hex
                // //     std::cout << ": " << vStr << std::endl;
                // // }

                // // Split the string
                // std::vector<std::string> split_str = split_utf8(key_);

                // // Output the split parts
                // for (const std::string& part : split_str) {
                //     token_vector.push_back(_vocab_base_byte_to_token.at(part));
                // }

                assert(_vocab_base_byte_to_token.count(k) == 0);

                _vocab_base_byte_to_token[k] = std::move(vStr);
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
            {
                auto it = value.find("use_velocities");
                if (it != value.end())
                {
                    bUseVelocities = it->get<bool>();
                }
            }

            {
                auto it = value.find("use_time_signatures");
                if (it != value.end())
                {
                    bUseTimeSignatures = it->get<bool>();
                }
            }

            {
                auto it = value.find("beat_res");
                if (it != value.end())
                {
                    for (auto& [k, v] : it->items())
                    {
                        size_t delimiterPos = k.find('_');
                        if (delimiterPos != std::string::npos) 
                        {
                            std::string leftStr = k.substr(0, delimiterPos);
                            std::string rightStr = k.substr(delimiterPos + 1);
                            std::pair<int, int> key;
                            try
                            {
                                key = {std::stoi(leftStr), std::stoi(rightStr)};
                            }
                            catch (const std::exception&)
                            {
                                throw std::runtime_error("beat_res: invalid syntax in config");
                            }
                            _beat_res[key] = v;
                        }
                    }
                }
            }

            {
                auto it = value.find("time_signature_range");
                if (it != value.end())
                {
                    for (auto& [res, beat_range] : it->items())
                    {
                        _time_signature_range[std::stoi(res)] = beat_range.get<std::vector<int>>();
                    }
                }
            }

            {
                auto it = value.find("beat_res_rest");
                if (it != value.end())
                {
                    for (auto& [k, v] : it->items())
                    {
                        size_t delimiterPos = k.find('_');
                        if (delimiterPos != std::string::npos) 
                        {
                            std::string leftStr = k.substr(0, delimiterPos);
                            std::string rightStr = k.substr(delimiterPos + 1);
                            std::pair<int, int> key;
                            try
                            {
                                key = {std::stoi(leftStr), std::stoi(rightStr)};
                            }
                            catch (const std::exception&)
                            {
                                throw std::runtime_error("beat_res: invalid syntax in config");
                            }
                            _beat_res[key] = v;
                        }
                    }
                }
            }

            {
                auto it = value.find("rest_range");
                if (it != value.end())
                {
                    throw std::runtime_error("rest_range unimplemented yet");
                }
            }

            {
                auto it = value.find("chord_maps");
                if (it != value.end())
                {
                    for (auto& [k, v] : it->items())
                    {
                        _chord_maps[k] = v.get<std::vector<int>>();
                    }
                }
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

std::vector<int32_t> MidiTokenizer::encode(const Score& score)
{
    return std::vector<int32_t>();
}

TokSequence MidiTokenizer::_convert_sequence_to_tokseq(const std::vector<int32_t>& tokens) const
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

void MidiTokenizer::_preprocess_tokseq_before_decoding(TokSequence& seq) const
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

void MidiTokenizer::complete_sequence(TokSequence& seq, bool complete_bytes) const
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

std::vector<int32_t> MidiTokenizer::_tokens_to_ids(const std::vector<std::string>& tokens) const
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

void MidiTokenizer::decode_token_ids(TokSequence& seq) const
{
    if (seq.are_ids_encoded)
    {
        std::vector<std::string> encoded_bytes(seq.ids.size()); 
        for (size_t i = 0; i < seq.ids.size(); i++)
        {
            auto& id_ = seq.ids[i];
            std::mutex& m = const_cast<std::mutex&>(_modelMutex);
            m.lock();
            encoded_bytes[i] = _model->IdToToken(id_);
            m.unlock();
            assert(!encoded_bytes[i].empty()); // if empty, it means id hasn't been found
        } 

        std::vector<std::string> decoded_tokens;
        decoded_tokens.reserve(encoded_bytes.size()); 
        for (std::string& byte_ : encoded_bytes)
        {
            const std::vector<std::string>& decoded = _vocab_learned_bytes_to_tokens.at(byte_); 
            // assert(!decoded.empty());
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

void MidiTokenizer::decodeToken(std::int32_t encodedToken, std::vector<int32_t>& outDecodedTokens) const
{
    std::vector<std::int32_t> inTokensVec(1);
    inTokensVec[0] = encodedToken;

    decodeIDs(inTokensVec, outDecodedTokens);
}

Score MidiTokenizer::decode(const std::vector<int32_t>& tokens)
{
    TokSequence tokSequence = _convert_sequence_to_tokseq(tokens);

    _preprocess_tokseq_before_decoding(tokSequence);

    Score score = _tokens_to_score(tokSequence);


    return score;
}

void MidiTokenizer::decodeIDs(const std::vector<int32_t>& tokens, std::vector<int32_t>& outTokens) const
{
    TokSequence tokSequence = _convert_sequence_to_tokseq(tokens);

    _preprocess_tokseq_before_decoding(tokSequence);

    outTokens = std::move(tokSequence.ids);
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

void MidiTokenizer::addTokensStartingByPosition(RangeGroup& outRangeGroup)
{
    std::vector<int32_t> decodedTokens(10);
    for (std::int32_t token = 0; token < _vocab_learned_bytes_to_tokens.size(); token++)
    {
        try 
        {
            decodeToken(token, decodedTokens);
        }
        catch (const std::exception&)
        {
            continue;
        }

        // for (std::int32_t i = 0; i < decodedTokens.size(); i++)
        if (!decodedTokens.empty())
        {
            std::int32_t decodedToken = decodedTokens[0];
            if (isPosition(decodedToken))
            {
                outRangeGroup.add(token);
            }
        }
    }
}
void MidiTokenizer::addTokensStartingByBarNone(RangeGroup& outRangeGroup)
{
    std::vector<int32_t> decodedTokens(10);
    for (std::int32_t token = 0; token < _vocab_learned_bytes_to_tokens.size(); token++)
    {
        try 
        {
            decodeToken(token, decodedTokens);
        }
        catch (const std::exception&)
        {
            continue;
        }

        // for (std::int32_t i = 0; i < decodedTokens.size(); i++)
        if (!decodedTokens.empty())
        {
            std::int32_t decodedToken = decodedTokens[0];
            if (isBarNone(decodedToken))
            {
                outRangeGroup.add(token);
            }
        }
    }
}
void MidiTokenizer::addTokensStartingByPitch(RangeGroup& outRangeGroup)
{
    std::vector<int32_t> decodedTokens(10);
    for (std::int32_t token = 0; token < _vocab_learned_bytes_to_tokens.size(); token++)
    {
        try 
        {
            decodeToken(token, decodedTokens);
        }
        catch (const std::exception&)
        {
            continue;
        }

        // for (std::int32_t i = 0; i < decodedTokens.size(); i++)
        if (!decodedTokens.empty())
        {
            std::int32_t decodedToken = decodedTokens[0];
            if (isPitch(decodedToken))
            {
                outRangeGroup.add(token);
            }
        }
    }
}
void MidiTokenizer::addTokensStartingByVelocity(RangeGroup& outRangeGroup)
{
    std::vector<int32_t> decodedTokens(10);
    for (std::int32_t token = 0; token < _vocab_learned_bytes_to_tokens.size(); token++)
    {
        try 
        {
            decodeToken(token, decodedTokens);
        }
        catch (const std::exception&)
        {
            continue;
        }

        // for (std::int32_t i = 0; i < decodedTokens.size(); i++)
        if (!decodedTokens.empty())
        {
            std::int32_t decodedToken = decodedTokens[0];
            if (isVelocity(decodedToken))
            {
                outRangeGroup.add(token);
            }
        }
    }
}
void MidiTokenizer::addTokensStartingByDuration(RangeGroup& outRangeGroup)
{
    std::vector<int32_t> decodedTokens(10);
    for (std::int32_t token = 0; token < _vocab_learned_bytes_to_tokens.size(); token++)
    {
        try 
        {
            decodeToken(token, decodedTokens);
        }
        catch (const std::exception&)
        {
            continue;
        }

        // for (std::int32_t i = 0; i < decodedTokens.size(); i++)
        if (!decodedTokens.empty())
        {
            std::int32_t decodedToken = decodedTokens[0];
            if (isDuration(decodedToken))
            {
                outRangeGroup.add(token);
            }
        }
    }
}

const std::string& MidiTokenizer::decodedTokenToString(int32_t decodedToken)
{
    return __vocab_base_inv[decodedToken];
}