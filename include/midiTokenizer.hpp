#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <cassert>
#include "json.hpp"
#include "fwd.h"

#include "tokenizers_cpp.h"

#include "tokenSequence.h"
#include <mutex>

using json = nlohmann::json;

using Score = std::vector<int32_t>;

inline constexpr std::pair<int, int> TIME_SIGNATURE = std::pair<int, int>(4,4);

// https://github.com/Natooz/MidiTok/blob/main/miditok/midi_tokenizer.py
class MidiTokenizer
{
private:
    // the other way, to decode id (int) -> token (str)
    std::map<int32_t, std::string> __vocab_base_inv;

    bool bUseVelocities = false;
    bool bUseDuration = false;
    bool bUseTimeSignatures = false;

    // "REMI", "TSD"
    std::string tokenization;

protected:    
    // vocab of prime tokens, can be viewed as unique char / bytes
    std::map<std::string, int32_t> _vocab_base;

    // byte (str) -> token (str), for basic tokens
    std::map<std::string, std::string> _vocab_base_byte_to_token;

    // id (int) -> byte (str), as this might not be chr(id) after tokenizer training
    std::map<int32_t, std::string> _vocab_base_id_to_byte;

    // Fast tokenizer model backed with 🤗tokenizers
    std::unique_ptr<tokenizers::Tokenizer> _model;
    std::mutex _modelMutex;


    // TODO / TO LOAD
    // byte(s) -> token(s), for faster BPE/Unigram/WordPiece decoding
    std::map<std::string, std::vector<std::string>> _vocab_learned_bytes_to_tokens;


    std::map<std::pair<int, int>, int32_t> _beat_res;
    std::map<int, std::vector<int>> _time_signature_range;
    std::map<std::string, std::vector<int>> _chord_maps;

    // Cache
    std::vector<int32_t> decodedTokens;
    std::vector<std::int32_t> encodedTokenToDecodedTokensBeginIndex;

    std::vector<uint8_t> decodedTokenToPitch;

public:

    std::map<int, int> _tpb_per_ts;
    std::map<int, std::map<std::string, int>> _tpb_tokens_to_ticks;
    std::map<int, std::map<std::string, int>> _tpb_rests_to_ticks;

    std::vector<std::pair<int, int>> time_signatures;
    std::vector<std::tuple<int, int, int>> durations;

    int time_division = 0;

private:
    void __create_vocab_learned_bytes_to_tokens();
    std::map<int, int> __create_tpb_per_ts();

    // Create the possible durations in beat / position units as tuples of intergers.

    // The tuples follow the form: ``(beat, pos, res)`` where ``beat`` is the number
    // of beats, ``pos`` the number of "positions" and ``res`` the beat resolution
    // considered (positions per beat).
    // Example: ``(2, 5, 8)`` means the duration is 2 beat long + position 5 / 8 of
    // the ongoing beat. This would give in ticks:
    // ``duration = (beat * res + pos) * ticks_per_beat // res``
    // Note that ``ticks_per_beat`` is different from the time division, as the number
    // of ticks per beat depends on the current time signature denominator.
    // If ticks_per_beat is 384:
    // ``duration = (2 * 8 + 5) * 384 // 8 = 1008`` ticks.

    // :return: the duration bins.
    std::vector<std::tuple<int, int, int>> _create_durations_tuples() const;

    // Create the correspondences between times in tick and token value (str).

    // These correspondences vary following the ticks/beat value, which depends on the
    // time signature.

    // The returned dictionary is used when decoding *Duration*/*TimeShift*/*Rest*
    // tokens while taking the time signature into account.

    // :param rest: will use rest values if given ``True``, otherwise durations.
    //     (default: ``False``)
    // :return: ticks per beat + token value to duration in tick.
    std::map<int, std::map<std::string, int>> __create_tpb_tokens_to_ticks(bool rest = false);

    // Create time signatures of the vocabulary, as tuples of integers.

    // The tuples have the form ``(num_beats, beat_res)`` where ``num_beats`` is the
    // number of beats per bar.
    // Example: ``(3, 4)`` means one bar is 3 beat long and each beat is a quarter
    // note.

    // :return: the time signatures.
    std::vector<std::pair<int, int>> __create_time_signatures() const;

protected:
    TokSequence _convert_sequence_to_tokseq(const std::vector<int32_t>& tokens) const;
    void loadFromJson(const std::string& filename);
    bool _are_ids_encoded(const std::vector<int32_t>& tokens) const;
    void _preprocess_tokseq_before_decoding(TokSequence& seq) const;

    // Convert a list of tokens (str) into their ids format (int).

    // :param tokens: list of tokens (str) to convert.
    // :return: list of corresponding ids (int).
    std::vector<int32_t> _tokens_to_ids(const std::vector<std::string>& tokens) const;


    // Convert a time token value of the form beat.position.resolution, in ticks.

    // This method is used to decode time tokens such as *Duration*, *TimeShift* or
    // *Rest*.

    // :param token_duration: Duration / TimeShift token value.
    // :param ticks_per_beat: number of ticks in a beat. This depends on the current
    //     time signature, and is equal to the Score's time division if the denominator
    //     is 4 (quarter).
    // :return: the duration / time-shift in ticks.
    std::int32_t _time_token_to_ticks(const std::string& token_duration, std::int32_t ticks_per_beat);
    std::int32_t _time_token_to_ticks(std::int32_t beat, std::int32_t pos, std::int32_t res, std::int32_t ticks_per_beat);


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
        loadFromJson(filename);
            
        // Time Signatures
        // Need to be set before creating duration values/tokens.
        time_signatures = {TIME_SIGNATURE};
        if (bUseTimeSignatures)
        {
            time_signatures = __create_time_signatures();
        }

        _tpb_per_ts = __create_tpb_per_ts();

        time_division = _tpb_per_ts[TIME_SIGNATURE.second];


        // Durations
        // Usages:
        // Duration: tpb --> np.array (ticks) to get the closest;
        // Duration/TimeShift/Rest: ticks + tpb --> token (str);
        // Duration/TimeShift/Rest: token + tpb --> ticks (int);
        durations = _create_durations_tuples();
        _tpb_tokens_to_ticks = __create_tpb_tokens_to_ticks();

        // Rests
        // _tpb_rests_to_ticks = __create_tpb_tokens_to_ticks(true);

        __createDecodingCache();
        __createDecodedToPitchCache();
    }

    void __createDecodingCache()
    {
        int32_t nbEncodedTokens = getNbEncodedTokens();
        encodedTokenToDecodedTokensBeginIndex.resize(nbEncodedTokens+1);

        std::vector<int32_t> outDecodedTokens;
        for (int32_t encodedToken = 0; encodedToken < nbEncodedTokens; ++encodedToken)
        {
            const int32_t begin = int32_t(decodedTokens.size());
            encodedTokenToDecodedTokensBeginIndex[encodedToken] = begin;

            outDecodedTokens.clear();
            try
            {
                decodeToken(encodedToken, outDecodedTokens);
                for (const int32_t decodedToken : outDecodedTokens)
                {
                    decodedTokens.push_back(decodedToken);
                }
            }
            catch (const std::exception&) {}
        }
        encodedTokenToDecodedTokensBeginIndex.back() = int32_t(decodedTokens.size());
    }

    void __createDecodedToPitchCache()
    {
        int32_t nbDecodedTokens = getNbDecodedTokens();
        decodedTokenToPitch.resize(nbDecodedTokens);
        for (int32_t decodedToken = 0; decodedToken < nbDecodedTokens; ++decodedToken)
        {
            if (isPitch(decodedToken))
            {
                decodedTokenToPitch[decodedToken] = int8_t(getPitchValue(decodedToken));
            }
        }
    }

    int max_num_pos_per_beat() const
    {
        int max = 0;
        for (const auto& [k, v] : _beat_res)
        {
            max = std::max(max, v);
        }
        return max;
    }

    // Decode a single token
    // @TODO : optimize by simply setting a pointer (with size) instead of using a std::vector
    void decodeToken(std::int32_t encodedToken, std::vector<int32_t>& outDecodedTokens) const;
    void decodeTokenFast(std::int32_t encodedToken, const int32_t*& outDecodedTokensBegin, const int32_t*& outDecodedTokensEnd) const
    {
        const std::int32_t begin = encodedTokenToDecodedTokensBeginIndex[encodedToken];
        const std::int32_t end = encodedTokenToDecodedTokensBeginIndex[encodedToken+1];

        outDecodedTokensBegin = decodedTokens.data() + begin;
        outDecodedTokensEnd = decodedTokens.data() + end;
    }

    std::vector<int32_t> encode(const Score& score);
    Score decode(const std::vector<int32_t>& tokens);

    // custom
    void decodeIDs(const std::vector<int32_t>& tokens, std::vector<int32_t>& outTokens) const;

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
    void decode_token_ids(TokSequence& seq) const;



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
    void complete_sequence(TokSequence& seq, bool complete_bytes = false) const;

    bool is_trained() const;

    std::int32_t getNbDecodedTokens() const
    {
        return static_cast<std::int32_t>(_vocab_base.size());
    }

    std::int32_t getNbEncodedTokens() const
    {
        return static_cast<std::int32_t>(_vocab_learned_bytes_to_tokens.size());
    }

public:
    // Optimized conversions
    // @TODO : implement directly with hashmaps

    static bool startBy(const char* str, const char* startBy)
    {
        int i = 0;
        while (str[i] == startBy[i] && str[i] != '\0' && startBy[i] != '\0')
        {
            i++;
        }
        return startBy[i] == '\0';
    }

    static std::string getUniqueValueStr(const std::string& str)
    {
        auto it = std::find(str.begin(), str.end(), '_');
        std::int32_t typeSize = std::int32_t(it - str.begin());
        return str.substr(it - str.begin()+1, str.size() - typeSize-1);
    }

    static std::int32_t getUniqueValueInt(const std::string& str)
    {
        return std::stoi(getUniqueValueStr(str));
    }

    bool isBarNone(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);

        // auto it = std::find(str.begin(), str.end(), '_');
        // std::int32_t typeSize = it - str.begin();
        // std::string type = str.substr(0, typeSize);
        // std::string value = str.substr(it - str.begin(), str.size() - typeSize);

        return startBy(str.c_str(), "Bar_None");
    }

    bool isTimeShift(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "TimeShift_");
    }

    std::string getTimeShiftValue(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueStr(str);
    }

    bool isPosition(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Position_");
    }

    std::int32_t getPositionValue(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueInt(str);
    }

    bool isPitch(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Pitch_");
    }

    bool isPitchFast(std::int32_t token) const
    {
        return decodedTokenToPitch[token] != 0;
    }

    std::int32_t getPitchValue(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueInt(str);
    }

    std::uint8_t getPitchValueFast(std::int32_t token) const
    {
        return decodedTokenToPitch[token];
    }

    bool isDuration(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Duration_");
    }

    std::int32_t getDurationValue(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueInt(str);
    }

    bool isVelocity(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Velocity_");
    }

    std::int32_t getVelocityValue(std::int32_t token)
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueInt(str);
    }

    bool isRest(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Rest_");
    }

    std::string getRestValue(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueStr(str);
    }

    bool isProgram(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Program_");
    }

    std::int32_t getProgramValue(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return getUniqueValueInt(str);
    }

    bool isTempo(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Tempo_");
    }

    bool isTimeSig(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "TimeSig_");
    }

    bool isPedal(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "Pedal_");
    }

    bool isPedalOff(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "PedalOff_");
    }

    bool isPitchBend(std::int32_t token) const
    {
        const std::string& str = __vocab_base_inv.at(token);
        return startBy(str.c_str(), "PitchBend_");
    }

    void addTokensStartingByPosition(RangeGroup& outRangeGroup);
    void addTokensStartingByBarNone(RangeGroup& outRangeGroup);
    void addTokensStartingByPitch(RangeGroup& outRangeGroup);
    void addTokensStartingByVelocity(RangeGroup& outRangeGroup);
    void addTokensStartingByDuration(RangeGroup& outRangeGroup);


    const std::string& decodedTokenToString(int32_t decodedToken);

    bool useVelocities() const
    {
        return bUseVelocities;
    }

    bool useDuration() const
    {
        return bUseDuration;
    }

    bool useTimeSignatures() const
    {
        return bUseDuration;
    }

    const char* getTokenizationType() const
    {
        return tokenization.c_str();
    }
};


