#include "midiTokenizer.hpp"

#include <memory>

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
                _vocab_base_byte_to_token[k[0]] = v;
            }

            std::map<std::string, char> tokenToByte;
            for (auto& [k, v] : value.items())
            {
                tokenToByte[v] = k[0];
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

Score MidiTokenizer::decode(const TokSequenceInt& tokens)
{
    return Score();
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