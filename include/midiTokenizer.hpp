#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "json.hpp"

using json = nlohmann::json;

// https://github.com/Natooz/MidiTok/blob/main/miditok/midi_tokenizer.py
class MidiTokenizer
{
private:
    std::map<int, std::string> __vocab_base_inv;

protected:    
    std::map<std::string, int> _vocab_base;
    std::map<char, std::string> _vocab_base_byte_to_token;

    std::map<char, std::string> _vocab_base_id_to_byte;

private:
    void __create_vocab_learned_bytes_to_tokens()
    {
        // @TODO
    }

protected:

    void loadFromJson(const std::string& filename)
    {
        std::ifstream file(filename);

        if(!file)
        {
            std::cout << "Couldn't load" << std::endl;
            return;
        } 


        json data = json::parse(file);



        // Grab config, or creates one with default parameters (for retro-compatibility
        // with previous version)
        // self.config = TokenizerConfig()
        // config_attributes = list(self.config.to_dict().keys())
        // std::vector<> config_attributes
        // std::map<std::string, std::string> old_add_tokens_attr = {
        //     {"Chord", "use_chords"},
        //     {"Rest", "use_rests"},
        //     {"Tempo", "use_tempos"},
        //     {"TimeSignature", "use_time_signatures"},
        //     {"Program", "use_program"},
        // };


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
                json modelData = json::parse(value.template get<std::string>());

                for (auto& [key2, value2] : modelData.items())
                {
                    if (key2 != "model")
                    std::cout << key2 << " / " << value2 << std::endl;

                }
                
                // @TODO



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


};


