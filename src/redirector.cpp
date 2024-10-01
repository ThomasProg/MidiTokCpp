#include "redirector.hpp"
#include <map>
#include "midiTokenizer.hpp"

void Redirector::call(int32_t token)
{
    tokenToCallback.at(token).call();
}

void Redirector::bindPitch(const MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, unsigned char/*uint8*/ pitch), void* data)
{
    const std::map<std::string, int32_t>& vocab = tokenizer.GetVocabBase();

    for (auto& [key, value] : vocab)
    {
        if (key.size() >= newKey.size() && key.compare(0, newKey.size(), newKey) == 0)
        {
            std::string args = key.substr(newKey.size());

            // Since it's bindPitch, we know args is just a single int
            std::uint8_t arg = std::uint8_t(std::stoi(args));

            Callback callback;
            callback.data = data;
            callback.callback = [newCallback, arg](void* data)
            {
                newCallback(data, arg);
            };

            tokenToCallback[value] = callback;
        }
    }
}