#include "redirector.hpp"
#include <map>
#include "midiTokenizer.hpp"

bool Redirector::tryCall(int32_t token)
{
    auto it = tokenToCallback.find(token);
    if (it == tokenToCallback.end())
        return false;

    it->second.call();
    return true;
}

void Redirector::bind(const class MidiTokenizer& tokenizer, const std::string& newKey, std::function<Callback::TCallback(std::string&& str)> bindWithParams, void* data)
{
    const std::map<std::string, int32_t>& vocab = tokenizer.GetVocabBase();

    for (auto& [key, value] : vocab)
    {
        if (key.size() >= newKey.size() && key.compare(0, newKey.size(), newKey) == 0)
        {
            std::string args = key.substr(newKey.size());

            Callback callback;
            callback.data = data;
            callback.callback = bindWithParams(std::move(args));
            tokenToCallback[value] = callback;
        }
    }
}

void Redirector::bindPitch(const MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t pitch), void* data)
{
    bind(tokenizer, newKey, [newCallback](std::string&& args) -> std::function<void(void*)>
    {
        // Since it's bindPitch, we know args is just a single int
        std::uint8_t arg = std::uint8_t(std::stoi(args));

        return [newCallback, arg](void* data)
            {
                newCallback(data, arg);
            };

    }, data);
}

void Redirector::bindPosition(const class MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t position), void* data)
{
    bind(tokenizer, newKey, [newCallback](std::string&& args) -> std::function<void(void*)>
    {
        // Since it's bindPosition, we know args is just a single int
        std::uint8_t arg = std::uint8_t(std::stoi(args));

        return [newCallback, arg](void* data)
            {
                newCallback(data, arg);
            };

    }, data);
}

void Redirector::bindBar(const class MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t bar, bool isBarNone), void* data)
{
    bind(tokenizer, newKey, [newCallback](std::string&& args) -> std::function<void(void*)>
    {
        // Since it's bindBar, we know args is just a single int, OR None
        if (args == "None")
        {
            return [newCallback](void* data)
                {
                    newCallback(data, 0, true);
                };
        }

        std::uint8_t arg = std::uint8_t(std::stoi(args));

        return [newCallback, arg](void* data)
            {
                newCallback(data, arg, false);
            };

    }, data);
}