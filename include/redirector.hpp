#pragma once

#include <functional>
#include <string>
#include <unordered_map>

class Redirector
{
    struct Callback
    {
        using TCallback = std::function<void(void*)>;

        void* data;
        TCallback callback;

        void call()
        {
            callback(data);
        }
    };

    std::unordered_map<int32_t, Callback> tokenToCallback;

public:
    bool tryCall(int32_t token);

    void bind(const class MidiTokenizer& tokenizer, const std::string& newKey, std::function<Callback::TCallback(std::string&& str)> bindWithParams, void* data = nullptr);

    void bindPitch(const class MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t pitch), void* data = nullptr);
    void bindPosition(const class MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t position), void* data = nullptr);
    void bindBar(const class MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, std::uint8_t bar, bool isBarNone), void* data = nullptr);
};