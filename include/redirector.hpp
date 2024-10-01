#pragma once

#include <functional>
#include <string>
#include <unordered_map>

class Redirector
{
    struct Callback
    {
        void* data;
        std::function<void(void*)> callback;

        void call()
        {
            callback(data);
        }
    };

    std::unordered_map<int32_t, Callback> tokenToCallback;

public:
    void call(int32_t token);
    void bindPitch(const class MidiTokenizer& tokenizer, const std::string& newKey, void (*newCallback)(void*, unsigned char/*int8*/ pitch), void* data = nullptr);
};