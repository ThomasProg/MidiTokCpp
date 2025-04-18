#pragma once

#include <vector>

class Sequencer
{
    // sorted
    // tick to callback
    using Callback = void(*)(int32_t tick, void* userData);
    std::vector<std::pair<int32_t, Callback>> tickToCallback;

    size_t currentIndex = 0;
    int32_t currentTick = 0;
    void* userData = nullptr;

public:
    void reset()
    {
        currentIndex = 0;
    }

    void setUserData(void* inUserData)
    {
        userData = inUserData;
    }

    void advance(int32_t newTick)
    {
        while (currentIndex < tickToCallback.size() && newTick >= tickToCallback[currentIndex].first)
        {
            tickToCallback[currentIndex].second(tickToCallback[currentIndex].first, userData);
            currentIndex++;
        }
        currentTick = newTick;
    }

    void addCallback(int32_t tick, Callback callback)
    {
        if (currentIndex < tickToCallback.size() && tick <= currentTick)
        {
            callback(tickToCallback[currentIndex].first, userData);
            currentIndex++;
        }

        auto it = std::lower_bound(tickToCallback.begin(), tickToCallback.end(), tick, [](const std::pair<int32_t, Callback>& pair, int32_t addedCallbackTick)
        {
            return pair.first < addedCallbackTick;
        });
        tickToCallback.insert(it, std::pair<int32_t, Callback>(tick, callback));
    }
};
