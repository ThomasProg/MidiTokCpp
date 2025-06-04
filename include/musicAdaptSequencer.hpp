#pragma once

#include <vector>

class Sequencer
{
    using Hash = int32_t;
    using Callback = void(*)(int32_t hash, int32_t tick, void* userData);
    struct EventData
    {
        Hash hash;
        int32_t tick;
        Callback callback;
        Callback undoCallback;
    };

    // sorted by tick
    // tick to callback
    std::vector<EventData> events;

    size_t currentIndex = 0;
    int32_t currentTick = 0;
    Hash nextHash = 1;
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
        while (currentIndex < events.size() && newTick >= events[currentIndex].tick)
        {
            events[currentIndex].callback(events[currentIndex].hash, events[currentIndex].tick, userData);
            currentIndex++;
        }
        currentTick = newTick;
    }

    void rewind(int32_t newTick)
    {
        while (currentIndex > 0 && newTick < events[currentIndex-1].tick)
        {
            currentIndex--;
            Callback undo = events[currentIndex].undoCallback;
            if (undo)
            { 
                undo(events[currentIndex].hash, events[currentIndex].tick, userData);
            }
        }
        currentTick = newTick;
    }

    void addEventData(const EventData& eventData)
    {
        // if (eventData.tick <= currentTick)
        // {
        //     eventData.callback(eventData.hash, eventData.tick, userData);
        //     currentIndex++;
        // }

        auto it = std::lower_bound(events.begin(), events.end(), eventData.tick, [](const EventData& eventData, int32_t addedCallbackTick)
        {
            return eventData.tick < addedCallbackTick;
        });
        events.insert(it, std::move(eventData));
    }

    Hash addCallback(int32_t tick, Callback callback, Callback undo)
    {
        int32_t currentHash = nextHash;

        EventData eventData;
        eventData.callback = callback;
        eventData.tick = tick;
        eventData.hash = currentHash;
        eventData.undoCallback = undo;

        addEventData(eventData);

        // compute next hash
        nextHash += 1;

        return currentHash;
    }

    void removeCallback(std::vector<EventData>::iterator it)
    {
        auto next = events.erase(it);

        if (size_t(next - events.begin()) <= currentIndex)
        {
            currentIndex = std::max(currentIndex-1, 0llu);
        }
    }

    // can be optimized with a map?
    void removeCallback(Hash hash)
    {
        removeCallback(std::find_if(events.begin(), events.end(), [hash](const EventData& elem)
        {
            return elem.hash == hash;
        }));
    }

    void updateCallbackTick(Hash hash, int32_t tick)
    {
        auto it = std::find_if(events.begin(), events.end(), [hash](const EventData& elem)
        {
            return elem.hash == hash;
        });

        EventData data = *it;
        data.tick = tick;
        removeCallback(it);
        addEventData(data);
    }
};
