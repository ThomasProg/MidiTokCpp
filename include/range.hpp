#pragma once

#include <cstdint>
#include <vector>
#include <cassert>
#include "range.h"

static inline bool isColliding(const Range& a, const Range& b)
{
    return isColliding(&a, &b);
}

// Ranges are sorted 
// (-5;-1)(2;4) (6;10) (16;19)
// Ranges are merged
// (0;4) (5;15) -> (0;15)
class RangeGroup
{
    // Sorted
    std::vector<Range> ranges;
    size_t totalSize = 0;

    std::vector<int32_t> fullSequenceCache;
    bool isDirty = false;

public:
    RangeGroup() = default;
    RangeGroup(const RangeGroup& rhs) : ranges(rhs.ranges), totalSize(rhs.totalSize), isDirty(true) {}
    RangeGroup(RangeGroup&& rhs) = default;

    RangeGroup& operator=(const RangeGroup& rhs) = delete;

    // RangeGroup& operator=(const RangeGroup& rhs)
    // {
    //     ranges = rhs.ranges;
    //     totalSize = rhs.totalSize; 
    // }
    RangeGroup& operator=(RangeGroup&& rhs) = default;
    ~RangeGroup() = default;

    // Uses dichotomy search
    // Returns an index
    std::size_t findLastRangeBeforeX(std::int32_t x, std::size_t beginIndex, std::size_t endIndex) const;

    std::size_t findLastRangeBeforeX(std::int32_t x) const
    {
        return findLastRangeBeforeX(x, 0, ranges.size()-1);
    }

    bool findRange(std::int32_t x, std::size_t beginIndex, std::size_t endIndex, std::size_t& foundIndex) const;

    inline bool findRange(std::int32_t x, std::size_t& foundIndex) const
    {
        return findRange(x, 0, ranges.size()-1, foundIndex);
    }

    inline void add(std::int32_t x)
    {
        isDirty = true;
        addRange({x,x});
    }

    void addRange(Range newRange);

    inline const std::vector<Range>& getRanges() const
    {
        return ranges;
    }

    inline std::size_t size() const
    {
        return totalSize;
        // return computeSize();
    }

    std::size_t computeSize() const;

    // write all ints in that RangeGroup inside an array
    // size of the array must be size()
    void write(int32_t* writeBuffer) const;

    void updateCache()
    {
        if (isDirty)
        {
            fullSequenceCache.resize(size());
            write(fullSequenceCache.data());
            isDirty = false;
        }
    }

    using iterator = std::vector<int32_t>::iterator;
    using reverse_iterator = std::vector<int32_t>::reverse_iterator;
    using const_iterator = std::vector<int32_t>::const_iterator;
    using const_reverse_iterator = std::vector<int32_t>::const_reverse_iterator;

    iterator begin() { assert(!isDirty); return fullSequenceCache.begin(); }
    iterator end()   { assert(!isDirty); return fullSequenceCache.end(); }
    const_iterator begin() const { assert(!isDirty); return fullSequenceCache.cbegin(); }
    const_iterator end() const  { assert(!isDirty); return fullSequenceCache.cend(); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const  { return end(); }

    reverse_iterator rbegin() { assert(!isDirty); return fullSequenceCache.rbegin(); }
    reverse_iterator rend()   { assert(!isDirty); return fullSequenceCache.rend(); }
    const_reverse_iterator rcbegin() const { assert(!isDirty); return fullSequenceCache.crbegin(); }
    const_reverse_iterator rcend() const  { assert(!isDirty); return fullSequenceCache.crend(); }
};


