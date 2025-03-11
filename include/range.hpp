#pragma once

#include <cstdint>
#include <vector>
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

public:
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
    }

    // write all ints in that RangeGroup inside an array
    // size of the array must be size()
    void write(int32_t* writeBuffer);
};


