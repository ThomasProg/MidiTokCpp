#include "range.h"
#include "range.hpp"

std::size_t RangeGroup::findLastRangeBeforeX(std::int32_t x, std::size_t beginIndex, std::size_t endIndex) const
{
    if (beginIndex == endIndex)
    {
        if (ranges[beginIndex].max < x)
        {
            return beginIndex;
        }
        else // if x is less than ranges[0]
        {
            return ranges.size(); // return invalid index
        }
    }

    std::size_t middleIndex = (endIndex + beginIndex) / 2;
    if (ranges[middleIndex].max < x)
    {
        return findLastRangeBeforeX(x, beginIndex, middleIndex);
    }
    else
    {
        return findLastRangeBeforeX(x, middleIndex+1, endIndex);
    }
}

bool RangeGroup::findRange(std::int32_t x, std::size_t beginIndex, std::size_t endIndex, std::size_t& foundIndex) const
{
    if (beginIndex > endIndex)
    {
        return false;
    }

    // middle index
    foundIndex = (endIndex + beginIndex) / 2;
    if (x < ranges[foundIndex].min)
    {
        // @TODO : foundIndex-1 might be +infinite
        // solve that
        return findRange(x, beginIndex, foundIndex-1, foundIndex);
    }
    else if (x > ranges[foundIndex].max)
    {
        return findRange(x, foundIndex+1, endIndex, foundIndex);
    }
    else
    {
        return true;
    }
}

void RangeGroup::addRange(Range newRange)
{
    // std::size_t lastRangeIndexMin = findLastRangeBeforeX(newRange.min);
    // std::size_t lastRangeIndexMax = findLastRangeBeforeX(newRange.max);
    // if (lastRangeIndexMin == ranges.back() && lastRangeIndexMax == ranges.back())
    // {
        
    // }

    // @TODO : optimized range suppression using dichotomy 

    std::int32_t startIndex = 0; // included

    while (startIndex < ranges.size() && newRange.min-1 > ranges[startIndex].max)
    {
        startIndex++;
    }

    std::int32_t endIndex = startIndex; // excluded

    // If indices, prevent merge:
    // while (endIndex < ranges.size() && newRange.max+1 > ranges[endIndex].min)
    while (endIndex < ranges.size() && newRange.max+1 >= ranges[endIndex].min)
    {
        endIndex++;
    }

    if (startIndex == endIndex)
    {
        ranges.insert(ranges.begin() + startIndex, newRange);
        return;
    }

    newRange.min = std::min(newRange.min, ranges[startIndex].min);
    newRange.max = std::max(newRange.max, ranges[endIndex-1].max);

    // replace an element we're going to remove, so that we don't insert later
    ranges[startIndex] = newRange;
    // do not remove the replaced element
    ranges.erase(ranges.begin() + startIndex + 1, ranges.begin() + endIndex);
}
