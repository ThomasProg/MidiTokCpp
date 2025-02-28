#pragma once

#include "utilities.h"
#include <utility>

// .h only header
// wraps utilities.h
// can be included from outside libraries

class CppStr
{
protected:
    CStr str;

public:
    void StrReset()
    {
        ::CStrReset(&str);
    }

    const char* Str()
    {
        return str.str;
    }

    size_t Length()
    {
        return str.length;
    }

    CppStr()
    {
        ::InitCStr(&str);
    }

    CppStr(const CStr& inStr)
    {
        str = inStr;
    }

    CppStr(CStr&& inStr)
    {
        str = std::move(inStr);
    }

    ~CppStr()
    {
        ::DestroyCStr(&str);
    }
};

#include <string>
inline CStr MakeCStr(const std::string inStr)
{
    return MakeCStr(inStr.c_str());
}
#include <sstream>
inline CStr MakeCStr(const std::ostringstream inStrStream)
{
    return MakeCStr(inStrStream.str());
}

inline std::wstring widen( const std::string& str )
{
    std::wostringstream wstm ;
    const std::ctype<wchar_t>& ctfacet = std::use_facet<std::ctype<wchar_t>>(wstm.getloc()) ;
    for( size_t i=0 ; i<str.size() ; ++i ) 
              wstm << ctfacet.widen( str[i] ) ;
    return wstm.str() ;
}


class CppResult
{
public:
    CResult result;

    CppResult()
    {
        result.message.str = nullptr;
        result.message.length = 0;
    }

    CppResult(const CResult& inRes) = delete;

    CppResult(CResult&& inRes)
    {
        result = std::move(inRes);
        inRes.message.str = nullptr;
        inRes.message.length = 0;
    }

    CppResult(const char* errorMsg) : result({MakeCStr(errorMsg)})
    {

    }

    CppResult& operator=(CppResult&& inRes)
    {
        result = std::move(inRes.result);
        inRes.result.message.str = nullptr;
        inRes.result.message.length = 0;
        return *this;
    }

    CppResult& operator=(const CppResult& inRes) = delete;


    ~CppResult()
    {
        DestroyCResult(&result);
    }

    CResult Release()
    {
        CResult outRes = result;
        result.message.str = nullptr;
        result.message.length = 0;
        return outRes;
    }

    bool IsSuccess() const
    {
        return ResultIsSuccess(&result);
    }

    const char* GetError() const
    {
        return result.message.str;
    }
};