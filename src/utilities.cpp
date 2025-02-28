#include "utilities.h"
#include <cstring>

void InitCStr(CStr* str)
{
    str->str = nullptr;
    str->length = 0;
}

void DestroyCStr(CStr* str)
{
    if (str->str != nullptr)
    {
        delete[] str->str;
    }
}

void CStrReset(CStr* str)
{
    if (str->str != nullptr)
    {
        delete[] str->str;
        str->str = nullptr;
        str->length = 0;
    }
}

CStr MakeCStr(const char* inStr)
{
    CStr str;
    size_t len = strlen(inStr);
    str.str = new char[len+1]; 
    strcpy(str.str, inStr);
    str.str[len] = '\0';
    str.length = len;
    return str;
}

