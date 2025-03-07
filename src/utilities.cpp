#include "utilities.h"
#include "utilities.hpp"
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

CStr CreateCStr()
{
    CStr str;
    InitCStr(&str);
    return str;
}

#include <onnxruntime_cxx_api.h>

template API_EXPORT class UniquePtr<Ort::Env>;

template API_EXPORT class UniquePtr<Ort::Session>;
template UniquePtr<Ort::Session> MakeUnique<Ort::Session, const Ort::Env&, const ORTCHAR_T*&&, const Ort::SessionOptions&>(const Ort::Env& env, const ORTCHAR_T*&& model_path, const Ort::SessionOptions& options);
template UniquePtr<Ort::Session> MakeUnique<Ort::Session, const Ort::Env&, const ORTCHAR_T*&&, const Ort::SessionOptions&, OrtPrepackedWeightsContainer*&&>(const Ort::Env& env, const ORTCHAR_T*&& model_path, const Ort::SessionOptions& options, OrtPrepackedWeightsContainer*&& prepacked_weights_container);
template UniquePtr<Ort::Session> MakeUnique<Ort::Session, const Ort::Env&, const void*&&, size_t&&, const Ort::SessionOptions&>(const Ort::Env& env, const void*&& model_data, size_t&& model_data_length, const Ort::SessionOptions& options);
template UniquePtr<Ort::Session> MakeUnique<Ort::Session, const Ort::Env&, const void*&&, size_t&&, const Ort::SessionOptions&, OrtPrepackedWeightsContainer*&&>(const Ort::Env& env, const void*&& model_data, size_t&& model_data_length, const Ort::SessionOptions& options, OrtPrepackedWeightsContainer*&& prepacked_weights_container);
