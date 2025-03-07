#pragma once

#include "json.hpp"
#include "fwd.h"
#include "utilities.hpp"

struct ModelLoadingParams
{
    nlohmann::json json;
    CppStr modelPath;

    std::string getModelType() const
    {
        return json["model_type"];
    }
};