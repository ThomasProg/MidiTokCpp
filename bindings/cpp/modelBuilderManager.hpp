#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>

#include "modelBuilderManager.h"

#include "json.hpp"
#include "fwd.h"

ASSERT_CPP_COMPILATION

using ModelLoadingParams = nlohmann::json;

class API_EXPORT ModelBuilder
{
public:
    virtual class AModel* loadModel(const ModelLoadingParams& jsonData) const = 0;
};

// Singleton Accessor
class API_EXPORT ModelBuilderManager
{
public:
    void registerModelBuilder(const char* modelType, ModelBuilder*&& modelBuilder);
    const ModelBuilder* loadBuilder(const ModelLoadingParams& jsonData) const;
    AModel* loadModel(const ModelLoadingParams& jsonData) const;
    AModel* loadModel(const char* folderPath) const;
};


API_EXPORT ModelBuilderManager& getModelBuilderManager();