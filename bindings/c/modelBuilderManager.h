#pragma once

#include "fwd.h"
#include "utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    AModel* (*loadModel)(const ModelLoader* loader);
} API_EXPORT CModelBuilder;

extern "C"
{
    API_EXPORT void modelBuilderManager_registerModelBuilder(CModelBuilder* builder);
    API_EXPORT void modelBuilderManager_loadModel(const char* folderPath, AModel** outModel, CResult* outResult);
}

#ifdef __cplusplus
}  // End extern "C"
#endif