#pragma once

#include <cstdio>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "utilities.hpp"

class APipeline;

// A model can be loaded before the pipeline.
// That way, depending on metadata, we can decide automatically what pipeline to use.
class AModel
{
public:
    virtual ~AModel() = default;

    virtual APipeline* CreatePipeline() = 0;
};

class AOnnxModel : public AModel
{
public:
    std::unique_ptr<Ort::Session> session;
    CResult loadOnnxPipeline(const Ort::Env& env, const std::string& modelPath);

    virtual CResult onPostOnnxLoad() { return CResult(); }
};

// Inference Pipeline
class APipeline
{
public:
    virtual void preGenerate(CppResult& outResult) = 0;
    virtual void generate(CppResult& outResult) = 0;
    virtual void postGenerate(CppResult& outResult) = 0;

    virtual AModel* getModel() const = 0;
};
