#include "abstractPipeline.hpp"
#include <sstream>
#include "utilities.hpp"
#include <onnxruntime_cxx_api.h>

AOnnxModel::~AOnnxModel()
{
    if (session != nullptr)
    {
        delete session;
    }
}

CResult AOnnxModel::loadOnnxModel(const Ort::Env& env, const char* modelPath)
{
    // Create session options and enable optimization
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try 
    {
        session = new Ort::Session(env, widen(modelPath).c_str(), session_options);

        return onPostOnnxLoad();
    }
    catch(const Ort::Exception& e)
    {
        std::ostringstream oss;
        oss << e.what() << "\nError Code: " << std::to_string(e.GetOrtErrorCode());
        CResult result;
        result.message = MakeCStr(oss.str());
        return result;
    }
}

void AOnnxModel::generate(const Ort::IoBinding& ioBindings, CppResult& outResult)
{
    try 
    {
        session->Run(Ort::RunOptions{nullptr}, ioBindings);
    }
    catch(const Ort::Exception& e)
    {
        std::string errorMsg;
        errorMsg += "Error occurred: " + std::string(e.what());
        errorMsg += "Error code: " + std::to_string(e.GetOrtErrorCode());
        outResult = CppResult(errorMsg.c_str());
    }
}