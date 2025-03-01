#include "abstractPipeline.hpp"
#include <sstream>
#include "utilities.hpp"

CResult AOnnxModel::loadOnnxPipeline(const Ort::Env& env, const std::string& modelPath)
{
    // Create session options and enable optimization
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try 
    {
        session = std::make_unique<Ort::Session>(env, widen(modelPath).c_str(), session_options);

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