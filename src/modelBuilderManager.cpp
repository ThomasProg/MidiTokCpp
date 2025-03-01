#include "modelBuilderManager.hpp"
#include "modelBuilderManager.h"

#include <unordered_map>
#include <string>
#include <fstream>

std::string folderPathToModelConfigPath(const std::string& folderPath)
{
    return folderPath + "/config.json";
}

std::string folderPathToModelModelPath(const std::string& folderPath)
{
    return folderPath + "/model.json";
}

// .hpp
class ModelBuilderManagerInternal
{
    std::unordered_map<std::string, std::unique_ptr<ModelBuilder>> modelTypeToBuilder;

public:
    void registerModelBuilder(const char* modelType, ModelBuilder*&& modelBuilder);
    const ModelBuilder* loadBuilder(const ModelLoadingParams& jsonData) const;
    AModel* loadModel(const ModelLoadingParams& jsonData) const;
    AModel* loadModel(const char* folderPath) const;
};

ModelBuilderManagerInternal modelBuilderManagerSingleton;

void ModelBuilderManagerInternal::registerModelBuilder(const char* modelType, ModelBuilder*&& modelBuilder)
{
    // default settings can be overwritten
    modelTypeToBuilder[modelType] = std::unique_ptr<ModelBuilder>(std::move(modelBuilder)); 
}

const ModelBuilder* ModelBuilderManagerInternal::loadBuilder(const ModelLoadingParams& jsonData) const
{
    std::string modelType = jsonData["model_type"];
    auto it = modelTypeToBuilder.find(modelType);

    if (it == modelTypeToBuilder.end())
    {
        return nullptr;
    }

    return it->second.get();
}

AModel* ModelBuilderManagerInternal::loadModel(const ModelLoadingParams& jsonData) const
{
    const ModelBuilder* builder = loadBuilder(jsonData);
    return builder->loadModel(jsonData);
}

AModel* ModelBuilderManagerInternal::loadModel(const char* folderPath) const
{
    nlohmann::json jsonData;

    // Open the file and parse it
    std::ifstream inputFile(folderPathToModelConfigPath(folderPath));
    if (inputFile.is_open()) {
        inputFile >> jsonData;
        inputFile.close();
    } 
    else 
    {
        throw std::runtime_error("Config File doesn't exist!");
    }

    return loadModel(jsonData);
}



ModelBuilderManager modelBuilderManager;
ModelBuilderManager& getModelBuilderManager()
{
    return modelBuilderManager;
}

void ModelBuilderManager::registerModelBuilder(const char* modelType, ModelBuilder*&& modelBuilder)
{
    return modelBuilderManagerSingleton.registerModelBuilder(modelType, std::move(modelBuilder));
}
const ModelBuilder* ModelBuilderManager::loadBuilder(const ModelLoadingParams& jsonData) const
{
    return modelBuilderManagerSingleton.loadBuilder(jsonData);
}
AModel* ModelBuilderManager::loadModel(const ModelLoadingParams& jsonData) const
{
    return modelBuilderManagerSingleton.loadModel(jsonData);
}

AModel* ModelBuilderManager::loadModel(const char* folderPath) const
{
    return modelBuilderManagerSingleton.loadModel(folderPath);
}

// .h
API_EXPORT void modelBuilderManager_registerModelBuilder(const char* modelType, CModelBuilder* builder)
{
    class WrapperCModelBuilder : public ModelBuilder
    {
    private:
        CModelBuilder ModelBuilder;
        
        virtual class AModel* loadModel(const ModelLoadingParams& jsonData) const override
        {
            return nullptr;
        }

    public:
        WrapperCModelBuilder(const CModelBuilder& builder) : ModelBuilder(builder) {}
    };

    ModelBuilder* wrapped = new WrapperCModelBuilder(*builder);
    modelBuilderManagerSingleton.registerModelBuilder(modelType, std::move(wrapped));
}

API_EXPORT void ModelBuilderManager_loadModel(const char* folderPath, AModel** outModel, CResult* outResult)
{
    try
    {
        *outModel = modelBuilderManagerSingleton.loadModel(folderPath);
    }
    catch (const std::exception&)
    {
        *outResult = CResult{MakeCStr("Couldn't load model")};
    }
}


