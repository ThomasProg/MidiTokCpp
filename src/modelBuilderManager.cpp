#include "modelBuilderManager.hpp"
#include "modelBuilderManager.h"

#include <unordered_map>
#include <string>
#include <fstream>

#include "modelLoadingParams.hpp"

std::string folderPathToModelConfigPath(const std::string& folderPath)
{
    return folderPath + "/config.json";
}

std::string folderPathToModelModelPath(const std::string& folderPath)
{
    return folderPath + "/model.json";
}

// .hpp

ModelLoadingParamsWrapper::ModelLoadingParamsWrapper() : internal(*new ModelLoadingParams())
{
    
}
ModelLoadingParamsWrapper::~ModelLoadingParamsWrapper()
{
    delete &internal;
}

CppStr ModelLoadingParamsWrapper::getModelType() const
{
    return MakeCStr(internal.json["model_type"]);
}

AModel* ModelBuilder::loadModelFromWrapper(const ModelLoadingParamsWrapper& loadingData) const
{
    return loadModel(loadingData.internal);
}

const TypeInfo& ModelBuilder::getStaticTypeInfo()
{
    static TypeInfo typeInfo{&Object::getStaticTypeInfo()};
    return typeInfo; 
}

const TypeInfo& ModelBuilder::getTypeInfo()
{
    return getStaticTypeInfo(); 
}

const TypeInfo& OnnxModelBuilder::getStaticTypeInfo()
{
    static TypeInfo typeInfo{&ModelBuilder::getStaticTypeInfo()};
    return typeInfo; 
}

const TypeInfo& OnnxModelBuilder::getTypeInfo()
{
    return getStaticTypeInfo(); 
}

class ModelBuilderManagerInternal
{
    std::unordered_map<std::string, std::unique_ptr<ModelBuilder>> modelTypeToBuilder;

public:
    void registerModelBuilder(const char* modelType, ModelBuilder*&& modelBuilder);
    const ModelBuilder* loadBuilder(const ModelLoadingParams& jsonData) const;
    AModel* loadModel(const ModelLoadingParams& jsonData) const;
    AModel* loadModel(const char* folderPath) const;

    ModelBuilder* findBuilder(const std::string& modelType) const;
};

ModelBuilderManagerInternal modelBuilderManagerSingleton;

void ModelBuilderManagerInternal::registerModelBuilder(const char* modelType, ModelBuilder*&& modelBuilder)
{
    // default settings can be overwritten
    modelTypeToBuilder[modelType] = std::unique_ptr<ModelBuilder>(std::move(modelBuilder)); 
}

const ModelBuilder* ModelBuilderManagerInternal::loadBuilder(const ModelLoadingParams& loadingData) const
{
    std::string modelType = loadingData.json["model_type"];
    return findBuilder(modelType);
}

AModel* ModelBuilderManagerInternal::loadModel(const ModelLoadingParams& loadingData) const
{
    const ModelBuilder* builder = loadBuilder(loadingData);
    return builder->loadModel(loadingData);
}

AModel* ModelBuilderManagerInternal::loadModel(const char* folderPath) const
{
    ModelLoadingParams loadingData;
    CppResult r = createModelLoadingParamsFromFolder(folderPath, &loadingData);
    if (!r.IsSuccess())
    {
        return nullptr;
    }
    return loadModel(std::move(loadingData));
}

ModelBuilder* ModelBuilderManagerInternal::findBuilder(const std::string& modelType) const
{
    auto it = modelTypeToBuilder.find(modelType);

    if (it == modelTypeToBuilder.end())
    {
        return nullptr;
    }

    return it->second.get();
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

ModelBuilder* ModelBuilderManager::findBuilder(const char* modelType) const
{
    return modelBuilderManagerSingleton.findBuilder(modelType);
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

API_EXPORT void ModelBuilderManager_loadModel(const char* folderPath, AModelHandle* outModel, CResult* outResult)
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

CResult createModelLoadingParamsFromFolder(const char* folderPath, ModelLoadingParams* outParams)
{
    assert(outParams != nullptr);
    outParams->modelPath = MakeCStr(folderPath);

    // Open the file and parse it
    std::ifstream inputFile(folderPathToModelConfigPath(folderPath));
    if (inputFile.is_open()) {
        inputFile >> outParams->json;
        inputFile.close();
    } 
    else 
    {
        return CResult{MakeCStr("Config File doesn't exist!")};
    }

    return CResult{CreateCStr()};
}

CResult createModelLoadingParamsWrapperFromFolder(const char* folderPath, ModelLoadingParamsWrapper* outParams)
{
    return createModelLoadingParamsFromFolder(folderPath, &outParams->internal);
}

CStr modelLoadingParams_getModelType(ModelLoadingParams* params)
{
    return MakeCStr(params->getModelType().c_str());
}