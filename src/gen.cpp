#include "gen.h"

#include "musicGenerator.hpp"
#include "redirector.hpp"
#include "midiTokenizer.hpp"
#include "midiConverter.hpp"
#include "range.hpp"

extern "C" 
{
std::int64_t computeMultiDimIndex(std::int64_t* shape, std::int64_t* indices)
{
    return computeMultiDimIdx(shape, indices);
}

EnvHandle createEnv(bool useLogging)
{
    if (useLogging)
        return new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "MusicGenerator");
    else
        return new Ort::Env();
}

void destroyEnv(EnvHandle env)
{
    delete env;
}

MidiTokenizerHandle createMidiTokenizer(const char* tokenizerPath)
{
    return new MidiTokenizer(tokenizerPath);
}
void destroyMidiTokenizer(MidiTokenizerHandle tokenizer)
{
    delete tokenizer;
}

MusicGeneratorHandle createMusicGenerator()
{
    return new MusicGenerator();
}
void destroyMusicGenerator(MusicGeneratorHandle musicGen)
{
    delete musicGen;
}

CResult generator_loadOnnxModel(MusicGeneratorHandle generator, EnvHandle env, const char* path)
{
    return generator->loadOnnxModel(*env, path);
}


void generator_generateNextToken(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    generator->generateNextToken(*runInstance);
}

CResult generator_preGenerate(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    CppResult res;
    generator->preGenerate(*runInstance, res);
    return res.Release();   
}

CResult generator_generate(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    CppResult res;
    generator->generate(*runInstance, res);
    return res.Release();   
}

CResult generator_postGenerate(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    CppResult res;
    generator->postGenerate(*runInstance, res);
    return res.Release();
}


void generator_setConfig(MusicGeneratorHandle generator, int64_t num_attention_heads, int64_t hidden_size, int64_t num_layer)
{
    ModelInfo& info = generator->modelInfo;
    info.num_layer = num_layer;
    info.num_attention_heads = num_attention_heads;
    info.hidden_size = hidden_size;
}

RunInstance* generator_createRunInstance(MusicGeneratorHandle generator)
{
    return generator->createRunInstance();
}

void generator_getNextTokens_greedyFiltered(const SearchArgs& args, bool (*filter)(std::int32_t token, void* data), void* data)
{
    MusicGenerator::getNextTokens_greedyFiltered(args, filter, data);
}
void generator_getNextTokens_greedyPreFiltered(const SearchArgs& args, std::int32_t* availableTokens, std::int32_t nbAvailableToken)
{
    MusicGenerator::getNextTokens_greedyPreFiltered(args, availableTokens, nbAvailableToken);
}
void generator_getNextTokens_greedy(const SearchArgs& args)
{
    MusicGenerator::getNextTokens_greedy(args);
}

void generator_setNbAttentionHeads(MusicGeneratorHandle generator, std::int64_t nbAttentionHeads)
{
    generator->modelInfo.num_attention_heads = nbAttentionHeads;
}
void generator_setHiddenSize(MusicGeneratorHandle generator, std::int64_t hiddenSize)
{
    generator->modelInfo.hidden_size = hiddenSize;
}
void generator_setNbLayers(MusicGeneratorHandle generator, std::int64_t nbLayers)
{
    generator->modelInfo.num_layer = nbLayers;
}
void generator_setVocabSize(MusicGeneratorHandle generator, std::int64_t vocabSize)
{
    generator->modelInfo.vocab_size = vocabSize;
}
void generator_setNbMaxPositions(MusicGeneratorHandle generator, std::int64_t nbMaxPositions)
{
    generator->modelInfo.nbMaxPositions = nbMaxPositions;
}

std::int64_t generator_getNbAttentionHeads(MusicGeneratorHandle generator)
{
    return generator->modelInfo.num_attention_heads;
}
std::int64_t generator_getHiddenSize(MusicGeneratorHandle generator)
{
    return generator->modelInfo.hidden_size;
}
std::int64_t generator_getNbLayers(MusicGeneratorHandle generator)
{
    return generator->modelInfo.num_layer;
}
std::int64_t generator_getVocabSize(MusicGeneratorHandle generator)
{
    return generator->modelInfo.vocab_size;
}
std::int64_t generator_getNbMaxPositions(MusicGeneratorHandle generator)
{
    return generator->modelInfo.nbMaxPositions;
}

RedirectorHandle createRedirector()
{
    return new Redirector();
}
void destroyRedirector(RedirectorHandle redirector)
{
    delete redirector;
}

TokenSequenceHandle createTokenSequence()
{
    return new TokenSequence();
}
void destroyTokenSequence(TokenSequenceHandle tokenSequence)
{
    delete tokenSequence;
}

void redirector_bindPitch(RedirectorHandle redirector, const MidiTokenizerHandle tokenizer, const char* prefix, void* data, void(*callback)(void*, std::uint8_t pitch))
{
    redirector->bindPitch(*tokenizer, prefix, callback, data);
}

bool redirector_call(RedirectorHandle redirector, int32_t token)
{
    return redirector->tryCall(token);
}

CResult tokenizer_decodeIDs(MidiTokenizerHandle tokenizer, const std::int32_t* inputIDs, std::int32_t size, std::int32_t** outputIDs, std::int32_t* outSize)
{
    if (size == 0)
    {
        return CResult();
    }

    std::vector<std::int32_t> inTokensVec;
    inTokensVec.reserve(size);
    for (std::int32_t i = 0; i < size; i++)
        inTokensVec.push_back(inputIDs[i]);

    std::vector<std::int32_t> outTokensVec;
    try
    {
        tokenizer->decodeIDs(inTokensVec, outTokensVec);
    }
    catch(const std::exception&)
    {
        return CResult({MakeCStr("error in decodeIDs()")});
    }

    *outSize = std::int32_t(outTokensVec.size());
    *outputIDs = new std::int32_t[*outSize]();

    for (std::int32_t i = 0; i < outTokensVec.size(); i++)
    {
        (*outputIDs)[i] = outTokensVec[i];
    }

    return CResult();
}

void tokenizer_decodeIDs_free(std::int32_t* outputIDs)
{
    delete[] outputIDs;
}

void tokenizer_decodeToken(MidiTokenizerHandle tokenizer, std::int32_t encodedToken, std::int32_t** outDecodedTokens, std::int32_t* outNbDecodedTokens)
{
    std::vector<int32_t> decodedTokens(10);
    try 
    {
        tokenizer->decodeToken(encodedToken, decodedTokens);
    }
    catch (const std::exception&)
    {
        *outNbDecodedTokens = 0;
        *outDecodedTokens = nullptr;
        return;
    }

    *outNbDecodedTokens = static_cast<std::int32_t>(decodedTokens.size());
    *outDecodedTokens = new std::int32_t[*outNbDecodedTokens]();
    // *outDecodedTokens = decodedTokens.data();

    for (std::int32_t i = 0; i < decodedTokens.size(); i++)
    {
        (*outDecodedTokens)[i] = decodedTokens[i];
    }
}

void tokenizer_decodeToken_free(std::int32_t* outputIDs)
{
    delete[] outputIDs;
}

void tokenizer_addTokensStartingByPosition(MidiTokenizerHandle tokenizer, RangeGroupHandle outRangeGroup)
{
    tokenizer->addTokensStartingByPosition(*outRangeGroup);
}
void tokenizer_addTokensStartingByBarNone(MidiTokenizerHandle tokenizer, RangeGroupHandle outRangeGroup)
{
    tokenizer->addTokensStartingByBarNone(*outRangeGroup);
}
void tokenizer_addTokensStartingByPitch(MidiTokenizerHandle tokenizer, RangeGroupHandle outRangeGroup)
{
    tokenizer->addTokensStartingByPitch(*outRangeGroup);
}
void tokenizer_addTokensStartingByVelocity(MidiTokenizerHandle tokenizer, RangeGroupHandle outRangeGroup)
{
    tokenizer->addTokensStartingByVelocity(*outRangeGroup);
}
void tokenizer_addTokensStartingByDuration(MidiTokenizerHandle tokenizer, RangeGroupHandle outRangeGroup)
{
    tokenizer->addTokensStartingByDuration(*outRangeGroup);
}

void tokenizer_addTokensStartingByTimeShift(MidiTokenizerHandle tokenizer, RangeGroupHandle outRangeGroup)
{
    return tokenizer->addTokensStartingByTimeShift(*outRangeGroup);
}

const char* tokenizer_decodedTokenToString(MidiTokenizerHandle tokenizer, std::int32_t decodedToken)
{
    const std::string& str = tokenizer->decodedTokenToString(decodedToken);
    return str.c_str();
}

std::int32_t tokenizer_getNbEncodedTokens(MidiTokenizerHandle tokenizer)
{
    return tokenizer->getNbEncodedTokens();
}

std::int32_t tokenizer_getNbDecodedTokens(MidiTokenizerHandle tokenizer)
{
    return tokenizer->getNbDecodedTokens();
}

bool tokenizer_useVelocities(MidiTokenizerHandle tokenizer)
{
    return tokenizer->useVelocities();
}
bool tokenizer_useDuration(MidiTokenizerHandle tokenizer)
{
    return tokenizer->useDuration();
}
bool tokenizer_useTimeSignatures(MidiTokenizerHandle tokenizer)
{
    return tokenizer->useTimeSignatures();
}
const char* tokenizer_getTokenizationType(MidiTokenizerHandle tokenizer)
{
    return tokenizer->getTokenizationType();
}

BatchHandle createBatch()
{
    return new Batch();
}
void destroyBatch(BatchHandle batch)
{
    delete batch; 
}

std::int32_t batch_getLastGeneratedToken(BatchHandle batch)
{
    return batch->lastGeneratedToken;
}

void batch_push(BatchHandle batch, DataType inInputId)
{
    batch->push(inInputId);
}
void batch_pop(BatchHandle batch)
{
    batch->pop();
}

void batch_set(BatchHandle batch, DataType* inputTokens, std::int32_t nbTokens, std::int32_t fromPos)
{
    std::vector<DataType> inTokens(nbTokens);

    for (std::int32_t i = 0; i < nbTokens; i++)
    {
        inTokens[i] = inputTokens[i];
    }

    batch->set(std::move(inTokens), fromPos);
}

std::int32_t batch_size(BatchHandle batch)
{
    return std::int32_t(batch->size());
}

void batch_getEncodedTokens(BatchHandle batch, DataType** outEncodedTokens, std::int32_t* outNbTokens)
{
    *outEncodedTokens =  batch->inputIds.data();
    *outNbTokens = std::int32_t(batch->inputIds.size());
}


RunInstanceHandle createRunInstance()
{
    return new RunInstance();
}
void destroyRunInstance(RunInstanceHandle runInstance)
{
    delete runInstance; 
}

void runInstance_addBatch(RunInstanceHandle runInstance, BatchHandle batch)
{
    runInstance->batches.push_back(batch);
}
void runInstance_removeBatch(RunInstanceHandle runInstance, BatchHandle batch)
{
    auto& batches = runInstance->batches; 
    batches.erase(std::find(batches.begin(), batches.end(), batch));
}

std::int32_t runInstance_nbBatches(RunInstanceHandle runInstance)
{
    return std::int32_t(runInstance->batches.size());
}

void runInstance_setMaxInputLength(RunInstanceHandle runInstance, std::int32_t newMaxInputLength)
{
    runInstance->maxInputLength = newMaxInputLength;
}

void runInstance_reset(RunInstanceHandle runInstance)
{
    runInstance->reset();
}

const float* runInstance_getPastTensor(RunInstanceHandle runInstance, std::int32_t index)
{
    return runInstance->pastTensors[index].GetTensorData<float>();
}

const float* runInstance_getPresentTensor(RunInstanceHandle runInstance, std::int32_t index)
{
    return runInstance->presentTensors[index].GetTensorData<float>();
}

void runInstance_getPresentTensorShape(RunInstanceHandle runInstance, MusicGeneratorHandle generator, std::int64_t* outShape)
{
    std::array<std::int64_t, 5>& outPresent = * (std::array<std::int64_t, 5>*) outShape;
    runInstance->getPresentTensorShape(generator->modelInfo, outPresent);
}
void runInstance_getPastTensorShape(RunInstanceHandle runInstance, MusicGeneratorHandle generator, std::int64_t* outShape)
{
    std::array<std::int64_t, 5>& outPastShape = * (std::array<std::int64_t, 5>*) outShape;
    runInstance->getPresentTensorShape(generator->modelInfo, outPastShape);
}

void runInstance_setSearchStrategyData(RunInstanceHandle runInstance, void* searchStrategyData)
{
    runInstance->searchStrategyData = searchStrategyData;
}
void runInstance_setSearchStrategy(RunInstanceHandle runInstance, TSearchStrategy searchStrategy)
{
    runInstance->searchStrategy = searchStrategy;
}

bool isBarNone(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->isBarNone(token);
}
bool isPosition(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->isPosition(token);
}
bool isPitch(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    // return tokenizer->isPitch(token);
    return tokenizer->isPitchFast(token);
}
bool isDuration(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->isDuration(token);
}
bool isVelocity(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->isVelocity(token);
}

std::int32_t getPosition(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->getPositionValue(token);
}
std::int32_t getPitch(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    // return tokenizer->getPitchValue(token);
    return std::int32_t(tokenizer->getPitchValueFast(token));
}






RangeGroupHandle createRangeGroup()
{
    return new RangeGroup();
}

RangeGroupHandle cloneRangeGroup(RangeGroupHandle rangeGroup)
{
    return new RangeGroup(*rangeGroup);
}

void destroyRangeGroup(RangeGroupHandle rangeGroup)
{
    delete rangeGroup;
}

void rangeGroupAdd(RangeGroupHandle rangeGroup, int32_t nb)
{
    // @TODO : opti for single number/
    rangeGroup->addRange(Range{nb, nb});
}
void rangeGroupAddRange(RangeGroupHandle rangeGroup, int32_t min, int32_t max)
{
    rangeGroup->addRange(Range{min, max});
}
void rangeGroupGetRanges(RangeGroupHandle rangeGroup, Range const** outRanges, size_t* outNbElements)
{
    const std::vector<Range>& ranges = rangeGroup->getRanges();
    *outRanges = ranges.data();
    *outNbElements = ranges.size();
}





}
