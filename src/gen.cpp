#include "gen.h"

#include "musicGenerator.hpp"
#include "redirector.hpp"
#include "midiTokenizer.hpp"
#include "midiConverter.hpp"

extern "C" 
{
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

void generator_loadOnnxModel(MusicGeneratorHandle generator, EnvHandle env, const char* path)
{
    generator->loadOnnxModel(*env, path);
}


void generator_generateNextToken(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    generator->generateNextToken(*runInstance);
}

void generator_preGenerate(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    generator->preGenerate(*runInstance);   
}

void generator_generate(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    generator->generate(*runInstance);
}

void generator_postGenerate(MusicGeneratorHandle generator, RunInstanceHandle runInstance)
{
    assert(generator != nullptr && runInstance != nullptr);
    generator->postGenerate(*runInstance);
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

void tokenizer_decodeIDs(MidiTokenizerHandle tokenizer, const std::int32_t* inputIDs, std::int32_t size, std::int32_t** outputIDs, std::int32_t* outSize)
{
    if (size == 0)
    {
        return;
    }

    std::vector<std::int32_t> inTokensVec;
    inTokensVec.reserve(size);
    for (std::int32_t i = 0; i < size; i++)
        inTokensVec.push_back(inputIDs[i]);

    std::vector<std::int32_t> outTokensVec;
    tokenizer->decodeIDs(inTokensVec, outTokensVec);

    *outSize = std::int32_t(outTokensVec.size());
    *outputIDs = new std::int32_t[*outSize]();

    for (std::int32_t i = 0; i < outTokensVec.size(); i++)
    {
        (*outputIDs)[i] = outTokensVec[i];
    }
}

void tokenizer_decodeIDs_free(std::int32_t* outputIDs)
{
    delete[] outputIDs;
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
    // Ort::TensorTypeAndShapeInfo tensorInfo = runInstance->pastTensors[index].GetTensorTypeAndShapeInfo();
    // std::vector<std::int64_t> shape = tensorInfo.GetShape();

    // *nbDims = shape.size();


    return runInstance->pastTensors[index].GetTensorData<float>();
}

const float* runInstance_getPresentTensor(RunInstanceHandle runInstance, std::int32_t index)
{
    return runInstance->presentTensors[index].GetTensorData<float>();
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
    return tokenizer->isPitch(token);
}

std::int32_t getPosition(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->getPositionValue(token);
}
std::int32_t getPitch(MidiTokenizerHandle tokenizer, std::int32_t token)
{
    return tokenizer->getPitchValue(token);
}



MidiConverterHandle createMidiConverter()
{
    return new MIDIConverter();
}
void destroyMidiConverter(MidiConverterHandle converter)
{
    delete converter;
}

void converterSetOnNote(MidiConverterHandle converter, void (*onNote)(void* data, const Note&))
{
    converter->onNote = onNote;
}
bool converterProcessToken(MidiConverterHandle converter, const int32_t* tokens, int32_t nbTokens, std::int32_t* index, void* data)
{
    assert(index != nullptr);
    return converter->processToken(tokens, nbTokens, *index, data);
}

void converterSetTokenizer(MidiConverterHandle converter, MidiTokenizerHandle tokenizer)
{
    converter->tokenizerHandle = tokenizer;
}





}
