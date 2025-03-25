#pragma once

#include "fwd.h"
#include "utilities.hpp"
#include <vector>

class IBatch
{
public:
    using DataType = int32_t;

    virtual void set(const DataType* inTokens, int32_t inNbTokens) = 0;
    virtual size_t size() const = 0;
};

class IIOHandler
{
private:
    void createInputTensor(Ort::Value* tensor);

public:
    virtual void bindInputs(CppResult& outResult);
    virtual void bindOutputs(CppResult& outResult);
    virtual void bind(CppResult& outResult);

    virtual void createInputIdsTensor();
    virtual void createPositionIdsTensor();
    virtual void createAttentionMaskTensor();
    virtual void createPresentTensors(int64_t presentLength);

    virtual void createLogitsTensor();
    virtual void createPastTensors(int64_t pastLength);

    virtual void updateInputIdsTensor() = 0;
    virtual void updatePositionIdsTensor() = 0;
    virtual void updateAttentionMaskTensor() = 0;

    // Bind Inputs
    virtual void bindInputIds() = 0;
    virtual void bindPositionIds() = 0;
    virtual void bindAttentionMask() = 0;
    virtual void bindPasts(CppResult& outResult) = 0;

    // Bind Outputs
    virtual void bindPresents(CppResult& outResult) = 0;
    virtual void bindLogits() = 0;



    virtual int32_t getNbBatches() const = 0;
    virtual int32_t getInputLength() const = 0; // Sequence Length
    virtual int32_t getVocabSize() const = 0;
    virtual int32_t getNbAttentionHeads() const = 0;
    virtual int32_t getHiddenSize() const = 0;
    virtual int32_t getNbLayers() const = 0;

    virtual Ort::Value* getInputIdsTensor() = 0;
    virtual Ort::Value* getPositionIdsTensor() { return nullptr; }
    virtual Ort::Value* getAttentionMaskTensor() { return nullptr; }
    virtual std::vector<Ort::Value>* getPresentTensors() { return nullptr; }
    virtual Ort::Value* getLogitsTensor() = 0;
    virtual std::vector<Ort::Value>* getPastTensors() { return nullptr; }
    virtual struct OrtAllocator* getAllocator() = 0;

    void createFirstTimeTensors(CppResult& outResult);

    SearchArgs createSearchArgs(Ort::Value& logitsTensor, std::vector<DataType>& outNextTokens);
    static void copyAndShiftPresentIntoNextPast(const float* presentData, float* pastData, int64_t presentShape[], int64_t pastShape[]);
};



