#include "gtest/gtest.h"
#include "midiTokenizer.hpp"
#include "gen.h"
#include "modelBuilderManager.hpp"
#include "logitProcessing.hpp"
#include "range.hpp"
#include "searchArgs.h"

TEST(MidiTokTests, GptTest)
{
    std::unique_ptr<MidiTokenizer> Tokenizer = std::make_unique<MidiTokenizer>(WORKSPACE_PATH "/tokenizer.json");

	const char* Path = WORKSPACE_PATH "Models/TSD/1.2.10";

	EnvHandle Env = createEnv(false);

	ModelLoadingParamsWrapper Params;
	CResult r = createModelLoadingParamsWrapperFromFolder(Path, &Params);
	EXPECT_TRUE(ResultIsSuccess(&r));

	//CStr ModelType = modelLoadingParams_getModelType(&params);
	CppStr ModelType = Params.getModelType();
	OnnxModelBuilder* Builder = getModelBuilderManager().findBuilder<OnnxModelBuilder>(ModelType.Str());
	Builder->env = Env;

	AModel* Model = Builder->loadModelFromWrapper(Params);
	IPipeline* Pipeline = Model->createPipeline();
	IAutoRegressivePipeline* ARPipeline = (IAutoRegressivePipeline*)Pipeline; // @TODO : dynamic cast



    std::vector<int32_t> EncodedTokens;
    EncodedTokens.push_back(0);

    AutoRegressiveBatchHandle Batch2 = ARPipeline->addBatch();
    ARPipeline->batchSet(Batch2, EncodedTokens.data(), int32_t(EncodedTokens.size()), 0);

    RangeGroup SearchedRangeGroup;
    SearchedRangeGroup.addRange({0, Tokenizer->getNbEncodedTokens()-1});

    auto Search = [](const struct SearchArgs& args, void* searchStrategyData)
    {
        RangeGroup* RangeGroupLocal = (RangeGroup*) searchStrategyData;
        rangeGroupUpdateCache(RangeGroupLocal);

        int nbTopTokenSize = 40;
        size_t CurrentRangeGroupSize = rangeGroupSize(RangeGroupLocal);
        EXPECT_TRUE(CurrentRangeGroupSize >= nbTopTokenSize);

        std::vector<int32_t> LogitIndices;
        LogitIndices.resize(CurrentRangeGroupSize);
        int32_t* LogitIndicesData = LogitIndices.data();
        rangeGroupWrite(RangeGroupLocal, LogitIndicesData);

        sortLogits(args.logitsTensor, LogitIndicesData, LogitIndicesData + CurrentRangeGroupSize, nbTopTokenSize);

        stableSoftmax(args.logitsTensor, LogitIndicesData, LogitIndicesData + nbTopTokenSize);

        int outToken = topPSampling(args.logitsTensor, LogitIndicesData, LogitIndicesData + nbTopTokenSize, 0.5);
        args.outNextTokens[0] = outToken;
    };

    ARPipeline->setSearchStrategyData(&SearchedRangeGroup);
    ARPipeline->setSearchStrategy(Search);

    ARPipeline->createHistory(*Tokenizer);




    // std::vector<int32_t> StartTokens;
    // ARPipeline->batchSet(Batch2, StartTokens.data(), StartTokens.size(), 0);
    // ARPipeline->setMaxInputLength(1024);

    int32_t NbTokensSinceLastRefresh = 0;
    int32_t LineNbMaxToken = 512;
    int32_t NbTokensToGenerate = 300;

    while (EncodedTokens.size() < NbTokensToGenerate)
	{
		// if (forceReupdate)
		// {
		// 	if (Pipeline != nullptr)
		// 	{
		// 		Pipeline->reset();
		// 	}

		// 	{
		// 		TArray<int32> Context;
		// 		int32 start = FMath::Max(0, EncodedTokens.Num() - LineNbMaxToken);
		// 		for (int32 i = start; i < EncodedTokens.Num(); i++)
		// 		{
		// 			Context.Add(EncodedTokens[i]);
		// 		}

		// 		if (Pipeline != nullptr)
		// 		{
		// 			Pipeline->batchSet(Batch2, Context.GetData(), Context.Num(), start);
		// 		}
		// 		else
		// 		{
		// 			batch_set(batch, Context.GetData(), Context.Num(), start);
		// 		}
		// 	}
		// }

		// if (NbTokensSinceLastRefresh >= LineNbMaxToken)
		// {
		// 	if (Pipeline != nullptr)
		// 	{
		// 		Pipeline->reset();
		// 	}

		// 	TArray<int32> Context;
		// 	int32 start = FMath::Max(0, EncodedTokens.Num() - LineNbMaxToken / 2);
		// 	for (int32 i = start; i < EncodedTokens.Num(); i++)
		// 	{
		// 		Context.Add(EncodedTokens[i]);
		// 	}

		// 	if (Pipeline != nullptr)
		// 	{
		// 		Pipeline->batchSet(Batch2, Context.GetData(), Context.Num(), start);
		// 	}

		// 	NbTokensSinceLastRefresh = 0;
		// }

		if (Pipeline != nullptr)
		{
			CppResult Result;
			Pipeline->preGenerate(Result);
            EXPECT_TRUE(Result.IsSuccess());

			Pipeline->generate(Result);
            EXPECT_TRUE(Result.IsSuccess());

			Pipeline->postGenerate(Result);
            EXPECT_TRUE(Result.IsSuccess());
		}

		int32_t newToken;
		if (Pipeline != nullptr)
		{
			newToken = ARPipeline->batchGetLastGeneratedToken(Batch2);
		}

		EncodedTokens.push_back(newToken);
		NbTokensSinceLastRefresh++;
	}


    for (int32_t token : EncodedTokens)
    {
        const int32_t* begin, *end; 
        Tokenizer->decodeTokenFast(token, begin, end);
        std::cout << token << "(";

        while (begin != end)
        {
            std::string tokenStr = Tokenizer->decodedTokenToString(*begin);
            std::cout << tokenStr << ", ";
            ++begin;
        }
        std::cout << "), ";
    }
}

TEST(MidiTokTests, LlamaTest)
{

}
