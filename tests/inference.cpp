#include "inference.hpp"

#include "gtest/gtest.h"
#include "midiTokenizer.hpp"
#include "gen.h"
#include "modelBuilderManager.hpp"
#include "logitProcessing.hpp"
#include "range.hpp"
#include "searchArgs.h"
#include "generationHistory.hpp"

void Inf::load(const char* folderPath, bool printLogs)
{
    std::string folderPathStr = folderPath;

    Tokenizer = std::make_unique<MidiTokenizer>(folderPathStr + "/tokenizer.json");

	Env = createEnv(printLogs);

	ModelLoadingParamsWrapper Params;
	CResult r = createModelLoadingParamsWrapperFromFolder(folderPathStr.c_str(), &Params);
	EXPECT_TRUE(ResultIsSuccess(&r));

	//CStr ModelType = modelLoadingParams_getModelType(&params);
	CppStr ModelType = Params.getModelType();
	OnnxModelBuilder* Builder = getModelBuilderManager().findBuilder<OnnxModelBuilder>(ModelType.Str());
	Builder->env = Env;

	AModel* Model = Builder->loadModelFromWrapper(Params);
	IPipeline* Pipeline = Model->createPipeline();
	ARPipeline = (IAutoRegressivePipeline*)Pipeline; // @TODO : dynamic cast



    std::vector<int32_t> EncodedTokens;
    EncodedTokens.push_back(0);

    Batch2 = ARPipeline->addBatch();
    ARPipeline->batchSet(Batch2, EncodedTokens.data(), int32_t(EncodedTokens.size()), 0);

    SearchedRangeGroup.addRange({0, Tokenizer->getNbEncodedTokens()-1});

    auto Search = [](const struct SearchArgs& args, void* searchStrategyData)
    {
		Inf* inf = (Inf*) searchStrategyData;
        RangeGroup* RangeGroupLocal = &inf->SearchedRangeGroup;;
        rangeGroupUpdateCache(RangeGroupLocal);

		// {
		// 	SpecialPenaltyTransformArgs sArgs;
		// 	sArgs.pitchWindowSize = 20;
		// 	sArgs.pitchMaxAdditivePenalty = 0.05f;
		// 	specialPenaltyTransform(args.logitsTensor, RangeGroupLocal, inf->ARPipeline->getHistory(inf->Batch2), &sArgs);
		// }

		MidiTokenizerHandle Tok = inf->Tokenizer.get();

		// {
		// 	musicalScalePenaltyTransform(args.logitsTensor, RangeGroupLocal, Scales::Ionian::CMajor::get(), Scales::Ionian::CMajor::size(), 1.05f, Tok);
		// }
		// {
		// 	pitchRangePenaltyTransform(args.logitsTensor, RangeGroupLocal, 40, 80, 0.05f, Tok);
		// }

        int nbTopTokenSize = 40;
        size_t CurrentRangeGroupSize = rangeGroupSize(RangeGroupLocal);
        EXPECT_TRUE(CurrentRangeGroupSize >= nbTopTokenSize);

        std::vector<int32_t> LogitIndices;
        LogitIndices.resize(CurrentRangeGroupSize);
        int32_t* LogitIndicesData = LogitIndices.data();
        rangeGroupWrite(RangeGroupLocal, LogitIndicesData);

        sortLogits(args.logitsTensor, LogitIndicesData, LogitIndicesData + CurrentRangeGroupSize, nbTopTokenSize);

        stableSoftmax(args.logitsTensor, LogitIndicesData, LogitIndicesData + nbTopTokenSize);

        int outToken = topPSampling(args.logitsTensor, LogitIndicesData, LogitIndicesData + nbTopTokenSize, 0.2f);
        args.outNextTokens[0] = outToken;
    };

    ARPipeline->setSearchStrategyData(this);
    ARPipeline->setSearchStrategy(Search);

    ARPipeline->createHistory(*Tokenizer);
}

void Inf::runInference()
{
    int32_t NbTokensSinceLastRefresh = 0;

    while (NbTokensSinceLastRefresh < NbTokensToGenerate)
	{
        CppResult Result;
        ARPipeline->preGenerate(Result);
        EXPECT_TRUE(Result.IsSuccess());

        ARPipeline->generate(Result);
        EXPECT_TRUE(Result.IsSuccess());

        ARPipeline->postGenerate(Result);
        EXPECT_TRUE(Result.IsSuccess());

		int32_t newToken;
        newToken = ARPipeline->batchGetLastGeneratedToken(Batch2);

        if (false)
        {
            const int32_t* begin, *end; 
            Tokenizer->decodeTokenFast(newToken, begin, end);
            std::cout << NbTokensSinceLastRefresh << " : " << newToken << "(";

            while (begin != end)
            {
                std::string tokenStr = Tokenizer->decodedTokenToString(*begin);
                std::cout << tokenStr << ", ";
                ++begin;
            }
            std::cout << "), " << std::endl;
        }

		EncodedTokens.push_back(newToken);
		NbTokensSinceLastRefresh++;
	}
}

void TokenHistoryTest::TestTokenHistory(const TokenHistory& tokenHistory)
{
	size_t nbTotalTokens = 0;
	for (auto [key, value] : tokenHistory.tokensData)
	{
		nbTotalTokens += value.births.size();
	}

	EXPECT_EQ(tokenHistory.tokens.size(), nbTotalTokens);
}