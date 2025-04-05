#include "gtest/gtest.h"

#include "inference.hpp"
#include "modelBuilderManager.hpp"
#include "llama.hpp"
#include "musicGenerator.hpp"
#include <iostream>
#include "note.h"
#include "midiConverter.hpp"
#include <chrono>

TEST(MidiTokTests, GptTest)
{
    getModelBuilderManager().registerModelBuilder("gpt2", new MusicGeneratorBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.2.10";
	Inf inf;
	inf.load(Path, false);
	inf.runInference();
}

TEST(MidiTokTests, LlamaTest)
{

}

TEST(MidiTokTests, MistralTest)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.load(Path, false);
	inf.runInference();
}

class GenerationHistoryTest
{
public: 
	static void test(Inf& inf, GenerationHistory* history)
	{
		if (history->converter != nullptr)
		{
			history->converter = new TSDConverter();
		}
		history->converter->tokenizerHandle = inf.Tokenizer.get();
		history->convert();
	
		for (const Note& note : history->notes)
		{
			std::cout << "Tick: " << note.tick << " / Pitch: " << note.pitch << std::endl;
		}
	
		// history->removeAfterTick(30);
	
		// std::cout << " === REMOVED === " << std::endl;
	
		// for (const Note& note : history->notes)
		// {
		// 	std::cout << "Tick: " << note.tick << " / Pitch: " << note.pitch << std::endl;
		// }
	
		EXPECT_TRUE(history->notes.size() == history->noteIndexToDecodedTokenIndex.size());
		EXPECT_TRUE(history->decodedTokenIndexToEncodedTokenIndex.size() == history->getDecodedTokensHistory().getTokensSize());
		TokenHistoryTest::TestTokenHistory(history->getDecodedTokensHistory());
		TokenHistoryTest::TestTokenHistory(history->getEncodedTokensHistory());
	}
};

TEST(MidiTokTests, MistralTestHistory)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 100;
	inf.load(Path, false);
	inf.runInference();

	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);

	GenerationHistoryTest test;

	test.test(inf, history);


}

TEST(MidiTokTests, MistralRegenerateTest)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 100;
	inf.load(Path, false);
	inf.runInference();

	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);

	inf.ARPipeline->batchUnwind(inf.Batch2, 50);

	inf.runInference();

	history->convert();
	const std::vector<Note>& notes = history->getNotes();

	std::cout << "Note ticks" << std::endl;
	int lastTick = 0;
	for (const Note& note : notes)
	{
		EXPECT_GE(note.tick, lastTick);
		lastTick = note.tick;
	}
}

TEST(MidiTokTests, MistralInference)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 300;
	inf.load(Path, false);
	inf.runInference();

	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);
	history->convert();
	const TokenHistory& encodedTokenHistory = history->getEncodedTokensHistory();
	const std::vector<int32_t>& encodedTokens = encodedTokenHistory.getTokens();

	for (const int32_t& token : encodedTokens)
	{
		std::cout << token << ", ";
	}
	std::cout << std::endl;
}

TEST(MidiTokTests, MistralInferenceTime)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 1026;
	inf.load(Path, false);
	inf.runInference();

	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);
	history->convert();

	inf.NbTokensToGenerate = 1000;

	auto start = std::chrono::high_resolution_clock::now();	
	
	inf.runInference();
	
	auto end = std::chrono::high_resolution_clock::now();

	long long elapsed = (end - start).count();

	std::cout << "Elapsed: " << elapsed << std::endl;
	std::cout << "Elapsed per token: " << double(elapsed)  / double(inf.NbTokensToGenerate) << std::endl;

	std::cout << std::endl;
}

TEST(MidiTokTests, MistralInferenceTimeWithUnwind)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 1024+600;
	inf.load(Path, false);
	inf.runInference();

	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);
	history->convert();

	long long elapsed = 0;

	for (int i = 0; i < 100; i++)
	{
		const std::vector<Note>& notes = history->getNotes();
		inf.ARPipeline->batchUnwind(inf.Batch2, notes.back().tick-1);
		inf.NbTokensToGenerate = 1;

		auto start = std::chrono::high_resolution_clock::now();	
	
		inf.runInference();
		
		auto end = std::chrono::high_resolution_clock::now();

		elapsed += (end - start).count();
	}

	std::cout << "Elapsed: " << elapsed << std::endl;
	std::cout << "Elapsed per token: " << double(elapsed)  / double(inf.NbTokensToGenerate) << std::endl;

	std::cout << std::endl;
}