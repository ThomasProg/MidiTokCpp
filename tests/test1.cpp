#include "gtest/gtest.h"

#include "inference.hpp"
#include "modelBuilderManager.hpp"
#include "llama.hpp"
#include "musicGenerator.hpp"
#include <iostream>
#include "note.h"
#include "midiConverter.hpp"

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
		std::cout << note.tick << std::endl;
		EXPECT_GE(note.tick, lastTick);
		lastTick = note.tick;
	}
}