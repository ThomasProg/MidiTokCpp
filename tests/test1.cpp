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
	inf.runInference(Path, false);
}

TEST(MidiTokTests, LlamaTest)
{

}

TEST(MidiTokTests, MistralTest)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.runInference(Path, false);
}

class GenerationHistoryTest
{
public: 
	static void test(Inf& inf, GenerationHistory* history)
	{
		history->converter = new TSDConverter();
		history->converter->tokenizerHandle = inf.Tokenizer.get();
		history->convert();
	
		for (const Note& note : history->notes)
		{
			std::cout << "Tick: " << note.tick << " / Pitch: " << note.pitch << std::endl;
		}
	
		history->removeAfterTick(30);
	
		std::cout << " === REMOVED === " << std::endl;
	
		for (const Note& note : history->notes)
		{
			std::cout << "Tick: " << note.tick << " / Pitch: " << note.pitch << std::endl;
		}
	
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
	inf.NbTokensToGenerate = 20;
	inf.runInference(Path, false);

	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);

	GenerationHistoryTest test;

	test.test(inf, history);


}
