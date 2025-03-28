#include "gtest/gtest.h"

#include "inference.hpp"
#include "modelBuilderManager.hpp"
#include "llama.hpp"
#include "musicGenerator.hpp"

TEST(MidiTokTests, GptTest)
{
    getModelBuilderManager().registerModelBuilder("gpt2", new MusicGeneratorBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.2.10";
	runInference(Path, false);
}

TEST(MidiTokTests, LlamaTest)
{

}

TEST(MidiTokTests, MistralTest)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	runInference(Path, false);
}
