#include "gtest/gtest.h"

#include "inference.hpp"
#include "modelBuilderManager.hpp"
#include "llama.hpp"
#include "musicGenerator.hpp"
#include <iostream>
#include "note.h"
#include "midiConverter.hpp"
#include "generationHistory.h"
#include <chrono>
#include "onAddTokensArgs.hpp"
#include "gen.h"

TEST(MidiTokTests, MistralAddDecodedTokensOnTokenTest)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 100;
	inf.load(Path, false);
    
	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);
    generationHistory_setOnEncodedTokenAddedData(history, history);
    generationHistory_setOnEncodedTokenAdded(history, [](OnAddTokensArgs* args)
    {
        const MidiTokenizer& tokenizer = args->getTokenizer();

        const int32_t* outDecodedTokensBegin;
        const int32_t* outDecodedTokensEnd;
        tokenizer_decodeTokenFast(&tokenizer, args->getNewEncodedToken(), &outDecodedTokensBegin, &outDecodedTokensEnd);

        while (outDecodedTokensBegin != outDecodedTokensEnd)
        {
            int32_t decodedToken = *outDecodedTokensBegin;

            // if (isPitch(&tokenizer, decodedToken))
            // {
            //     int32_t pitch = getPitch(&tokenizer, decodedToken);

            //     // Triad
            //     args->addDecodedToken(findPitchToken(&tokenizer, 1));
            // }
            args->addDecodedToken(findPitchToken(&tokenizer, 40));
            args->addDecodedToken(*outDecodedTokensBegin);

            ++outDecodedTokensBegin;
        }

        GenerationHistory* History = (GenerationHistory*) args;
        generationHistory_setOnEncodedTokenAdded(History, generationHistory_getDefaultOnEncodedTokenAdded());
    });

	inf.runInference();

	const TokenHistory& decodedTokenHistory = history->getDecodedTokensHistory();
	const std::vector<int32_t>& decodedTokens = decodedTokenHistory.getTokens();

    MidiTokenizer* tok = inf.Tokenizer.get();

	for (const int32_t& token : decodedTokens)
	{
		std::cout << tok->decodedTokenToString(token) << std::endl;
	}
	std::cout << std::endl;
}


TEST(MidiTokTests, MistralAddNotesOnTokenTest)
{
	getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());

	const char* Path = WORKSPACE_PATH "Models/TSD/1.3.2";
	Inf inf;
	inf.NbTokensToGenerate = 100;
	inf.load(Path, false);
    
	GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);
    history->onNoteAddedData = &inf;
    history->onNoteAdded = [](void* userData)
    {
        Inf& inf = *(Inf*)userData;

        GenerationHistory* history = inf.ARPipeline->getHistory(inf.Batch2);
        
        const std::vector<Note>& notes = history->getNotes();

        if (notes.empty())
        {
            return;
        }

        Note note;
        note.tick = notes.back().tick + 5;
        note.duration = 1;
        note.pitch = 60;
        note.velocity = 100;
        history->addStandaloneNote(note);
    };
	inf.runInference();

    history->convert();
	const TokenHistory& decodedTokenHistory = history->getDecodedTokensHistory();
	const std::vector<Note>& notes = history->getNotes();

    MidiTokenizer* tok = inf.Tokenizer.get();

	for (const Note& note : notes)
	{
		std::cout << "Pitch: " << note.pitch << " / Duration: " << note.duration << " / Velocity:" << note.velocity << std::endl;
	}
	std::cout << std::endl;
}
