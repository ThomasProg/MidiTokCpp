// based on compTest.py from MidiTokMusicGen
// Test in C

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <vector>

#include "gen.h"

void OnPitch(void* data, unsigned char pitch)
{
	int pitchInt = int(pitch);

    std::cout << pitchInt << std::endl;
}

void test()
{
	EnvHandle env = createEnv(true);
	MidiTokenizerHandle tok = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");
	MusicGeneratorHandle generator = createMusicGenerator();

	generator_loadOnnxModel(generator, env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx");

	RunInstanceHandle runInstance = createRunInstance();
	// runInstance_resetBatches(runInstance, 1);

	// BatchHandle batch = runInstance_getBatch(runInstance, 0);

	BatchHandle batch = createBatch();
	runInstance_addBatch(runInstance, batch);

	int32_t input_ids[] = {
	942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
	12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
	2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
	};
	int32_t size = sizeof(input_ids) / sizeof(*input_ids);
	batch_set(batch, input_ids, size, 0);

	for (int i = 0; i < 10; i++)
	{
		generator_generateNextToken(generator, runInstance);
	}

	batch_pop(batch);
	batch_pop(batch);
	batch_pop(batch);

	batch_push(batch, 942);
	batch_push(batch, 65);
	batch_push(batch, 1579);

	for (int i = 0; i < 10; i++)
	{
		generator_generateNextToken(generator, runInstance);
	}

	int32_t* tokens;
	int32_t tokensSize;
	batch_getEncodedTokens(batch, &tokens, &tokensSize);


}



int main()
{
	// test();

	std::cout << "Workspace path : " << WORKSPACE_PATH << std::endl;

	EnvHandle env = createEnv(false);
	MidiTokenizerHandle tok = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");
	MusicGeneratorHandle generator = createMusicGenerator();

	generator_loadOnnxModel(generator, env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model_past.onnx");

	int32_t input_ids[] = {
	942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
	12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
	2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
	};

	auto start = std::chrono::high_resolution_clock::now();

	int32_t size = sizeof(input_ids) / sizeof(*input_ids);
	// InputHandle input = generator_generateInput(generator, input_ids, size);
	RunInstanceHandle runInstance = createRunInstance();
	BatchHandle batch = createBatch();
	runInstance_addBatch(runInstance, batch);
	batch_set(batch, input_ids, size, 0);


	for (int i = 0; i < 150; i++)
	{
		generator_generateNextToken(generator, runInstance);
	}

	DataType* encodedTokens = nullptr; 
	std::int32_t nbTokens;
	batch_getEncodedTokens(batch, &encodedTokens, &nbTokens);

	std::vector<int32_t> encodedTokensVec;
	for (int i = 0; i < nbTokens; i++)
	{
		encodedTokensVec.push_back(encodedTokens[i]);
	}


	int32_t* outTokens = nullptr;
	int32_t outTokensSize = 0;
	tokenizer_decodeIDs(tok, encodedTokens, nbTokens, &outTokens, &outTokensSize);

	std::vector<int32_t> outTokensVec;
	for (int i = 0; i < outTokensSize; i++)
	{
		outTokensVec.push_back(outTokens[i]);
	}


	// input_decodeIDs(input, tok, &outTokens, &outTokensSize);

    // Get the ending time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "duration : " << duration << std::endl;




	RedirectorHandle redirector = createRedirector();

	redirector_bindPitch(redirector, tok, "Pitch_", nullptr, OnPitch);

	for (int32_t i = 0; i < outTokensSize; i++)
	{
		bool found = redirector_call(redirector, outTokens[i]);
	}

	tokenizer_decodeIDs_free(outTokens);

	destroyRedirector(redirector);

	destroyBatch(batch);
	destroyRunInstance(runInstance);

	destroyMusicGenerator(generator);
	destroyMidiTokenizer(tok);
	destroyEnv(env);
}


// int main()
// {
    
//     // std::cout << "Current working directory: " <<  WORKSPACE_PATH << std::endl;

//     // std::unique_ptr<MidiTokenizer> tokenizer = std::make_unique<MidiTokenizer>(WORKSPACE_PATH "/tokenizer.json");
//     MidiTokenizerHandle tokenizer = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");

//     // std::unique_ptr<Ort::Env> env = MusicGenerator::createOnnxEnv();
//     EnvHandle env = createEnv(false);

//     MusicGeneratorHandle generator = createMusicGenerator();
//     generator_loadOnnxModel(generator, env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx");


//     Input::DataType input_ids[] = {
//         942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
//         12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
//         2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
//     };

//     InputHandle input = input()

//     // Input input = generator->generateInput(std::vector<Input::DataType>(std::begin(input_ids), std::end(input_ids)));
//     Input* input = generator_generateInput(generator, input_ids, sizeof(input_ids) / sizeof(*input_ids));

//     for (int i = 0; i < 1; i++)
//     {
//         generator->generate(*input);
//     }

//     for (auto& v : input->inputData[0])
//     {
//         std::cout << v << '\t';

//     }

//     Redirector redirector;

//     redirector.bindPitch(*tokenizer, "Pitch_", [](void*, unsigned char/*uint8*/ pitch)
//     {
//         std::cout << "Pitch : " << int(pitch) << std::endl;
//     });



//     std::vector<int32_t> outTokens;
//     tokenizer->decodeIDs(input->inputData[0], outTokens);

//     for (int32_t token : outTokens)
//     {
//         try 
//         {
//             redirector.call(token);
//         } 
//         catch(const std::exception&)
//         {

//         }


//     }

//     // tokenizer->decode(input.inputData[0]);

//     destroyMusicGenerator(generator);
//     destroyEnv(env);
//     destroyMidiTokenizer(tokenizer);
// }