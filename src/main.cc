// based on compTest.py from MidiTokMusicGen
// Test in C

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <array>

#include "gen.h"

void OnPitch(void* data, unsigned char pitch)
{
	int pitchInt = int(pitch);

    std::cout << pitchInt << std::endl;
}

#include <cassert>

#include "musicGenerator.hpp" // @TODO : remove

void testComp()
{
	EnvHandle env = createEnv(false);
	MidiTokenizerHandle tok = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");
	MusicGeneratorHandle generator = createMusicGenerator();

	generator_loadOnnxModel(generator, env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model_past.onnx");

	int32_t input_ids[] = {
	942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
	12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
	2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
	};

	std::vector<std::int32_t> inputIds(std::begin(input_ids), std::end(input_ids));

	auto start = std::chrono::high_resolution_clock::now();

	// RunInstanceHandle runInstance = createRunInstance();
	RunInstanceHandle runInstance = generator_createRunInstance(generator);
	RunInstanceHandle runInstanceForced = generator_createRunInstance(generator);
	BatchHandle batch = createBatch();
	BatchHandle batchForced = createBatch();

	int LineNbMaxToken = 256;

	runInstance_addBatch(runInstance, batch);
	runInstance_addBatch(runInstanceForced, batchForced);
	runInstance_setMaxInputLength(runInstance, LineNbMaxToken);
	runInstance_setMaxInputLength(runInstanceForced, LineNbMaxToken);

	batch_set(batch, inputIds.data(), inputIds.size(), 0);

	std::vector<int32_t> encodedTokensVec;
	for (int i = 0; i < 1000; i++)
	{
		{
			runInstance_reset(runInstanceForced);

			std::vector<std::int32_t> context;
			int32_t start = std::max(0, int(inputIds.size() - LineNbMaxToken));
			for (int i = start; i < inputIds.size(); i++)
			{
				context.push_back(inputIds[i]);
			}
			batch_set(batchForced, context.data(), context.size(), 0);
		}

		// generator_generateNextToken(generator, runInstance);
		generator_preGenerate(generator, runInstance);
		generator_preGenerate(generator, runInstanceForced);
		generator_generate(generator, runInstance);
		generator_generate(generator, runInstanceForced);



		const float* presentTensor = runInstance_getPresentTensor(runInstance, 0);
		const float* presentTensorForced = runInstance_getPresentTensor(runInstanceForced, 0);

		std::array<std::int64_t, 5> presentShape;
		runInstance_getPresentTensorShape(runInstance, generator, presentShape.data());
		std::array<std::int64_t, 5> presentShapeForced;
		runInstance_getPresentTensorShape(runInstanceForced, generator, presentShapeForced.data());

		assert(presentShape == presentShapeForced);

		float v1;
		float v2;
		float v3;
		float v1Forced;
		float v2Forced;
		float v3Forced;

		if (presentShape[3] > 3 && presentShapeForced[3] > 3)
		{

			std::array<std::int64_t, 5> indices1 = {0,0,0,3,0};
			std::array<std::int64_t, 5> indices2 = {0,0,0,2,0};
			std::array<std::int64_t, 5> indices3 = {0,0,0,1,0};

			v1 = presentTensor[computeMultiDimIndex(presentShape.data(), indices1.data())];
			v2 = presentTensor[computeMultiDimIndex(presentShape.data(), indices2.data())];
			v3 = presentTensor[computeMultiDimIndex(presentShape.data(), indices3.data())];
			v1Forced = presentTensorForced[computeMultiDimIndex(presentShapeForced.data(), indices1.data())];
			v2Forced = presentTensorForced[computeMultiDimIndex(presentShapeForced.data(), indices2.data())];
			v3Forced = presentTensorForced[computeMultiDimIndex(presentShapeForced.data(), indices3.data())];



			float r = v1-v2;
		}

		generator_postGenerate(generator, runInstance);
		generator_postGenerate(generator, runInstanceForced);

		int32_t newToken = batch_getLastGeneratedToken(batch);
		int32_t newTokenForced = batch_getLastGeneratedToken(batchForced);

		// if (newToken != newTokenForced)
		// {
		// 	std::cout << "runInstance\n";
		// 	PrintTensorContent<int32_t>(runInstance->presentTensors[0]);

		// 	std::cout << "runInstanceForced\n";
		// 	PrintTensorContent<int32_t>(runInstanceForced->presentTensors[0]);
		// }

		assert(newToken == newTokenForced);
		encodedTokensVec.push_back(newToken);
		inputIds.push_back(newToken);
	}

	// DataType* encodedTokens = nullptr; 
	// std::int32_t nbTokens;
	// batch_getEncodedTokens(batch, &encodedTokens, &nbTokens);

	// std::vector<int32_t> encodedTokensVec;
	// for (int i = 0; i < nbTokens; i++)
	// {
	// 	encodedTokensVec.push_back(encodedTokens[i]);
	// }


	int32_t* outTokens = nullptr;
	int32_t outTokensSize = 0;
	tokenizer_decodeIDs(tok, encodedTokensVec.data(), encodedTokensVec.size(), &outTokens, &outTokensSize);

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

int main()
{
	testComp();
	return 0;

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

	std::vector<std::int32_t> inputIds(std::begin(input_ids), std::end(input_ids));

	auto start = std::chrono::high_resolution_clock::now();

	// RunInstanceHandle runInstance = createRunInstance();
	RunInstanceHandle runInstance = generator_createRunInstance(generator);
	BatchHandle batch = createBatch();

	int LineNbMaxToken = 256;

	runInstance_addBatch(runInstance, batch);
	runInstance_setMaxInputLength(runInstance, LineNbMaxToken);

	bool forceUpdate = false;

	if (!forceUpdate)
	{
		batch_set(batch, inputIds.data(), inputIds.size(), 0);
	}

	std::vector<int32_t> encodedTokensVec;
	for (int i = 0; i < 1000; i++)
	{
		if (forceUpdate)
		{
			runInstance_reset(runInstance);

			std::vector<std::int32_t> context;
			int32_t start = std::max(0, int(inputIds.size() - LineNbMaxToken));
			for (int i = start; i < inputIds.size(); i++)
			{
				context.push_back(inputIds[i]);
			}
			batch_set(batch, context.data(), context.size(), 0);
		}

		// generator_generateNextToken(generator, runInstance);
		generator_preGenerate(generator, runInstance);
		generator_generate(generator, runInstance);

		const float* presentTensor = runInstance_getPresentTensor(runInstance, 0);

		generator_postGenerate(generator, runInstance);

		int32_t newToken = batch_getLastGeneratedToken(batch);
		encodedTokensVec.push_back(newToken);
		inputIds.push_back(newToken);
	}

	// DataType* encodedTokens = nullptr; 
	// std::int32_t nbTokens;
	// batch_getEncodedTokens(batch, &encodedTokens, &nbTokens);

	// std::vector<int32_t> encodedTokensVec;
	// for (int i = 0; i < nbTokens; i++)
	// {
	// 	encodedTokensVec.push_back(encodedTokens[i]);
	// }


	int32_t* outTokens = nullptr;
	int32_t outTokensSize = 0;
	tokenizer_decodeIDs(tok, encodedTokensVec.data(), encodedTokensVec.size(), &outTokens, &outTokensSize);

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