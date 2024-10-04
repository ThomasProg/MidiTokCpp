// based on compTest.py from MidiTokMusicGen
// Test in C

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "gen.h"

void OnPitch(void* data, unsigned char pitch)
{
	int pitchInt = int(pitch);

    std::cout << pitchInt << std::endl;
}

int main()
{
	EnvHandle env = createEnv(false);
	MidiTokenizerHandle tok = createMidiTokenizer(WORKSPACE_PATH "/tokenizer.json");
	MusicGeneratorHandle generator = createMusicGenerator();

	generator_loadOnnxModel(generator, env, WORKSPACE_PATH "/onnx_model_path/gpt2-midi-model3_past.onnx");

	int32_t input_ids[] = {
	942,    65,  1579,  1842,   616,    46,  3032,  1507,   319,  1447,
	12384,  1016,  1877,   319, 15263,  3396,   302,  2667,  1807,  3388,
	2649,  1173,    50,   967,  1621,   256,  1564,   653,  1701,   377
	};

	int32_t size = sizeof(input_ids) / sizeof(*input_ids);
	InputHandle input = generator_generateInput(generator, input_ids, size);


	for (int i = 0; i < 10; i++)
	{
		generator_generateNextToken(generator, input);
	}

	int32_t* outTokens = nullptr;
	int32_t outTokensSize = 0;

	input_decodeIDs(input, tok, &outTokens, &outTokensSize);





	RedirectorHandle redirector = createRedirector();

	redirector_bindPitch(redirector, tok, "Pitch_", nullptr, OnPitch);

	for (int32_t i = 0; i < outTokensSize; i++)
	{
		bool found = redirector_call(redirector, outTokens[i]);
	}

	input_decodeIDs_free(outTokens);

	destroyRedirector(redirector);

	generator_generateInput_free(input);



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