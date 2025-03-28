#include "mistral.h"
#include "modelBuilderManager.hpp"
#include "llama.hpp"

void registerMistralModelBuilder()
{
    getModelBuilderManager().registerModelBuilder("mistral", new Llama::LlamaBuilder());
}