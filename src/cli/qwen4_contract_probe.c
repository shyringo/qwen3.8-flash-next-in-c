#include "qwen4_model.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL-00001-of-N.gguf\n", argv[0]);
        return 2;
    }
    Q4Model *model = q4_model_open_gguf(argv[1], 2048u);
    if (!model) return 1;
    printf("layers=%u hidden=%u vocab=%u experts=%u active=%u\n",
           q4_model_layer_count(model), q4_model_hidden_size(model),
           q4_model_vocab_size(model), q4_model_expert_count(model),
           q4_model_active_expert_count(model));
    q4_model_close(model);
    return 0;
}
