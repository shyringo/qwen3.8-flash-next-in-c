#include "qwen4_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf PREFIX_TOKEN [...] NEXT_TOKEN\n",
                argv[0]);
        return 2;
    }
    uint32_t context = 2048u;
    const char *context_text = getenv("Q4_CONTEXT");
    if (context_text && *context_text) {
        char *end = NULL;
        const unsigned long value = strtoul(context_text, &end, 10);
        if (!end || *end || value > UINT32_MAX) return 2;
        context = (uint32_t)value;
    }
    Q4Model *model = q4_model_open_gguf(argv[1], context);
    fprintf(stderr, "state-probe: model opened\n");
    Q4ModelState *state = q4_model_state_create();
    fprintf(stderr, "state-probe: state allocated\n");
    float *first = NULL;
    if (!model || !state) goto fail;
    const float *logits = NULL;
    for (int i = 2; i + 1 < argc; ++i) {
        char *end = NULL;
        const unsigned long token = strtoul(argv[i], &end, 10);
        if (!end || *end || token >= q4_model_vocab_size(model) ||
            !q4_model_forward_token(model, (uint32_t)token, &logits))
            goto fail;
        fprintf(stderr, "state-probe: prefix token %d complete\n", i - 1);
    }
    if (!q4_model_state_save(model, state)) goto fail;
    fprintf(stderr, "state-probe: state saved\n");
    char *end = NULL;
    const unsigned long next = strtoul(argv[argc - 1], &end, 10);
    if (!end || *end || next >= q4_model_vocab_size(model) ||
        !q4_model_forward_token(model, (uint32_t)next, &logits)) goto fail;
    fprintf(stderr, "state-probe: first continuation complete\n");
    const size_t bytes = (size_t)q4_model_vocab_size(model) * sizeof(float);
    first = (float *)malloc(bytes);
    if (!first) goto fail;
    memcpy(first, logits, bytes);
    if (!q4_model_state_restore(model, state, &logits) ||
        !q4_model_forward_token(model, (uint32_t)next, &logits)) goto fail;
    fprintf(stderr, "state-probe: replay complete\n");
    if (memcmp(first, logits, bytes) != 0) {
        fprintf(stderr, "qwen4 state replay: logits differ\n");
        goto fail;
    }
    printf("qwen4 state replay: byte-exact at position %u\n",
           q4_model_position(model));
    free(first);
    q4_model_state_destroy(state);
    q4_model_close(model);
    return 0;
fail:
    free(first);
    q4_model_state_destroy(state);
    q4_model_close(model);
    return 1;
}
