#define _POSIX_C_SOURCE 200809L

#include "qwen4_model.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKEN_ID [...] [--prefill]\n", argv[0]);
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
    if (!model) return 1;
    const float *logits = NULL;
    const double start = now_seconds();
    const int prefill = strcmp(argv[argc - 1], "--prefill") == 0;
    const int token_end = argc - prefill;
    const uint32_t token_count = (uint32_t)(token_end - 2);
    uint32_t *tokens = (uint32_t *)malloc(token_count * sizeof(*tokens));
    if (!tokens) { q4_model_close(model); return 1; }
    for (int i = 2; i < token_end; ++i) {
        char *end = NULL;
        const unsigned long token = strtoul(argv[i], &end, 10);
        if (!end || *end || token >= q4_model_vocab_size(model)) {
            free(tokens);
            q4_model_close(model);
            return 1;
        }
        tokens[i - 2] = (uint32_t)token;
        if (!prefill && !q4_model_forward_token(model, (uint32_t)token, &logits)) {
            free(tokens); q4_model_close(model); return 1;
        }
    }
    if (prefill && !q4_model_prefill(model, tokens, token_count, &logits)) {
        free(tokens); q4_model_close(model); return 1;
    }
    free(tokens);
    uint32_t ids[5] = {0};
    float values[5] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t id = 0; id < q4_model_vocab_size(model); ++id) {
        int slot = 4;
        if (logits[id] <= values[slot]) continue;
        while (slot > 0 && logits[id] > values[slot - 1]) {
            values[slot] = values[slot - 1]; ids[slot] = ids[slot - 1]; --slot;
        }
        values[slot] = logits[id]; ids[slot] = id;
    }
    printf("tokens=%u mode=%s time=%.6f top5=", token_count,
           prefill ? "prefill" : "sequential", now_seconds() - start);
    for (int i = 0; i < 5; ++i)
        printf("%s%u:%.9g", i ? "," : "", ids[i], values[i]);
    putchar('\n');
    const char *dump = getenv("Q4_DUMP_LOGITS");
    if (dump && *dump) {
        FILE *file = fopen(dump, "wb");
        if (!file || fwrite(logits, sizeof(float), q4_model_vocab_size(model),
                            file) != q4_model_vocab_size(model) || fclose(file)) {
            q4_model_close(model);
            return 1;
        }
    }
    q4_model_close(model);
    return 0;
}
