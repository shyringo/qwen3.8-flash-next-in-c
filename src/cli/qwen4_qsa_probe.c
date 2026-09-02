#define _POSIX_C_SOURCE 200809L

#include "qwen4_model.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKEN_ID TOKEN_COUNT\n", argv[0]);
        return 2;
    }
    char *token_end = NULL, *count_end = NULL;
    const unsigned long token = strtoul(argv[2], &token_end, 10);
    const unsigned long count = strtoul(argv[3], &count_end, 10);
    if (!token_end || *token_end || !count_end || *count_end || !count ||
        token > UINT32_MAX || count > 262144u) return 2;
    const uint32_t context = count < 2049u ? 2049u : (uint32_t)count;
    Q4Model *model = q4_model_open_gguf(argv[1], context);
    if (!model || token >= q4_model_vocab_size(model)) {
        q4_model_close(model); return 1;
    }
    const float *logits = NULL;
    const double started = now_seconds();
    for (uint32_t position = 0; position < (uint32_t)count; ++position) {
        if (!q4_model_forward_token(model, (uint32_t)token, &logits)) {
            q4_model_close(model); return 1;
        }
        if ((position + 1u) % 128u == 0u || position + 1u == count)
            fprintf(stderr, "qwen4 QSA probe: %u/%lu tokens\n",
                    position + 1u, count);
    }
    uint32_t best = 0u;
    float value = -FLT_MAX;
    for (uint32_t id = 0; id < q4_model_vocab_size(model); ++id) {
        if (logits[id] > value) { value = logits[id]; best = id; }
    }
    printf("tokens=%lu top1=%u:%g elapsed=%.3fs\n",
           count, best, value, now_seconds() - started);
    q4_model_close(model);
    return 0;
}
