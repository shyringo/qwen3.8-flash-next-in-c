#define _POSIX_C_SOURCE 200809L

#include "qwen4_model.h"

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

static uint32_t argmax(const float *values, uint32_t count)
{
    uint32_t best = 0u;
    for (uint32_t i = 1u; i < count; ++i)
        if (values[i] > values[best]) best = i;
    return best;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL-00001-of-N.gguf\n", argv[0]);
        return 2;
    }
    static const uint32_t tokens[4] = {248045u, 846u, 198u, 116348u};
    Q4Model *model = q4_model_open_gguf(argv[1], 256u);
    if (!model) {
        fprintf(stderr, "qwen4 batch bench: model open failed\n");
        return 1;
    }
    const float *logits = NULL;
    uint32_t expected[4], actual[4];
    for (uint32_t i = 0; i < 4u; ++i) {
        if (!q4_model_forward_token(model, tokens[i], &logits)) {
            fprintf(stderr, "qwen4 batch bench: reference token %u failed\n", i);
            return 1;
        }
        expected[i] = argmax(logits, q4_model_vocab_size(model));
    }
    const size_t bytes = (size_t)q4_model_vocab_size(model) * sizeof(float);
    float *reference = (float *)malloc(bytes);
    if (!reference) {
        fprintf(stderr, "qwen4 batch bench: reference allocation failed\n");
        return 1;
    }
    memcpy(reference, logits, bytes);
    q4_model_reset(model);
    const double started = now_seconds();
    if (!q4_model_verify_greedy(model, tokens, 4u, actual, &logits)) {
        fprintf(stderr, "qwen4 batch bench: batch verification failed\n");
        return 1;
    }
    const double elapsed = now_seconds() - started;
    const int exact = memcmp(expected, actual, sizeof(expected)) == 0 &&
                      memcmp(reference, logits, bytes) == 0;
    printf("batch=4 result=%s seconds=%.6f throughput=%.3f positions/s\n",
           exact ? "byte-exact" : "DIFFERS", elapsed, 4.0 / elapsed);
    printf("top1=%u,%u,%u,%u\n",
           actual[0], actual[1], actual[2], actual[3]);
    free(reference);
    q4_model_close(model);
    return exact ? 0 : 1;
}
