#ifndef QWEN4_GGUF_H
#define QWEN4_GGUF_H

#include "qwen38_gguf.h"

#include <stddef.h>

#define Q4_GGUF_MAX_SHARDS 16u

typedef struct {
    Q38GGUF shards[Q4_GGUF_MAX_SHARDS];
    size_t shard_count;
} Q4GGUFSet;

int q4_gguf_set_open(Q4GGUFSet *set, const char *first_shard,
                     int header_only);
void q4_gguf_set_close(Q4GGUFSet *set);
const Q38GGUFTensor *q4_gguf_find_tensor(const Q4GGUFSet *set,
                                         const char *name);
const Q38GGUF *q4_gguf_metadata(const Q4GGUFSet *set);

#endif
