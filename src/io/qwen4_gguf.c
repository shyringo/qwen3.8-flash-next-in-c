#include "qwen4_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int split_path(char *output, size_t capacity, const char *first,
                      size_t shard, size_t count)
{
    static const char marker[] = "-00001-of-";
    const char *at = strstr(first, marker);
    if (!at || strstr(at + 1, marker)) return 0;
    const char *total = at + sizeof(marker) - 1u;
    char expected[16];
    const int expected_length = snprintf(expected, sizeof(expected),
                                         "%05zu.gguf", count);
    if (expected_length <= 0 || strcmp(total, expected) != 0) return 0;
    const int prefix = (int)(at - first);
    const int length = snprintf(output, capacity, "%.*s-%05zu-of-%05zu.gguf",
                                prefix, first, shard, count);
    return length > 0 && (size_t)length < capacity;
}

void q4_gguf_set_close(Q4GGUFSet *set)
{
    if (!set) return;
    for (size_t i = 0; i < set->shard_count; ++i)
        q38_gguf_close(&set->shards[i]);
    memset(set, 0, sizeof(*set));
}

int q4_gguf_set_open(Q4GGUFSet *set, const char *first_shard,
                     int header_only)
{
    if (!set || !first_shard) return 0;
    memset(set, 0, sizeof(*set));
    if (!(header_only ? q38_gguf_open_header(&set->shards[0], first_shard)
                      : q38_gguf_open(&set->shards[0], first_shard)))
        return 0;
    set->shard_count = 1u;
    uint64_t count = 1u;
    if (!q38_gguf_meta_u64(&set->shards[0], "split.count", &count)) count = 1u;
    if (!count || count > Q4_GGUF_MAX_SHARDS) {
        fprintf(stderr, "qwen4: unsupported GGUF shard count %llu\n",
                (unsigned long long)count);
        q4_gguf_set_close(set);
        return 0;
    }
    if (count == 1u) return 1;
    char path[4096];
    for (size_t shard = 2u; shard <= (size_t)count; ++shard) {
        if (!split_path(path, sizeof(path), first_shard, shard, (size_t)count) ||
            !(header_only ? q38_gguf_open_header(&set->shards[shard - 1u], path)
                          : q38_gguf_open(&set->shards[shard - 1u], path))) {
            fprintf(stderr, "qwen4: unable to open GGUF shard %zu of %llu\n",
                    shard, (unsigned long long)count);
            q4_gguf_set_close(set);
            return 0;
        }
        set->shard_count = shard;
        uint64_t shard_count = 0;
        uint64_t shard_no = UINT64_MAX;
        if (!q38_gguf_meta_u64(&set->shards[shard - 1u], "split.count",
                               &shard_count) || shard_count != count ||
            !q38_gguf_meta_u64(&set->shards[shard - 1u], "split.no",
                               &shard_no) || shard_no != shard - 1u) {
            fprintf(stderr, "qwen4: inconsistent GGUF split metadata\n");
            q4_gguf_set_close(set);
            return 0;
        }
    }
    return 1;
}

const Q38GGUFTensor *q4_gguf_find_tensor(const Q4GGUFSet *set,
                                         const char *name)
{
    if (!set || !name) return NULL;
    const Q38GGUFTensor *found = NULL;
    for (size_t i = 0; i < set->shard_count; ++i) {
        const Q38GGUFTensor *candidate =
            q38_gguf_find_tensor(&set->shards[i], name);
        if (!candidate) continue;
        if (found) {
            fprintf(stderr, "qwen4: duplicate tensor across shards: %s\n", name);
            return NULL;
        }
        found = candidate;
    }
    return found;
}

const Q38GGUF *q4_gguf_metadata(const Q4GGUFSet *set)
{
    return set && set->shard_count ? &set->shards[0] : NULL;
}
