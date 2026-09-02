#include "qwen4_gguf.h"

#include <inttypes.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL-00001-of-N.gguf\n", argv[0]);
        return 2;
    }
    Q4GGUFSet set;
    if (!q4_gguf_set_open(&set, argv[1], 1)) return 1;
    for (size_t file = 0; file < set.shard_count; ++file) {
        const Q38GGUF *gguf = &set.shards[file];
        printf("# %s: tensors=%" PRIu64 " data_offset=%" PRIu64 "\n",
               argv[1], gguf->tensor_count, gguf->data_offset);
        for (uint64_t i = 0; i < gguf->tensor_count; ++i) {
            const Q38GGUFTensor *tensor = &gguf->tensors[i];
            printf("%.*s type=%u shape=[", (int)tensor->name.length,
                   tensor->name.data, tensor->type);
            for (uint32_t dim = 0; dim < tensor->n_dims; ++dim)
                printf("%s%" PRIu64, dim ? "," : "", tensor->shape[dim]);
            printf("] bytes=%" PRIu64 " available=%s\n", tensor->nbytes,
                   tensor->data ? "yes" : "no");
        }
    }
    q4_gguf_set_close(&set);
    return 0;
}
