#include "qwen38_gguf.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}

static uint32_t read_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t read_u64(const uint8_t *p)
{
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
    return value;
}

static void print_scalar(uint32_t type, const uint8_t *data)
{
    switch (type) {
    case Q38_GGUF_META_UINT8: printf("%u", data[0]); break;
    case Q38_GGUF_META_INT8: printf("%d", (int8_t)data[0]); break;
    case Q38_GGUF_META_UINT16: printf("%u", read_u16(data)); break;
    case Q38_GGUF_META_INT16: printf("%d", (int16_t)read_u16(data)); break;
    case Q38_GGUF_META_UINT32: printf("%" PRIu32, read_u32(data)); break;
    case Q38_GGUF_META_INT32: printf("%" PRId32, (int32_t)read_u32(data)); break;
    case Q38_GGUF_META_UINT64: printf("%" PRIu64, read_u64(data)); break;
    case Q38_GGUF_META_INT64: printf("%" PRId64, (int64_t)read_u64(data)); break;
    case Q38_GGUF_META_FLOAT32: {
        uint32_t bits = read_u32(data);
        float value;
        memcpy(&value, &bits, sizeof(value));
        printf("%.9g", value);
        break;
    }
    case Q38_GGUF_META_FLOAT64: {
        uint64_t bits = read_u64(data);
        double value;
        memcpy(&value, &bits, sizeof(value));
        printf("%.17g", value);
        break;
    }
    case Q38_GGUF_META_BOOL: printf("%s", data[0] ? "true" : "false"); break;
    default: printf("?"); break;
    }
}

static size_t scalar_size(uint32_t type)
{
    switch (type) {
    case Q38_GGUF_META_UINT8:
    case Q38_GGUF_META_INT8:
    case Q38_GGUF_META_BOOL: return 1u;
    case Q38_GGUF_META_UINT16:
    case Q38_GGUF_META_INT16: return 2u;
    case Q38_GGUF_META_UINT32:
    case Q38_GGUF_META_INT32:
    case Q38_GGUF_META_FLOAT32: return 4u;
    case Q38_GGUF_META_UINT64:
    case Q38_GGUF_META_INT64:
    case Q38_GGUF_META_FLOAT64: return 8u;
    default: return 0u;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL-SHARD.gguf\n", argv[0]);
        return 2;
    }
    Q38GGUF gguf;
    if (!q38_gguf_open(&gguf, argv[1])) return 1;
    printf("GGUF v%u tensors=%" PRIu64 " metadata=%" PRIu64 "\n",
           gguf.version, gguf.tensor_count, gguf.metadata_count);
    for (uint64_t i = 0; i < gguf.metadata_count; ++i) {
        const Q38GGUFMeta *meta = &gguf.metadata[i];
        printf("%.*s = ", (int)meta->key.length, meta->key.data);
        if (meta->type == Q38_GGUF_META_STRING) {
            printf("%.*s", (int)meta->data_size, (const char *)meta->data);
        } else if (meta->type == Q38_GGUF_META_ARRAY) {
            printf("array(type=%u,count=%" PRIu64 ")", meta->array_type,
                   meta->count);
            const size_t size = scalar_size(meta->array_type);
            if (size && meta->count <= 64u) {
                printf(" [");
                for (uint64_t j = 0; j < meta->count; ++j) {
                    if (j) printf(", ");
                    print_scalar(meta->array_type, meta->data + j * size);
                }
                printf("]");
            }
        } else {
            print_scalar(meta->type, meta->data);
        }
        putchar('\n');
    }
    q38_gguf_close(&gguf);
    return 0;
}
