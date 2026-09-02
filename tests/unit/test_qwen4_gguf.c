#include "qwen4_gguf.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

static void put_u32(FILE *file, uint32_t value)
{
    for (int i = 0; i < 4; ++i) fputc((int)(value >> (8 * i)) & 255, file);
}

static void put_u64(FILE *file, uint64_t value)
{
    for (int i = 0; i < 8; ++i) fputc((int)(value >> (8 * i)) & 255, file);
}

static void put_string(FILE *file, const char *text)
{
    const size_t length = strlen(text);
    put_u64(file, (uint64_t)length);
    fwrite(text, 1, length, file);
}

static int write_shard(const char *path, uint32_t number,
                       const char *tensor_name, float value)
{
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    put_u32(file, 0x46554747u);
    put_u32(file, 3u);
    put_u64(file, tensor_name ? 1u : 0u);
    put_u64(file, 3u);

    put_string(file, "general.architecture");
    put_u32(file, Q38_GGUF_META_STRING);
    put_string(file, "qwen4exp");
    put_string(file, "split.count");
    put_u32(file, Q38_GGUF_META_UINT16);
    fputc(2, file); fputc(0, file);
    put_string(file, "split.no");
    put_u32(file, Q38_GGUF_META_UINT16);
    fputc((int)number, file); fputc(0, file);

    if (tensor_name) {
        put_string(file, tensor_name);
        put_u32(file, 1u);
        put_u64(file, 1u);
        put_u32(file, Q38_GGML_F32);
        put_u64(file, 0u);
    }
    long offset = ftell(file);
    if (offset < 0) { fclose(file); return 0; }
    while ((offset++ & 31) != 0) fputc(0, file);
    if (tensor_name) fwrite(&value, sizeof(value), 1u, file);
    return fclose(file) == 0;
}

int main(void)
{
    const char *first = "build/synthetic-00001-of-00002.gguf";
    const char *second = "build/synthetic-00002-of-00002.gguf";
    CHECK(write_shard(first, 0u, NULL, 0.0f), "write metadata shard");
    CHECK(write_shard(second, 1u, "test.weight", 42.0f),
          "write tensor shard");

    Q4GGUFSet set;
    CHECK(q4_gguf_set_open(&set, first, 0), "open split GGUF set");
    CHECK(set.shard_count == 2u, "discover two shards");
    const Q38GGUF *metadata = q4_gguf_metadata(&set);
    CHECK(metadata && metadata->tensor_count == 0u,
          "metadata-only first shard");
    uint64_t split_count = 0;
    CHECK(q38_gguf_meta_u64(metadata, "split.count", &split_count) &&
          split_count == 2u, "read non-u32 split metadata");
    const Q38GGUFTensor *tensor = q4_gguf_find_tensor(&set, "test.weight");
    float value = 0.0f;
    CHECK(tensor && tensor->data, "cross-shard tensor lookup");
    memcpy(&value, tensor->data, sizeof(value));
    CHECK(value == 42.0f, "mapped tensor value");
    q4_gguf_set_close(&set);

    puts("qwen4 split GGUF: ok");
    return 0;
}
