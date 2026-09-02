#include "qwen4_ops.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main(void)
{
    const uint64_t mul[3] = { 23703573157769ULL, 20109073645365ULL,
                              8052911324071ULL };
    uint64_t offsets[16], sizes[16];
    for (uint32_t i = 0; i < 16u; ++i) {
        offsets[i] = (uint64_t)i * 20000003u;
        sizes[i] = 20000003u + i * 2u;
    }
    int32_t rows[16];
    q4_ple_indices(rows, 123u, 45u, 6u, mul, offsets, sizes);
    CHECK(rows[0] >= 0 && (uint64_t)rows[0] < offsets[0] + sizes[0],
          "bigram row range");
    CHECK((uint64_t)rows[8] >= offsets[8] &&
          (uint64_t)rows[8] < offsets[8] + sizes[8],
          "trigram row range");

    float logits[512] = {0};
    for (uint32_t i = 0; i < 512u; ++i) logits[i] = (float)i * 0.01f;
    int indices[10];
    float weights[10];
    q4_router_topk(indices, weights, logits);
    CHECK(indices[0] == 511 && indices[9] == 502, "router top-k order");
    float sum = 0.0f;
    for (uint32_t i = 0; i < 10u; ++i) sum += weights[i];
    CHECK(fabsf(sum - 1.0f) < 1e-6f, "router selected weight normalization");

    unsigned char storage[96] = {0};
    Q38GGUFTensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.n_dims = 3u;
    tensor.shape[0] = 4u;
    tensor.shape[1] = 2u;
    tensor.shape[2] = 3u;
    tensor.nbytes = sizeof(storage);
    tensor.data = storage;
    Q38GGUFTensor view;
    CHECK(q4_tensor_expert_view(&view, &tensor, 2u), "expert tensor view");
    CHECK(view.n_dims == 2u && view.nbytes == 32u &&
          view.data == storage + 64u, "expert plane offset");

    const float input[4] = {3.0f, 4.0f, 0.0f, 2.0f};
    const float gain[4] = {1.0f, 1.0f, 2.0f, 2.0f};
    float normalized[4];
    q4_group_rmsnorm(normalized, input, gain, 2u, 2u, 0.0f);
    CHECK(fabsf(normalized[0] - 0.848528137f) < 1e-6f &&
          fabsf(normalized[1] - 1.13137085f) < 1e-6f,
          "grouped RMS normalization");

    float residual[4] = {1, 2, 3, 4};
    const float branch[2] = {2, -1};
    const float inject[2] = {0, 0};
    q4_hc_combine(residual, branch, inject, 2u, 2u);
    CHECK(residual[0] == 3 && residual[1] == 1 &&
          residual[2] == 5 && residual[3] == 3,
          "zero-injection residual combine");
    puts("qwen4 ops: ok");
    return 0;
}
