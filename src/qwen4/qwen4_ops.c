#include "qwen4_ops.h"

#include <math.h>
#include <string.h>

void q4_ple_indices(int32_t output[16], uint32_t token,
                    uint32_t previous, uint32_t previous2,
                    const uint64_t multipliers[3],
                    const uint64_t offsets[16],
                    const uint64_t sizes[16])
{
    const uint64_t bigram = (uint64_t)token * multipliers[0] ^
                            (uint64_t)previous * multipliers[1];
    const uint64_t trigram = bigram ^
                             (uint64_t)previous2 * multipliers[2];
    for (uint32_t head = 0; head < 8u; ++head)
        output[head] = (int32_t)(bigram % sizes[head] + offsets[head]);
    for (uint32_t head = 8u; head < 16u; ++head)
        output[head] = (int32_t)(trigram % sizes[head] + offsets[head]);
}

void q4_router_topk(int indices[10], float weights[10],
                    const float logits[512])
{
    float maximum = logits[0];
    for (uint32_t i = 1; i < 512u; ++i)
        if (logits[i] > maximum) maximum = logits[i];
    float probabilities[512];
    float total = 0.0f;
    for (uint32_t i = 0; i < 512u; ++i) {
        probabilities[i] = expf(logits[i] - maximum);
        total += probabilities[i];
    }
    for (uint32_t i = 0; i < 512u; ++i) probabilities[i] /= total;
    for (uint32_t slot = 0; slot < 10u; ++slot) {
        int best = -1;
        float value = -INFINITY;
        for (uint32_t i = 0; i < 512u; ++i) {
            int used = 0;
            for (uint32_t j = 0; j < slot; ++j)
                if (indices[j] == (int)i) used = 1;
            if (!used && probabilities[i] > value) {
                value = probabilities[i];
                best = (int)i;
            }
        }
        indices[slot] = best;
        weights[slot] = value;
    }
    float selected = 0.0f;
    for (uint32_t i = 0; i < 10u; ++i) selected += weights[i];
    if (selected < 6.103515625e-5f) selected = 6.103515625e-5f;
    for (uint32_t i = 0; i < 10u; ++i) weights[i] /= selected;
}

int q4_tensor_expert_view(Q38GGUFTensor *view,
                          const Q38GGUFTensor *tensor, uint32_t expert)
{
    if (!view || !tensor || tensor->n_dims != 3u ||
        expert >= tensor->shape[2] || tensor->nbytes % tensor->shape[2])
        return 0;
    *view = *tensor;
    view->n_dims = 2u;
    view->nbytes = tensor->nbytes / tensor->shape[2];
    view->data = tensor->data + (uint64_t)expert * view->nbytes;
    view->iq1_s_repack = NULL;
    view->q8_0_repack = NULL;
    view->f32_repack = NULL;
    return 1;
}

void q4_group_rmsnorm(float *output, const float *input, const float *gain,
                      uint32_t hidden, uint32_t groups, float epsilon)
{
    for (uint32_t group = 0; group < groups; ++group) {
        const uint64_t base = (uint64_t)group * hidden;
        double squares = 0.0;
        for (uint32_t i = 0; i < hidden; ++i)
            squares += (double)input[base + i] * input[base + i];
        const float scale = 1.0f /
            sqrtf((float)(squares / hidden) + epsilon);
        for (uint32_t i = 0; i < hidden; ++i)
            output[base + i] = input[base + i] * scale * gain[base + i];
    }
}

void q4_hc_combine(float *residual, const float *branch,
                   const float *inject, uint32_t hidden, uint32_t groups)
{
    for (uint32_t group = 0; group < groups; ++group) {
        const float weight = 2.0f /
            (1.0f + expf(-inject[group] / (float)groups));
        float *stream = residual + (uint64_t)group * hidden;
        for (uint32_t i = 0; i < hidden; ++i)
            stream[i] += branch[i] * weight;
    }
}
