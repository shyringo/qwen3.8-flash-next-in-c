#ifndef QWEN4_OPS_H
#define QWEN4_OPS_H

#include "qwen38_gguf.h"

#include <stdint.h>

void q4_ple_indices(int32_t output[16], uint32_t token,
                    uint32_t previous, uint32_t previous2,
                    const uint64_t multipliers[3],
                    const uint64_t offsets[16],
                    const uint64_t sizes[16]);

void q4_router_topk(int indices[10], float weights[10],
                    const float logits[512]);

int q4_tensor_expert_view(Q38GGUFTensor *view,
                          const Q38GGUFTensor *tensor, uint32_t expert);

void q4_group_rmsnorm(float *output, const float *input, const float *gain,
                      uint32_t hidden, uint32_t groups, float epsilon);

void q4_hc_combine(float *residual, const float *branch,
                   const float *inject, uint32_t hidden, uint32_t groups);

#endif
