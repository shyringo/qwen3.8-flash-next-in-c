#define _POSIX_C_SOURCE 200809L

#include "qwen4_model.h"

#include "qwen4_gguf.h"
#include "qwen4_ops.h"
#include "qwen38_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define Q4_LAYERS 48u
#define Q4_HIDDEN 2560u
#define Q4_HC 4u
#define Q4_HC_DIM (Q4_HIDDEN * Q4_HC)
#define Q4_HC_RANK 320u
#define Q4_EXPERTS 512u
#define Q4_ACTIVE_EXPERTS 10u
#define Q4_EXPERT_FFN 640u
#define Q4_VOCAB 248320u
#define Q4_LINEAR_QKV 10240u
#define Q4_LINEAR_VALUE 6144u
#define Q4_ATTN_Q_GATE 12288u
#define Q4_ATTN_KV 512u
#define Q4_ATTN_OUT 6144u
#define Q4_LINEAR_HEADS 48u
#define Q4_LINEAR_KEY_HEADS 16u
#define Q4_HEAD_DIM 128u
#define Q4_FULL_LAYERS 12u
#define Q4_QSA_LAYERS Q4_FULL_LAYERS
#define Q4_QSA_RATIO 4u
#define Q4_QSA_HEADS 4u
#define Q4_QSA_DIM 128u
#define Q4_QSA_TOKEN_BUDGET 2048u
#define Q4_QSA_HEAP_BLOCKS 513u
#define Q4_MODEL_CONTEXT 262144u
#define Q4_BATCH_TILE 4u
#define Q4_MOE_UNION_MAX (Q4_BATCH_TILE * Q4_ACTIVE_EXPERTS)
#define Q4_ATTN_HEADS 24u
#define Q4_ATTN_KV_HEADS 2u
#define Q4_ATTN_HEAD_DIM 256u
#define Q4_PLE_HEADS 16u
#define Q4_PLE_HEAD_DIM 160u
#define Q4_PLE_HISTORY 9u
#define Q4_RMS_EPS 1e-6f
#define Q4_EOS 248044u

typedef struct {
    const Q38GGUFTensor *norm;
    const Q38GGUFTensor *down;
    const Q38GGUFTensor *up;
    const Q38GGUFTensor *inject;
} Q4HCWeights;

typedef struct {
    const Q38GGUFTensor *qkv;
    const Q38GGUFTensor *gate;
    const Q38GGUFTensor *alpha;
    const Q38GGUFTensor *beta;
    const Q38GGUFTensor *conv;
    const Q38GGUFTensor *dt;
    const Q38GGUFTensor *a;
    const Q38GGUFTensor *norm;
    const Q38GGUFTensor *out;
    const Q38GGUFTensor *qkv_scale;
    const Q38GGUFTensor *gate_scale;
    const Q38GGUFTensor *alpha_scale;
    const Q38GGUFTensor *beta_scale;
    const Q38GGUFTensor *out_scale;
} Q4LinearWeights;

typedef struct {
    const Q38GGUFTensor *q;
    const Q38GGUFTensor *k;
    const Q38GGUFTensor *v;
    const Q38GGUFTensor *out;
    const Q38GGUFTensor *q_norm;
    const Q38GGUFTensor *k_norm;
    const Q38GGUFTensor *index_q;
    const Q38GGUFTensor *index_k;
    const Q38GGUFTensor *index_q_norm;
    const Q38GGUFTensor *index_k_norm;
    const Q38GGUFTensor *q_scale;
    const Q38GGUFTensor *k_scale;
    const Q38GGUFTensor *v_scale;
    const Q38GGUFTensor *out_scale;
} Q4AttentionWeights;

typedef struct {
    const Q38GGUFTensor *key;
    const Q38GGUFTensor *value;
    const Q38GGUFTensor *norm_key;
    const Q38GGUFTensor *norm_query;
    const Q38GGUFTensor *norm_conv;
    const Q38GGUFTensor *conv;
} Q4PLEWeights;

typedef struct {
    Q4HCWeights attn_hc;
    Q4HCWeights ffn_hc;
    Q4LinearWeights linear;
    Q4AttentionWeights attention;
    Q4PLEWeights ple;
    const Q38GGUFTensor *router;
    const Q38GGUFTensor *expert_gate;
    const Q38GGUFTensor *expert_up;
    const Q38GGUFTensor *expert_down;
    const Q38GGUFTensor *shared_router;
    const Q38GGUFTensor *shared_gate;
    const Q38GGUFTensor *shared_up;
    const Q38GGUFTensor *shared_down;
    const Q38GGUFTensor *expert_gate_scale;
    const Q38GGUFTensor *expert_up_scale;
    const Q38GGUFTensor *expert_down_scale;
    const Q38GGUFTensor *shared_gate_scale;
    const Q38GGUFTensor *shared_up_scale;
    const Q38GGUFTensor *shared_down_scale;
} Q4LayerWeights;

typedef struct {
    float *hc_norm;
    float *hc_low;
    float *hc_gate;
    float *mixed;
    float *inject;
    float *branch;
    float *wide0;
    float *wide1;
    float *beta;
    float *alpha;
    float *attention;
    float *query;
    float *gate;
    float *key;
    float *value;
    float *scores;
    float *index_query;
    float *index_key;
    float *index_scores;
    uint32_t *index_heap;
    uint8_t *selected;
    float *batch_hc_norm;
    float *batch_hc_low;
    float *batch_hc_gate;
    float *batch_mixed;
    float *batch_inject;
    float *batch_wide0;
    float *batch_wide1;
    float *batch_beta;
    float *batch_alpha;
    float *batch_attention;
    float *batch_key;
    float *batch_value;
    float *batch_branch;
    float *batch_router;
    float *batch_moe_gate;
    float *batch_moe_up;
    float *batch_moe_hidden;
    float *batch_moe_output;
    float *batch_routed_output;
    float *batch_shared_output;
    float *batch_logits;
    float *union_gate;
    float *union_up;
    float *union_hidden;
    float *union_output;
    Q38Q8KBlock *union_quantized;
    Q38Q8KBlock *batch_quantized;
    float *router;
    float *expert_gate;
    float *expert_up;
    float *expert_hidden;
    float *expert_output;
    float *shared_output;
    float *routed_gate;
    float *routed_up;
    float *routed_hidden;
    float *routed_output;
    float *ple_embedding;
    float *ple_key;
    float *ple_value;
    float *ple_gated;
    float *ple_norm;
    float *logits;
    Q38Q8KBlock *quantized;
} Q4Scratch;

struct Q4Model {
    Q4GGUFSet gguf;
    uint32_t context_length;
    uint32_t layers;
    uint32_t hidden;
    uint32_t experts;
    uint32_t active_experts;
    const Q38GGUFTensor *embedding;
    const Q38GGUFTensor *ple_embedding;
    Q4HCWeights output_hc;
    const Q38GGUFTensor *output;
    const Q38GGUFTensor *output_scale;
    Q4LayerWeights layer[Q4_LAYERS];
    uint32_t position;
    uint32_t previous[2];
    uint64_t ple_multipliers[3];
    uint64_t ple_offsets[16];
    uint64_t ple_sizes[16];
    float *residual;
    float *conv_state;
    float *delta_state;
    float *key_cache;
    float *value_cache;
    float *ple_history;
    float *index_tail;
    float *index_blocks;
    Q4Scratch scratch;
    double time_ple;
    double time_hc;
    double time_attention;
    double time_gdn_input;
    double time_gdn_state;
    double time_gdn_output;
    double time_full_attention;
    double time_batch_hc;
    double time_batch_attention;
    double time_batch_moe;
    double time_moe;
    double time_head;
};

struct Q4ModelState {
    uint32_t position;
    uint32_t previous[2];
    uint32_t cache_capacity;
    float *residual;
    float *conv_state;
    float *delta_state;
    float *key_cache;
    float *value_cache;
    float *ple_history;
    float *index_tail;
    float *index_blocks;
    float *logits;
};

static const Q38GGUFTensor *weight(const Q4GGUFSet *set, const char *name)
{
    const Q38GGUFTensor *tensor = q4_gguf_find_tensor(set, name);
    if (!tensor) fprintf(stderr, "qwen4: missing tensor %s\n", name);
    return tensor;
}

static const Q38GGUFTensor *optional_weight(const Q4GGUFSet *set,
                                            const char *name)
{
    return q4_gguf_find_tensor(set, name);
}

static const Q38GGUFTensor *layer_weight(const Q4GGUFSet *set,
                                         uint32_t layer, const char *suffix)
{
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return weight(set, name);
}

static int shape(const Q38GGUFTensor *tensor, const char *name,
                 uint32_t dims, uint64_t d0, uint64_t d1, uint64_t d2)
{
    const int ok = tensor && tensor->n_dims == dims && tensor->shape[0] == d0 &&
                   (dims < 2u || tensor->shape[1] == d1) &&
                   (dims < 3u || tensor->shape[2] == d2);
    if (!ok) fprintf(stderr, "qwen4: incompatible tensor %s\n", name);
    return ok;
}

static int bind_hc(Q4HCWeights *hc, const Q4GGUFSet *set, uint32_t layer,
                   const char *prefix)
{
    char suffix[96];
#define HC_BIND(field, tail) do { \
    snprintf(suffix, sizeof(suffix), "%s_%s.weight", prefix, #tail); \
    hc->field = layer_weight(set, layer, suffix); \
} while (0)
    HC_BIND(norm, norm);
    HC_BIND(down, down);
    HC_BIND(up, up);
    HC_BIND(inject, inject);
#undef HC_BIND
    return hc->norm && hc->down && hc->up && hc->inject &&
           shape(hc->norm, suffix, 1, Q4_HC_DIM, 0, 0) &&
           shape(hc->down, suffix, 2, Q4_HC_DIM, Q4_HC_RANK, 0) &&
           shape(hc->up, suffix, 2, Q4_HC_RANK, Q4_HC_DIM, 0) &&
           shape(hc->inject, suffix, 2, Q4_HC_DIM, Q4_HC, 0);
}

static int bind_layer(Q4Model *model, const Q4GGUFSet *set, uint32_t layer)
{
    Q4LayerWeights *w = &model->layer[layer];
    int ok = bind_hc(&w->attn_hc, set, layer, "hc_attn") &&
             bind_hc(&w->ffn_hc, set, layer, "hc_ffn");
    if ((layer + 1u) % 4u == 0u) {
#define ATTN(field, suffix) w->attention.field = layer_weight(set, layer, suffix)
#define ATTN_OPTIONAL(field, suffix) do { char name[128]; \
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix); \
    w->attention.field = optional_weight(set, name); } while (0)
        ATTN(q, "attn_q.weight");
        ATTN(k, "attn_k.weight");
        ATTN(v, "attn_v.weight");
        ATTN(out, "attn_output.weight");
        ATTN(q_norm, "attn_q_norm.weight");
        ATTN(k_norm, "attn_k_norm.weight");
        ATTN(index_q, "indexer.q_proj.weight");
        ATTN(index_k, "indexer.k_proj.weight");
        ATTN(index_q_norm, "indexer.q_norm.weight");
        ATTN(index_k_norm, "indexer.k_norm.weight");
        ATTN_OPTIONAL(q_scale, "attn_q.scale");
        ATTN_OPTIONAL(k_scale, "attn_k.scale");
        ATTN_OPTIONAL(v_scale, "attn_v.scale");
        ATTN_OPTIONAL(out_scale, "attn_output.scale");
#undef ATTN_OPTIONAL
#undef ATTN
        ok &= shape(w->attention.q, "attn_q", 2, Q4_HIDDEN,
                    Q4_ATTN_Q_GATE, 0) &&
              shape(w->attention.k, "attn_k", 2, Q4_HIDDEN, Q4_ATTN_KV, 0) &&
              shape(w->attention.v, "attn_v", 2, Q4_HIDDEN, Q4_ATTN_KV, 0) &&
              shape(w->attention.out, "attn_out", 2, Q4_ATTN_OUT,
                    Q4_HIDDEN, 0) &&
              shape(w->attention.q_norm, "attn_q_norm", 1, 256, 0, 0) &&
              shape(w->attention.k_norm, "attn_k_norm", 1, 256, 0, 0) &&
              shape(w->attention.index_q, "index_q", 2, Q4_HIDDEN, 512, 0) &&
              shape(w->attention.index_k, "index_k", 2, Q4_HIDDEN, 128, 0) &&
              shape(w->attention.index_q_norm, "index_q_norm", 1, 128, 0, 0) &&
              shape(w->attention.index_k_norm, "index_k_norm", 1, 128, 0, 0);
    } else {
#define LINEAR(field, suffix) w->linear.field = layer_weight(set, layer, suffix)
#define LINEAR_OPTIONAL(field, suffix) do { char name[128]; \
    snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix); \
    w->linear.field = optional_weight(set, name); } while (0)
        LINEAR(qkv, "attn_qkv.weight");
        LINEAR(gate, "attn_gate.weight");
        LINEAR(alpha, "ssm_alpha.weight");
        LINEAR(beta, "ssm_beta.weight");
        LINEAR(conv, "ssm_conv1d.weight");
        LINEAR(dt, "ssm_dt.bias");
        LINEAR(a, "ssm_a");
        LINEAR(norm, "ssm_norm.weight");
        LINEAR(out, "ssm_out.weight");
        LINEAR_OPTIONAL(qkv_scale, "attn_qkv.scale");
        LINEAR_OPTIONAL(gate_scale, "attn_gate.scale");
        LINEAR_OPTIONAL(alpha_scale, "ssm_alpha.scale");
        LINEAR_OPTIONAL(beta_scale, "ssm_beta.scale");
        LINEAR_OPTIONAL(out_scale, "ssm_out.scale");
#undef LINEAR_OPTIONAL
#undef LINEAR
        ok &= shape(w->linear.qkv, "attn_qkv", 2, Q4_HIDDEN,
                    Q4_LINEAR_QKV, 0) &&
              shape(w->linear.gate, "attn_gate", 2, Q4_HIDDEN,
                    Q4_LINEAR_VALUE, 0) &&
              shape(w->linear.alpha, "ssm_alpha", 2, Q4_HIDDEN, 48, 0) &&
              shape(w->linear.beta, "ssm_beta", 2, Q4_HIDDEN, 48, 0) &&
              shape(w->linear.conv, "ssm_conv", 2, 4, Q4_LINEAR_QKV, 0) &&
              shape(w->linear.dt, "ssm_dt", 1, 48, 0, 0) &&
              shape(w->linear.a, "ssm_a", 1, 48, 0, 0) &&
              shape(w->linear.norm, "ssm_norm", 1, 128, 0, 0) &&
              shape(w->linear.out, "ssm_out", 2, Q4_LINEAR_VALUE,
                    Q4_HIDDEN, 0);
    }
    w->router = layer_weight(set, layer, "ffn_gate_inp.weight");
    w->expert_gate = layer_weight(set, layer, "ffn_gate_exps.weight");
    w->expert_up = layer_weight(set, layer, "ffn_up_exps.weight");
    w->expert_down = layer_weight(set, layer, "ffn_down_exps.weight");
    w->shared_router = layer_weight(set, layer, "ffn_gate_inp_shexp.weight");
    w->shared_gate = layer_weight(set, layer, "ffn_gate_shexp.weight");
    w->shared_up = layer_weight(set, layer, "ffn_up_shexp.weight");
    w->shared_down = layer_weight(set, layer, "ffn_down_shexp.weight");
    char scale_name[128];
#define OPTIONAL_SCALE(field, suffix) do { \
    snprintf(scale_name, sizeof(scale_name), "blk.%u.%s", layer, suffix); \
    w->field = optional_weight(set, scale_name); } while (0)
    OPTIONAL_SCALE(expert_gate_scale, "ffn_gate_exps.scale");
    OPTIONAL_SCALE(expert_up_scale, "ffn_up_exps.scale");
    OPTIONAL_SCALE(expert_down_scale, "ffn_down_exps.scale");
    OPTIONAL_SCALE(shared_gate_scale, "ffn_gate_shexp.scale");
    OPTIONAL_SCALE(shared_up_scale, "ffn_up_shexp.scale");
    OPTIONAL_SCALE(shared_down_scale, "ffn_down_shexp.scale");
#undef OPTIONAL_SCALE
    ok &= shape(w->router, "router", 2, Q4_HIDDEN, Q4_EXPERTS, 0) &&
          shape(w->expert_gate, "expert_gate", 3, Q4_HIDDEN,
                Q4_EXPERT_FFN, Q4_EXPERTS) &&
          shape(w->expert_up, "expert_up", 3, Q4_HIDDEN,
                Q4_EXPERT_FFN, Q4_EXPERTS) &&
          shape(w->expert_down, "expert_down", 3, Q4_EXPERT_FFN,
                Q4_HIDDEN, Q4_EXPERTS) &&
          shape(w->shared_router, "shared_router", 1, Q4_HIDDEN, 0, 0) &&
          shape(w->shared_gate, "shared_gate", 2, Q4_HIDDEN,
                Q4_EXPERT_FFN, 0) &&
          shape(w->shared_up, "shared_up", 2, Q4_HIDDEN,
                Q4_EXPERT_FFN, 0) &&
          shape(w->shared_down, "shared_down", 2, Q4_EXPERT_FFN,
                Q4_HIDDEN, 0);
    if (layer == 1u) {
#define PLE(field, suffix) w->ple.field = layer_weight(set, layer, suffix)
        PLE(key, "ple_key.weight");
        PLE(value, "ple_value.weight");
        PLE(norm_key, "ple_norm_key.weight");
        PLE(norm_query, "ple_norm_query.weight");
        PLE(norm_conv, "ple_norm_conv.weight");
        PLE(conv, "ple_conv1d.weight");
#undef PLE
        ok &= shape(w->ple.key, "ple_key", 2, Q4_HIDDEN, Q4_HC_DIM, 0) &&
              shape(w->ple.value, "ple_value", 2, Q4_HIDDEN, Q4_HIDDEN, 0) &&
              shape(w->ple.norm_key, "ple_norm_key", 1, Q4_HC_DIM, 0, 0) &&
              shape(w->ple.norm_query, "ple_norm_query", 1, Q4_HC_DIM, 0, 0) &&
              shape(w->ple.norm_conv, "ple_norm_conv", 1, Q4_HC_DIM, 0, 0) &&
              shape(w->ple.conv, "ple_conv", 2, 4, Q4_HC_DIM, 0);
    }
    return ok;
}

static int q4_q8_type(uint32_t type)
{
    return type == Q38_GGML_Q2_K || type == Q38_GGML_Q3_K ||
           type == Q38_GGML_Q4_K || type == Q38_GGML_Q5_K ||
           type == Q38_GGML_Q6_K || type == Q38_GGML_IQ2_XXS ||
           type == Q38_GGML_IQ2_XS || type == Q38_GGML_IQ3_XXS ||
           type == Q38_GGML_IQ1_S || type == Q38_GGML_IQ3_S ||
           type == Q38_GGML_IQ2_S || type == Q38_GGML_IQ4_XS ||
           type == Q38_GGML_IQ4_NL || type == Q38_GGML_IQ1_M;
}

static int project(Q4Model *model, float *output, const float *input,
                   uint32_t length, const Q38GGUFTensor *tensor)
{
    if (q4_q8_type(tensor->type) && length % 256u == 0u) {
        if (!q38_quantize_q8_k(model->scratch.quantized, input, length)) return 0;
        return q38_tensor_gemv_q8_k(output, model->scratch.quantized,
                                    length, tensor);
    }
    return q38_tensor_gemv_f32(output, input, tensor);
}

static int project_prequantized(float *output, const float *input,
                                const Q38Q8KBlock *quantized,
                                uint32_t length,
                                const Q38GGUFTensor *tensor)
{
    if (q4_q8_type(tensor->type) && length % 256u == 0u)
        return q38_tensor_gemv_q8_k(output, quantized, length, tensor);
    return q38_tensor_gemv_f32(output, input, tensor);
}

static int project_batch(Q4Model *model, float *output, const float *input,
                         uint32_t batch_size, uint32_t length,
                         const Q38GGUFTensor *tensor)
{
    if (q4_q8_type(tensor->type) && length % 256u == 0u) {
        if (!q38_quantize_q8_k(model->scratch.batch_quantized, input,
                               (uint64_t)batch_size * length)) return 0;
        return q38_tensor_gemm_q8_k(output, model->scratch.batch_quantized,
                                    batch_size, length, tensor);
    }
    return q38_tensor_gemm_f32(output, input, batch_size, length, tensor);
}

static int project_batch_prequantized(float *output, const float *input,
                                      const Q38Q8KBlock *quantized,
                                      uint32_t batch_size, uint32_t length,
                                      const Q38GGUFTensor *tensor)
{
    if (q4_q8_type(tensor->type) && length % 256u == 0u)
        return q38_tensor_gemm_q8_k(output, quantized, batch_size,
                                    length, tensor);
    return q38_tensor_gemm_f32(output, input, batch_size, length, tensor);
}

static float scalar(const Q38GGUFTensor *tensor, uint64_t index)
{
    float value = 0.0f;
    if (tensor && tensor->type == Q38_GGML_F32 && index < tensor->shape[0])
        memcpy(&value, tensor->data + index * sizeof(value), sizeof(value));
    return value;
}

static int dot_vector(float *output, const float *input,
                      const Q38GGUFTensor *tensor, uint32_t length)
{
    if (!output || !input || !tensor || tensor->n_dims != 1u ||
        tensor->shape[0] != length || tensor->type != Q38_GGML_F32)
        return 0;
    const float *weights = (const float *)tensor->data;
    float sum = 0.0f;
    for (uint32_t i = 0; i < length; ++i)
        sum = fmaf(weights[i], input[i], sum);
    *output = sum;
    return 1;
}

static void trace_vector(const char *name, uint32_t layer,
                         const float *values, uint32_t count)
{
    if (!getenv("Q4_TRACE") || (layer != 0u && layer != UINT32_MAX)) return;
    double sum = 0.0, sum_abs = 0.0, sum_sq = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += values[i]; sum_abs += fabs(values[i]);
        sum_sq += (double)values[i] * values[i];
    }
    fprintf(stderr, "TRACE %s-%u n=%u first=%a sum=%a abs=%a sq=%a\n",
            name, layer, count, (double)values[0], sum, sum_abs, sum_sq);
}

static int finite_vector(const char *name, const float *values, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            fprintf(stderr, "qwen4: non-finite %s at %u\n", name, i);
            return 0;
        }
    }
    return 1;
}

static void apply_scale(float *values, uint32_t length,
                        const Q38GGUFTensor *scale, uint32_t index)
{
    if (!scale) return;
    const float factor = scalar(scale, index);
    for (uint32_t i = 0; i < length; ++i) values[i] *= factor;
}

static void prefetch_tensor(const Q38GGUFTensor *tensor)
{
    if (!tensor || !tensor->data || !tensor->nbytes) return;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;
    const uintptr_t mask = (uintptr_t)page_size - 1u;
    const uintptr_t start = (uintptr_t)tensor->data & ~mask;
    const uintptr_t end = ((uintptr_t)tensor->data + tensor->nbytes + mask) & ~mask;
    (void)posix_madvise((void *)start, end - start, POSIX_MADV_WILLNEED);
}

static float sigmoidf_local(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float softplusf_local(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

static void rmsnorm(float *output, const float *input,
                    const Q38GGUFTensor *gain, uint32_t length)
{
    double squares = 0.0;
    for (uint32_t i = 0; i < length; ++i)
        squares += (double)input[i] * input[i];
    const float scale = 1.0f /
        sqrtf((float)(squares / length) + Q4_RMS_EPS);
    for (uint32_t i = 0; i < length; ++i)
        output[i] = input[i] * scale * scalar(gain, i);
}

static void l2norm(float *values, uint32_t length)
{
    double squares = 0.0;
    for (uint32_t i = 0; i < length; ++i)
        squares += (double)values[i] * values[i];
    const float scale = 1.0f / fmaxf(sqrtf((float)squares), Q4_RMS_EPS);
    for (uint32_t i = 0; i < length; ++i) values[i] *= scale;
}

static float dot128(const float *left, const float *right)
{
    float result = 0.0f;
    for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i)
        result = fmaf(left[i], right[i], result);
    return result;
}

static float decay_dot128(float *state, float decay, const float *right)
{
    float result = 0.0f;
    for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
        state[i] *= decay;
        result = fmaf(state[i], right[i], result);
    }
    return result;
}

static float update_dot128(float *state, const float *input, float scale,
                           const float *right)
{
    float result = 0.0f;
    for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
        state[i] = fmaf(input[i], scale, state[i]);
        result = fmaf(state[i], right[i], result);
    }
    return result;
}

#if defined(__AVX2__) && defined(__FMA__)
static void delta_columns8(float *state, const float *key, const float *value,
                           float beta, const float *query, float decay,
                           float query_scale, float *output)
{
    const __m256 decay8 = _mm256_set1_ps(decay);
    __m256 prediction = _mm256_setzero_ps();
    for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
        __m256 current = _mm256_loadu_ps(
            state + (uint64_t)i * Q4_HEAD_DIM);
        current = _mm256_mul_ps(current, decay8);
        _mm256_storeu_ps(state + (uint64_t)i * Q4_HEAD_DIM, current);
        prediction = _mm256_fmadd_ps(
            current, _mm256_set1_ps(key[i]), prediction);
    }
    const __m256 change = _mm256_mul_ps(
        _mm256_sub_ps(_mm256_loadu_ps(value), prediction),
        _mm256_set1_ps(beta));
    __m256 result = _mm256_setzero_ps();
    for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
        __m256 current = _mm256_loadu_ps(
            state + (uint64_t)i * Q4_HEAD_DIM);
        current = _mm256_fmadd_ps(
            _mm256_set1_ps(key[i]), change, current);
        _mm256_storeu_ps(state + (uint64_t)i * Q4_HEAD_DIM, current);
        result = _mm256_fmadd_ps(
            current, _mm256_set1_ps(query[i]), result);
    }
    _mm256_storeu_ps(output,
        _mm256_mul_ps(result, _mm256_set1_ps(query_scale)));
}
#endif

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static void rope_stride(float *values, uint32_t heads, uint32_t stride,
                        uint32_t position)
{
    const float theta_scale = powf(10000000.0f, -2.0f / 64.0f);
    for (uint32_t head = 0; head < heads; ++head) {
        float *vector = values + (uint64_t)head * stride;
        float theta = (float)position;
        for (uint32_t pair = 0; pair < 32u; ++pair) {
            const float c = cosf(theta), s = sinf(theta);
            const float x0 = vector[pair], x1 = vector[pair + 32u];
            vector[pair] = x0 * c - x1 * s;
            vector[pair + 32u] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
    }
}

static void rope(float *values, uint32_t heads, uint32_t position)
{
    rope_stride(values, heads, Q4_ATTN_HEAD_DIM, position);
}

static int hc_mix(Q4Model *model, const Q4HCWeights *hc,
                  const float *residual, float *mixed, float *inject)
{
    Q4Scratch *s = &model->scratch;
    q4_group_rmsnorm(s->hc_norm, residual,
                     (const float *)hc->norm->data,
                     Q4_HIDDEN, Q4_HC, Q4_RMS_EPS);
    if (!finite_vector("HC norm", s->hc_norm, Q4_HC_DIM)) return 0;
    if (!project(model, s->hc_low, s->hc_norm, Q4_HC_DIM, hc->down)) return 0;
    if (!finite_vector("HC down", s->hc_low, Q4_HC_RANK)) return 0;
    for (uint32_t i = 0; i < Q4_HC_RANK; ++i) {
        const float x = s->hc_low[i] / (float)Q4_HC;
        s->hc_low[i] = x * sigmoidf_local(x);
    }
    if (!finite_vector("HC SiLU", s->hc_low, Q4_HC_RANK)) return 0;
    if (!project(model, s->hc_gate, s->hc_low, Q4_HC_RANK, hc->up)) return 0;
    if (!finite_vector("HC up", s->hc_gate, Q4_HC_DIM)) return 0;
    memset(mixed, 0, Q4_HIDDEN * sizeof(float));
    for (uint32_t group = 0; group < Q4_HC; ++group) {
        const uint64_t base = (uint64_t)group * Q4_HIDDEN;
        for (uint32_t i = 0; i < Q4_HIDDEN; ++i) {
            const float gated = s->hc_norm[base + i] *
                                sigmoidf_local(s->hc_gate[base + i]);
            mixed[i] += gated / (float)Q4_HC;
        }
    }
    return !inject || project(model, inject, s->hc_norm, Q4_HC_DIM, hc->inject);
}

static int hc_mix_batch(Q4Model *model, const Q4HCWeights *hc,
                        float *const *residual, uint32_t batch_size)
{
    Q4Scratch *s = &model->scratch;
    for (uint32_t token = 0; token < batch_size; ++token)
        q4_group_rmsnorm(s->batch_hc_norm + (uint64_t)token * Q4_HC_DIM,
                         residual[token], (const float *)hc->norm->data,
                         Q4_HIDDEN, Q4_HC, Q4_RMS_EPS);
    if (!project_batch(model, s->batch_hc_low, s->batch_hc_norm,
                       batch_size, Q4_HC_DIM, hc->down)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *low = s->batch_hc_low + (uint64_t)token * Q4_HC_RANK;
        for (uint32_t i = 0; i < Q4_HC_RANK; ++i) {
            const float x = low[i] / (float)Q4_HC;
            low[i] = x * sigmoidf_local(x);
        }
    }
    if (!project_batch(model, s->batch_hc_gate, s->batch_hc_low,
                       batch_size, Q4_HC_RANK, hc->up)) return 0;
    if (hc->inject &&
        !q38_tensor_gemm_f32(s->batch_inject, s->batch_hc_norm,
                             batch_size, Q4_HC_DIM, hc->inject)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *mixed = s->batch_mixed + (uint64_t)token * Q4_HIDDEN;
        const float *norm = s->batch_hc_norm +
                            (uint64_t)token * Q4_HC_DIM;
        const float *gate = s->batch_hc_gate +
                            (uint64_t)token * Q4_HC_DIM;
        memset(mixed, 0, Q4_HIDDEN * sizeof(float));
        for (uint32_t group = 0; group < Q4_HC; ++group) {
            const uint64_t base = (uint64_t)group * Q4_HIDDEN;
            for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
                mixed[i] += norm[base + i] *
                    sigmoidf_local(gate[base + i]) / (float)Q4_HC;
        }
    }
    return 1;
}

static void linear_attention_state(Q4Model *model, uint32_t layer,
                                   float *wide0, float *wide1,
                                   float *beta, float *alpha,
                                   float *attention)
{
    const Q4LinearWeights *w = &model->layer[layer].linear;
    const uint32_t recurrent = layer - (layer + 1u) / 4u;
    float *conv_state = model->conv_state +
        (uint64_t)recurrent * Q4_LINEAR_QKV * 3u;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t channel = 0; channel < Q4_LINEAR_QKV; ++channel) {
        float *state = conv_state + (uint64_t)channel * 3u;
        const float *kernel = (const float *)(w->conv->data +
                              (uint64_t)channel * 4u * sizeof(float));
        float value = state[0] * kernel[0] + state[1] * kernel[1] +
                      state[2] * kernel[2] + wide0[channel] * kernel[3];
        state[0] = state[1]; state[1] = state[2]; state[2] = wide0[channel];
        wide0[channel] = value * sigmoidf_local(value);
    }
    float *query = wide0;
    float *key = query + Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM;
    float *value = key + Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM;
    for (uint32_t head = 0; head < Q4_LINEAR_KEY_HEADS; ++head) {
        l2norm(query + (uint64_t)head * Q4_HEAD_DIM, Q4_HEAD_DIM);
        l2norm(key + (uint64_t)head * Q4_HEAD_DIM, Q4_HEAD_DIM);
    }
    trace_vector("q_conv_predelta", layer, query,
                 Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM);
    trace_vector("k_conv_predelta", layer, key,
                 Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM);
    trace_vector("v_conv_predelta", layer, value,
                 Q4_LINEAR_HEADS * Q4_HEAD_DIM);
    for (uint32_t head = 0; head < Q4_LINEAR_HEADS; ++head) {
        beta[head] = sigmoidf_local(beta[head]);
        alpha[head] = scalar(w->a, head) *
            softplusf_local(alpha[head] + scalar(w->dt, head));
    }
    trace_vector("gate", layer, alpha, Q4_LINEAR_HEADS);
    trace_vector("beta_sigmoid", layer, beta, Q4_LINEAR_HEADS);
    float *delta = model->delta_state + (uint64_t)recurrent *
        Q4_LINEAR_HEADS * Q4_HEAD_DIM * Q4_HEAD_DIM;
    const float query_scale = 1.0f / sqrtf((float)Q4_HEAD_DIM);
    const int fuse_decay = !getenv("Q4_DISABLE_DECAY_FUSION");
    const int column8 = !getenv("Q4_DISABLE_DELTA_COLUMN8");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t vh = 0; vh < Q4_LINEAR_HEADS; ++vh) {
        const uint32_t kh = vh % Q4_LINEAR_KEY_HEADS;
        const float *q = query + (uint64_t)kh * Q4_HEAD_DIM;
        const float *k = key + (uint64_t)kh * Q4_HEAD_DIM;
        const float *v = value + (uint64_t)vh * Q4_HEAD_DIM;
        float *state = delta + (uint64_t)vh * Q4_HEAD_DIM * Q4_HEAD_DIM;
        float *head_out = attention + (uint64_t)vh * Q4_HEAD_DIM;
        const float decay = expf(alpha[vh]);
#if defined(__AVX2__) && defined(__FMA__)
        if (column8) {
            for (uint32_t column = 0; column < Q4_HEAD_DIM; column += 8u)
                delta_columns8(state + column, k, v + column, beta[vh], q,
                               decay, query_scale, head_out + column);
            continue;
        }
        for (uint32_t column = 0; column < Q4_HEAD_DIM; ++column) {
            float prediction = 0.0f;
            for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
                float *cell = state + (uint64_t)i * Q4_HEAD_DIM + column;
                *cell *= decay;
                prediction = fmaf(*cell, k[i], prediction);
            }
            const float change = (v[column] - prediction) * beta[vh];
            float result = 0.0f;
            for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
                float *cell = state + (uint64_t)i * Q4_HEAD_DIM + column;
                *cell = fmaf(k[i], change, *cell);
                result = fmaf(*cell, q[i], result);
            }
            head_out[column] = result * query_scale;
        }
        continue;
#else
        (void)column8;
#endif
        for (uint32_t column = 0; column < Q4_HEAD_DIM; ++column) {
            float *state_column = state + (uint64_t)column * Q4_HEAD_DIM;
            float prediction;
            if (!fuse_decay) {
                for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i)
                    state_column[i] *= decay;
                prediction = dot128(state_column, k);
            } else {
                prediction = decay_dot128(state_column, decay, k);
            }
            const float change = (v[column] - prediction) * beta[vh];
            head_out[column] = update_dot128(state_column, k, change, q) *
                               query_scale;
        }
    }
    for (uint32_t head = 0; head < Q4_LINEAR_HEADS; ++head) {
        float *head_out = attention + (uint64_t)head * Q4_HEAD_DIM;
        rmsnorm(head_out, head_out, w->norm, Q4_HEAD_DIM);
        const float *output_gate = wide1 + (uint64_t)head * Q4_HEAD_DIM;
        for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i)
            head_out[i] *= sigmoidf_local(output_gate[i]);
    }
    trace_vector("final_output", layer, attention, Q4_LINEAR_VALUE);
}

static void linear_attention_state_batch(Q4Model *model, uint32_t layer,
                                         float *wide0, float *wide1,
                                         float *beta, float *alpha,
                                         float *attention,
                                         uint32_t batch_size)
{
    const Q4LinearWeights *w = &model->layer[layer].linear;
    const uint32_t recurrent = layer - (layer + 1u) / 4u;
    float *conv_state = model->conv_state +
        (uint64_t)recurrent * Q4_LINEAR_QKV * 3u;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t channel = 0; channel < Q4_LINEAR_QKV; ++channel) {
        float *state = conv_state + (uint64_t)channel * 3u;
        const float *kernel = (const float *)(w->conv->data +
                              (uint64_t)channel * 4u * sizeof(float));
        for (uint32_t token = 0; token < batch_size; ++token) {
            float *token_wide = wide0 +
                                (uint64_t)token * Q4_LINEAR_QKV;
            const float input = token_wide[channel];
            const float value = state[0] * kernel[0] +
                                state[1] * kernel[1] +
                                state[2] * kernel[2] + input * kernel[3];
            state[0] = state[1];
            state[1] = state[2];
            state[2] = input;
            token_wide[channel] = value * sigmoidf_local(value);
        }
    }
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *token_wide = wide0 + (uint64_t)token * Q4_LINEAR_QKV;
        float *query = token_wide;
        float *key = query + Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM;
        float *token_beta = beta + (uint64_t)token * Q4_LINEAR_HEADS;
        float *token_alpha = alpha + (uint64_t)token * Q4_LINEAR_HEADS;
        for (uint32_t head = 0; head < Q4_LINEAR_KEY_HEADS; ++head) {
            l2norm(query + (uint64_t)head * Q4_HEAD_DIM, Q4_HEAD_DIM);
            l2norm(key + (uint64_t)head * Q4_HEAD_DIM, Q4_HEAD_DIM);
        }
        for (uint32_t head = 0; head < Q4_LINEAR_HEADS; ++head) {
            token_beta[head] = sigmoidf_local(token_beta[head]);
            token_alpha[head] = scalar(w->a, head) *
                softplusf_local(token_alpha[head] + scalar(w->dt, head));
        }
    }
    float *delta = model->delta_state + (uint64_t)recurrent *
        Q4_LINEAR_HEADS * Q4_HEAD_DIM * Q4_HEAD_DIM;
    const float query_scale = 1.0f / sqrtf((float)Q4_HEAD_DIM);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t vh = 0; vh < Q4_LINEAR_HEADS; ++vh) {
        float *state = delta + (uint64_t)vh * Q4_HEAD_DIM * Q4_HEAD_DIM;
        const uint32_t kh = vh % Q4_LINEAR_KEY_HEADS;
        for (uint32_t token = 0; token < batch_size; ++token) {
            float *token_wide = wide0 +
                                (uint64_t)token * Q4_LINEAR_QKV;
            const float *query = token_wide +
                (uint64_t)kh * Q4_HEAD_DIM;
            const float *key = token_wide +
                Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM +
                (uint64_t)kh * Q4_HEAD_DIM;
            const float *value = token_wide +
                2u * Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM +
                (uint64_t)vh * Q4_HEAD_DIM;
            float *head_out = attention +
                ((uint64_t)token * Q4_LINEAR_HEADS + vh) * Q4_HEAD_DIM;
            const float token_beta = beta[
                (uint64_t)token * Q4_LINEAR_HEADS + vh];
            const float decay = expf(alpha[
                (uint64_t)token * Q4_LINEAR_HEADS + vh]);
#if defined(__AVX2__) && defined(__FMA__)
            for (uint32_t column = 0; column < Q4_HEAD_DIM; column += 8u)
                delta_columns8(state + column, key, value + column,
                               token_beta, query, decay, query_scale,
                               head_out + column);
#else
            for (uint32_t column = 0; column < Q4_HEAD_DIM; ++column) {
                float *state_column = state +
                                      (uint64_t)column * Q4_HEAD_DIM;
                const float prediction = decay_dot128(
                    state_column, decay, key);
                const float change = (value[column] - prediction) *
                                     token_beta;
                head_out[column] = update_dot128(
                    state_column, key, change, query) * query_scale;
            }
#endif
        }
    }
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *token_attention = attention +
            (uint64_t)token * Q4_LINEAR_VALUE;
        const float *token_gate = wide1 +
            (uint64_t)token * Q4_LINEAR_VALUE;
        for (uint32_t head = 0; head < Q4_LINEAR_HEADS; ++head) {
            float *head_out = token_attention +
                              (uint64_t)head * Q4_HEAD_DIM;
            rmsnorm(head_out, head_out, w->norm, Q4_HEAD_DIM);
            const float *output_gate = token_gate +
                (uint64_t)head * Q4_HEAD_DIM;
            for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i)
                head_out[i] *= sigmoidf_local(output_gate[i]);
        }
    }
}

static int linear_attention(Q4Model *model, uint32_t layer,
                            const float *input, float *output)
{
    Q4Scratch *s = &model->scratch;
    const Q4LinearWeights *w = &model->layer[layer].linear;
    const int profiling = getenv("Q4_PROFILE") != NULL;
    double segment_started = profiling ? now_seconds() : 0.0;
    if (!q38_quantize_q8_k(s->quantized, input, Q4_HIDDEN) ||
        !project_prequantized(s->wide0, input, s->quantized,
                              Q4_HIDDEN, w->qkv) ||
        !project_prequantized(s->wide1, input, s->quantized,
                              Q4_HIDDEN, w->gate) ||
        !project_prequantized(s->beta, input, s->quantized,
                              Q4_HIDDEN, w->beta) ||
        !project_prequantized(s->alpha, input, s->quantized,
                              Q4_HIDDEN, w->alpha)) return 0;
    apply_scale(s->wide0, Q4_LINEAR_QKV, w->qkv_scale, 0u);
    apply_scale(s->wide1, Q4_LINEAR_VALUE, w->gate_scale, 0u);
    apply_scale(s->beta, Q4_LINEAR_HEADS, w->beta_scale, 0u);
    apply_scale(s->alpha, Q4_LINEAR_HEADS, w->alpha_scale, 0u);
    if (profiling) {
        model->time_gdn_input += now_seconds() - segment_started;
        segment_started = now_seconds();
    }
    const uint32_t recurrent = layer - (layer + 1u) / 4u;
    float *conv_state = model->conv_state +
        (uint64_t)recurrent * Q4_LINEAR_QKV * 3u;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t channel = 0; channel < Q4_LINEAR_QKV; ++channel) {
        float *state = conv_state + (uint64_t)channel * 3u;
        const float *kernel = (const float *)(w->conv->data +
                              (uint64_t)channel * 4u * sizeof(float));
        float value = state[0] * kernel[0] + state[1] * kernel[1] +
                      state[2] * kernel[2] + s->wide0[channel] * kernel[3];
        state[0] = state[1]; state[1] = state[2]; state[2] = s->wide0[channel];
        s->wide0[channel] = value * sigmoidf_local(value);
    }
    float *query = s->wide0;
    float *key = query + Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM;
    float *value = key + Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM;
    for (uint32_t head = 0; head < Q4_LINEAR_KEY_HEADS; ++head) {
        l2norm(query + (uint64_t)head * Q4_HEAD_DIM, Q4_HEAD_DIM);
        l2norm(key + (uint64_t)head * Q4_HEAD_DIM, Q4_HEAD_DIM);
    }
    trace_vector("q_conv_predelta", layer, query,
                 Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM);
    trace_vector("k_conv_predelta", layer, key,
                 Q4_LINEAR_KEY_HEADS * Q4_HEAD_DIM);
    trace_vector("v_conv_predelta", layer, value,
                 Q4_LINEAR_HEADS * Q4_HEAD_DIM);
    for (uint32_t head = 0; head < Q4_LINEAR_HEADS; ++head) {
        s->beta[head] = sigmoidf_local(s->beta[head]);
        s->alpha[head] = scalar(w->a, head) *
            softplusf_local(s->alpha[head] + scalar(w->dt, head));
    }
    trace_vector("gate", layer, s->alpha, Q4_LINEAR_HEADS);
    trace_vector("beta_sigmoid", layer, s->beta, Q4_LINEAR_HEADS);
    float *delta = model->delta_state + (uint64_t)recurrent *
        Q4_LINEAR_HEADS * Q4_HEAD_DIM * Q4_HEAD_DIM;
    const float query_scale = 1.0f / sqrtf((float)Q4_HEAD_DIM);
    const int fuse_decay = !getenv("Q4_DISABLE_DECAY_FUSION");
    const int column8 = !getenv("Q4_DISABLE_DELTA_COLUMN8");
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t vh = 0; vh < Q4_LINEAR_HEADS; ++vh) {
        const uint32_t kh = vh % Q4_LINEAR_KEY_HEADS;
        const float *q = query + (uint64_t)kh * Q4_HEAD_DIM;
        const float *k = key + (uint64_t)kh * Q4_HEAD_DIM;
        const float *v = value + (uint64_t)vh * Q4_HEAD_DIM;
        float *state = delta + (uint64_t)vh * Q4_HEAD_DIM * Q4_HEAD_DIM;
        float *head_out = s->attention + (uint64_t)vh * Q4_HEAD_DIM;
        const float decay = expf(s->alpha[vh]);
#if defined(__AVX2__) && defined(__FMA__)
        if (column8) {
            for (uint32_t column = 0; column < Q4_HEAD_DIM; column += 8u)
                delta_columns8(state + column,
                               k, v + column, s->beta[vh], q, decay,
                               query_scale, head_out + column);
            continue;
        }
        for (uint32_t column = 0; column < Q4_HEAD_DIM; ++column) {
            float prediction = 0.0f;
            for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
                float *cell = state + (uint64_t)i * Q4_HEAD_DIM + column;
                *cell *= decay;
                prediction = fmaf(*cell, k[i], prediction);
            }
            const float change = (v[column] - prediction) * s->beta[vh];
            float result = 0.0f;
            for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i) {
                float *cell = state + (uint64_t)i * Q4_HEAD_DIM + column;
                *cell = fmaf(k[i], change, *cell);
                result = fmaf(*cell, q[i], result);
            }
            head_out[column] = result * query_scale;
        }
        continue;
#else
        (void)column8;
#endif
        for (uint32_t column = 0; column < Q4_HEAD_DIM; ++column) {
            float *state_column = state + (uint64_t)column * Q4_HEAD_DIM;
            float prediction;
            if (!fuse_decay) {
                for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i)
                    state_column[i] *= decay;
                prediction = dot128(state_column, k);
            } else {
                prediction = decay_dot128(state_column, decay, k);
            }
            const float change = (v[column] - prediction) * s->beta[vh];
            head_out[column] = update_dot128(state_column, k, change, q) *
                               query_scale;
        }
    }
    for (uint32_t head = 0; head < Q4_LINEAR_HEADS; ++head) {
        float *head_out = s->attention + (uint64_t)head * Q4_HEAD_DIM;
        rmsnorm(head_out, head_out, w->norm, Q4_HEAD_DIM);
        const float *output_gate = s->wide1 + (uint64_t)head * Q4_HEAD_DIM;
        for (uint32_t i = 0; i < Q4_HEAD_DIM; ++i)
            head_out[i] *= sigmoidf_local(output_gate[i]);
    }
    trace_vector("final_output", layer, s->attention, Q4_LINEAR_VALUE);
    if (profiling) {
        model->time_gdn_state += now_seconds() - segment_started;
        segment_started = now_seconds();
    }
    if (!project(model, output, s->attention, Q4_LINEAR_VALUE, w->out)) return 0;
    apply_scale(output, Q4_HIDDEN, w->out_scale, 0u);
    if (profiling) model->time_gdn_output += now_seconds() - segment_started;
    trace_vector("linear_attn_out", layer, output, Q4_HIDDEN);
    return 1;
}

static int linear_attention_batch(Q4Model *model, uint32_t layer,
                                  const float *input, float *output,
                                  uint32_t batch_size)
{
    Q4Scratch *s = &model->scratch;
    const Q4LinearWeights *w = &model->layer[layer].linear;
    if (!q38_quantize_q8_k(s->batch_quantized, input,
                           (uint64_t)batch_size * Q4_HIDDEN) ||
        !project_batch_prequantized(s->batch_wide0, input,
            s->batch_quantized, batch_size, Q4_HIDDEN, w->qkv) ||
        !project_batch_prequantized(s->batch_wide1, input,
            s->batch_quantized, batch_size, Q4_HIDDEN, w->gate) ||
        !project_batch_prequantized(s->batch_beta, input,
            s->batch_quantized, batch_size, Q4_HIDDEN, w->beta) ||
        !project_batch_prequantized(s->batch_alpha, input,
            s->batch_quantized, batch_size, Q4_HIDDEN, w->alpha)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *wide0 = s->batch_wide0 +
                       (uint64_t)token * Q4_LINEAR_QKV;
        float *wide1 = s->batch_wide1 +
                       (uint64_t)token * Q4_LINEAR_VALUE;
        float *beta = s->batch_beta +
                      (uint64_t)token * Q4_LINEAR_HEADS;
        float *alpha = s->batch_alpha +
                       (uint64_t)token * Q4_LINEAR_HEADS;
        float *attention = s->batch_attention +
                           (uint64_t)token * Q4_LINEAR_VALUE;
        apply_scale(wide0, Q4_LINEAR_QKV, w->qkv_scale, 0u);
        apply_scale(wide1, Q4_LINEAR_VALUE, w->gate_scale, 0u);
        apply_scale(beta, Q4_LINEAR_HEADS, w->beta_scale, 0u);
        apply_scale(alpha, Q4_LINEAR_HEADS, w->alpha_scale, 0u);
        if (getenv("Q4_DISABLE_GDN_BATCH_STATE"))
            linear_attention_state(model, layer, wide0, wide1,
                                   beta, alpha, attention);
    }
    if (!getenv("Q4_DISABLE_GDN_BATCH_STATE"))
        linear_attention_state_batch(model, layer, s->batch_wide0,
            s->batch_wide1, s->batch_beta, s->batch_alpha,
            s->batch_attention, batch_size);
    if (!project_batch(model, output, s->batch_attention, batch_size,
                       Q4_LINEAR_VALUE, w->out)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token)
        apply_scale(output + (uint64_t)token * Q4_HIDDEN,
                    Q4_HIDDEN, w->out_scale, 0u);
    return 1;
}

static int qsa_heap_worse(const float *scores, uint32_t left, uint32_t right)
{
    return scores[left] < scores[right] ||
           (scores[left] == scores[right] && left > right);
}

static void qsa_heap_up(uint32_t *heap, uint32_t size, const float *scores)
{
    uint32_t child = size - 1u;
    while (child) {
        const uint32_t parent = (child - 1u) / 2u;
        if (!qsa_heap_worse(scores, heap[child], heap[parent])) break;
        const uint32_t swap = heap[parent];
        heap[parent] = heap[child]; heap[child] = swap;
        child = parent;
    }
}

static void qsa_heap_down(uint32_t *heap, uint32_t size,
                          const float *scores)
{
    uint32_t parent = 0u;
    for (;;) {
        uint32_t child = parent * 2u + 1u;
        if (child >= size) break;
        if (child + 1u < size &&
            qsa_heap_worse(scores, heap[child + 1u], heap[child])) ++child;
        if (!qsa_heap_worse(scores, heap[child], heap[parent])) break;
        const uint32_t swap = heap[parent];
        heap[parent] = heap[child]; heap[child] = swap;
        parent = child;
    }
}

static int qsa_select(Q4Model *model, uint32_t layer, uint32_t position,
                      const float *input, const Q38Q8KBlock *quantized)
{
    Q4Scratch *s = &model->scratch;
    const Q4AttentionWeights *w = &model->layer[layer].attention;
    const uint32_t full = layer / 4u;
    if (!project_prequantized(s->index_key, input, quantized,
                              Q4_HIDDEN, w->index_k)) return 0;
    float *tail = model->index_tail +
        ((uint64_t)full * Q4_QSA_RATIO + position % Q4_QSA_RATIO) *
        Q4_QSA_DIM;
    memcpy(tail, s->index_key, Q4_QSA_DIM * sizeof(float));

    const uint32_t block_capacity =
        (model->context_length + Q4_QSA_RATIO - 1u) / Q4_QSA_RATIO;
    if (position % Q4_QSA_RATIO == Q4_QSA_RATIO - 1u) {
        float *pooled = s->index_key;
        const float *rows = model->index_tail +
            (uint64_t)full * Q4_QSA_RATIO * Q4_QSA_DIM;
        for (uint32_t i = 0; i < Q4_QSA_DIM; ++i) {
            float sum = 0.0f;
            for (uint32_t row = 0; row < Q4_QSA_RATIO; ++row)
                sum += rows[(uint64_t)row * Q4_QSA_DIM + i];
            pooled[i] = sum / (float)Q4_QSA_RATIO;
        }
        rmsnorm(pooled, pooled, w->index_k_norm, Q4_QSA_DIM);
        rope_stride(pooled, 1u, Q4_QSA_DIM,
                    position + 1u - Q4_QSA_RATIO);
        float *target = model->index_blocks +
            ((uint64_t)full * block_capacity + position / Q4_QSA_RATIO) *
            Q4_QSA_DIM;
        memcpy(target, pooled, Q4_QSA_DIM * sizeof(float));
    }

    if (position < 2048u) return 1;
    if (!project_prequantized(s->index_query, input, quantized,
                              Q4_HIDDEN, w->index_q)) return 0;
    for (uint32_t head = 0; head < Q4_QSA_HEADS; ++head)
        rmsnorm(s->index_query + (uint64_t)head * Q4_QSA_DIM,
                s->index_query + (uint64_t)head * Q4_QSA_DIM,
                w->index_q_norm, Q4_QSA_DIM);
    rope_stride(s->index_query, Q4_QSA_HEADS, Q4_QSA_DIM, position);

    const uint32_t completed = (position + 1u) / Q4_QSA_RATIO;
    for (uint32_t block = 0; block < completed; ++block) {
        const float *key = model->index_blocks +
            ((uint64_t)full * block_capacity + block) * Q4_QSA_DIM;
        float score = 0.0f;
        for (uint32_t head = 0; head < Q4_QSA_HEADS; ++head) {
            const float *query = s->index_query +
                                 (uint64_t)head * Q4_QSA_DIM;
            const float dot = dot128(query, key);
            if (dot > 0.0f) score += dot;
        }
        s->index_scores[block] = score;
    }

    memset(s->selected, 0, (size_t)(position + 1u));
    const uint32_t tail_start = completed * Q4_QSA_RATIO;
    const uint32_t tail_count = position + 1u - tail_start;
    const uint32_t width = position + 1u <
                           Q4_QSA_TOKEN_BUDGET + Q4_QSA_RATIO - 1u
                         ? position + 1u
                         : Q4_QSA_TOKEN_BUDGET + Q4_QSA_RATIO - 1u;
    const uint32_t full_tokens = width - tail_count;
    const uint32_t keep = (full_tokens + Q4_QSA_RATIO - 1u) /
                          Q4_QSA_RATIO;
    for (uint32_t block = 0; block < completed; ++block) {
        if (block < keep) {
            s->index_heap[block] = block;
            qsa_heap_up(s->index_heap, block + 1u, s->index_scores);
        } else if (qsa_heap_worse(s->index_scores,
                                  s->index_heap[0], block)) {
            s->index_heap[0] = block;
            qsa_heap_down(s->index_heap, keep, s->index_scores);
        }
    }
    const uint32_t partial = full_tokens % Q4_QSA_RATIO
                           ? s->index_heap[0] : UINT32_MAX;
    for (uint32_t slot = 0; slot < keep; ++slot) {
        const uint32_t base = s->index_heap[slot] * Q4_QSA_RATIO;
        const uint32_t count = s->index_heap[slot] == partial
                             ? full_tokens % Q4_QSA_RATIO : Q4_QSA_RATIO;
        memset(s->selected + base, 1, count);
    }
    if (tail_start <= position)
        memset(s->selected + tail_start, 1,
               (size_t)(position + 1u - tail_start));
    if (getenv("Q4_TRACE_QSA")) {
        uint32_t selected = 0u;
        for (uint32_t pos = 0; pos <= position; ++pos)
            selected += s->selected[pos] != 0u;
        fprintf(stderr,
                "QSA layer=%u position=%u blocks=%u selected=%u tail=%u\n",
                layer, position, keep, selected, tail_count);
    }
    return 1;
}

static int full_attention(Q4Model *model, uint32_t layer, uint32_t position,
                          const float *input, float *output)
{
    const double profile_started = getenv("Q4_PROFILE") ? now_seconds() : 0.0;
    Q4Scratch *s = &model->scratch;
    const Q4AttentionWeights *w = &model->layer[layer].attention;
    if (!q38_quantize_q8_k(s->quantized, input, Q4_HIDDEN) ||
        !project_prequantized(s->wide0, input, s->quantized,
                              Q4_HIDDEN, w->q) ||
        !project_prequantized(s->key, input, s->quantized,
                              Q4_HIDDEN, w->k) ||
        !project_prequantized(s->value, input, s->quantized,
                              Q4_HIDDEN, w->v)) return 0;
    apply_scale(s->wide0, Q4_ATTN_Q_GATE, w->q_scale, 0u);
    apply_scale(s->key, Q4_ATTN_KV, w->k_scale, 0u);
    apply_scale(s->value, Q4_ATTN_KV, w->v_scale, 0u);
    if (model->context_length > 2048u &&
        !qsa_select(model, layer, position, input, s->quantized)) return 0;
    for (uint32_t head = 0; head < Q4_ATTN_HEADS; ++head) {
        memcpy(s->query + (uint64_t)head * Q4_ATTN_HEAD_DIM,
               s->wide0 + (uint64_t)head * 2u * Q4_ATTN_HEAD_DIM,
               Q4_ATTN_HEAD_DIM * sizeof(float));
        memcpy(s->gate + (uint64_t)head * Q4_ATTN_HEAD_DIM,
               s->wide0 + (uint64_t)head * 2u * Q4_ATTN_HEAD_DIM +
               Q4_ATTN_HEAD_DIM, Q4_ATTN_HEAD_DIM * sizeof(float));
        rmsnorm(s->query + (uint64_t)head * Q4_ATTN_HEAD_DIM,
                s->query + (uint64_t)head * Q4_ATTN_HEAD_DIM,
                w->q_norm, Q4_ATTN_HEAD_DIM);
    }
    for (uint32_t head = 0; head < Q4_ATTN_KV_HEADS; ++head)
        rmsnorm(s->key + (uint64_t)head * Q4_ATTN_HEAD_DIM,
                s->key + (uint64_t)head * Q4_ATTN_HEAD_DIM,
                w->k_norm, Q4_ATTN_HEAD_DIM);
    rope(s->query, Q4_ATTN_HEADS, position);
    rope(s->key, Q4_ATTN_KV_HEADS, position);
    const uint32_t full = layer / 4u;
    const uint64_t cache = ((uint64_t)full * model->context_length +
                            position) * Q4_ATTN_KV;
    memcpy(model->key_cache + cache, s->key, Q4_ATTN_KV * sizeof(float));
    memcpy(model->value_cache + cache, s->value, Q4_ATTN_KV * sizeof(float));
    const int sparse = model->context_length > 2048u && position >= 2048u;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t head = 0; head < Q4_ATTN_HEADS; ++head) {
        const uint32_t kv = head / (Q4_ATTN_HEADS / Q4_ATTN_KV_HEADS);
        float *scores = s->scores + (uint64_t)head * model->context_length;
        float maximum = -INFINITY;
        const float *q = s->query + (uint64_t)head * Q4_ATTN_HEAD_DIM;
        for (uint32_t pos = 0; pos <= position; ++pos) {
            if (sparse && !s->selected[pos]) continue;
            const uint64_t row = ((uint64_t)full * model->context_length + pos) *
                                 Q4_ATTN_KV + (uint64_t)kv * Q4_ATTN_HEAD_DIM;
            float score = 0.0f;
            for (uint32_t i = 0; i < Q4_ATTN_HEAD_DIM; ++i)
                score = fmaf(q[i], model->key_cache[row + i], score);
            score /= sqrtf((float)Q4_ATTN_HEAD_DIM);
            scores[pos] = score;
            if (score > maximum) maximum = score;
        }
        float denominator = 0.0f;
        for (uint32_t pos = 0; pos <= position; ++pos) {
            if (sparse && !s->selected[pos]) continue;
            scores[pos] = expf(scores[pos] - maximum);
            denominator += scores[pos];
        }
        float *head_out = s->attention + (uint64_t)head * Q4_ATTN_HEAD_DIM;
        memset(head_out, 0, Q4_ATTN_HEAD_DIM * sizeof(float));
        for (uint32_t pos = 0; pos <= position; ++pos) {
            if (sparse && !s->selected[pos]) continue;
            const uint64_t row = ((uint64_t)full * model->context_length + pos) *
                                 Q4_ATTN_KV + (uint64_t)kv * Q4_ATTN_HEAD_DIM;
            const float probability = scores[pos] / denominator;
            for (uint32_t i = 0; i < Q4_ATTN_HEAD_DIM; ++i)
                head_out[i] = fmaf(probability,
                                   model->value_cache[row + i], head_out[i]);
        }
        const float *gate = s->gate + (uint64_t)head * Q4_ATTN_HEAD_DIM;
        for (uint32_t i = 0; i < Q4_ATTN_HEAD_DIM; ++i)
            head_out[i] *= sigmoidf_local(gate[i]);
    }
    if (!project(model, output, s->attention, Q4_ATTN_OUT, w->out)) return 0;
    apply_scale(output, Q4_HIDDEN, w->out_scale, 0u);
    if (getenv("Q4_PROFILE"))
        model->time_full_attention += now_seconds() - profile_started;
    return 1;
}

static int full_attention_batch(Q4Model *model, uint32_t layer,
                                uint32_t base_position, const float *input,
                                float *output, uint32_t batch_size)
{
    Q4Scratch *s = &model->scratch;
    const Q4AttentionWeights *w = &model->layer[layer].attention;
    if (model->context_length > 2048u || batch_size != Q4_BATCH_TILE) {
        for (uint32_t token = 0; token < batch_size; ++token)
            if (!full_attention(model, layer, base_position + token,
                                input + (uint64_t)token * Q4_HIDDEN,
                                output + (uint64_t)token * Q4_HIDDEN)) return 0;
        return 1;
    }
    if (!q38_quantize_q8_k(s->batch_quantized, input,
                           (uint64_t)batch_size * Q4_HIDDEN) ||
        !project_batch_prequantized(s->batch_wide0, input,
                                    s->batch_quantized, batch_size,
                                    Q4_HIDDEN, w->q) ||
        !project_batch_prequantized(s->batch_key, input,
                                    s->batch_quantized, batch_size,
                                    Q4_HIDDEN, w->k) ||
        !project_batch_prequantized(s->batch_value, input,
                                    s->batch_quantized, batch_size,
                                    Q4_HIDDEN, w->v)) return 0;
    const uint32_t full = layer / 4u;
    for (uint32_t token = 0; token < batch_size; ++token) {
        const uint32_t position = base_position + token;
        float *wide = s->batch_wide0 +
                      (uint64_t)token * Q4_ATTN_Q_GATE;
        float *key = s->batch_key + (uint64_t)token * Q4_ATTN_KV;
        float *value = s->batch_value + (uint64_t)token * Q4_ATTN_KV;
        apply_scale(wide, Q4_ATTN_Q_GATE, w->q_scale, 0u);
        apply_scale(key, Q4_ATTN_KV, w->k_scale, 0u);
        apply_scale(value, Q4_ATTN_KV, w->v_scale, 0u);
        for (uint32_t head = 0; head < Q4_ATTN_HEADS; ++head)
            rmsnorm(wide + (uint64_t)head * 2u * Q4_ATTN_HEAD_DIM,
                    wide + (uint64_t)head * 2u * Q4_ATTN_HEAD_DIM,
                    w->q_norm, Q4_ATTN_HEAD_DIM);
        for (uint32_t head = 0; head < Q4_ATTN_KV_HEADS; ++head)
            rmsnorm(key + (uint64_t)head * Q4_ATTN_HEAD_DIM,
                    key + (uint64_t)head * Q4_ATTN_HEAD_DIM,
                    w->k_norm, Q4_ATTN_HEAD_DIM);
        rope_stride(wide, Q4_ATTN_HEADS,
                    2u * Q4_ATTN_HEAD_DIM, position);
        rope(key, Q4_ATTN_KV_HEADS, position);
        const uint64_t cache =
            ((uint64_t)full * model->context_length + position) * Q4_ATTN_KV;
        memcpy(model->key_cache + cache, key, Q4_ATTN_KV * sizeof(float));
        memcpy(model->value_cache + cache, value, Q4_ATTN_KV * sizeof(float));
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t head = 0; head < Q4_ATTN_HEADS; ++head) {
        const uint32_t kv = head /
            (Q4_ATTN_HEADS / Q4_ATTN_KV_HEADS);
        float *scores = s->scores + (uint64_t)head * model->context_length;
        for (uint32_t token = 0; token < batch_size; ++token) {
            const uint32_t position = base_position + token;
            const float *wide = s->batch_wide0 +
                                (uint64_t)token * Q4_ATTN_Q_GATE;
            const float *query = wide +
                (uint64_t)head * 2u * Q4_ATTN_HEAD_DIM;
            float maximum = -INFINITY;
            for (uint32_t pos = 0; pos <= position; ++pos) {
                const uint64_t row =
                    ((uint64_t)full * model->context_length + pos) *
                    Q4_ATTN_KV + (uint64_t)kv * Q4_ATTN_HEAD_DIM;
                float score = 0.0f;
                for (uint32_t i = 0; i < Q4_ATTN_HEAD_DIM; ++i)
                    score = fmaf(query[i], model->key_cache[row + i], score);
                score /= sqrtf((float)Q4_ATTN_HEAD_DIM);
                scores[pos] = score;
                if (score > maximum) maximum = score;
            }
            float denominator = 0.0f;
            for (uint32_t pos = 0; pos <= position; ++pos) {
                scores[pos] = expf(scores[pos] - maximum);
                denominator += scores[pos];
            }
            float *head_out = s->batch_attention +
                ((uint64_t)token * Q4_ATTN_HEADS + head) * Q4_ATTN_HEAD_DIM;
            memset(head_out, 0, Q4_ATTN_HEAD_DIM * sizeof(float));
            for (uint32_t pos = 0; pos <= position; ++pos) {
                const uint64_t row =
                    ((uint64_t)full * model->context_length + pos) *
                    Q4_ATTN_KV + (uint64_t)kv * Q4_ATTN_HEAD_DIM;
                const float probability = scores[pos] / denominator;
                for (uint32_t i = 0; i < Q4_ATTN_HEAD_DIM; ++i)
                    head_out[i] = fmaf(probability,
                        model->value_cache[row + i], head_out[i]);
            }
            const float *gate = wide +
                (uint64_t)head * 2u * Q4_ATTN_HEAD_DIM + Q4_ATTN_HEAD_DIM;
            for (uint32_t i = 0; i < Q4_ATTN_HEAD_DIM; ++i)
                head_out[i] *= sigmoidf_local(gate[i]);
        }
    }
    if (!project_batch(model, output, s->batch_attention, batch_size,
                       Q4_ATTN_OUT, w->out)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token)
        apply_scale(output + (uint64_t)token * Q4_HIDDEN,
                    Q4_HIDDEN, w->out_scale, 0u);
    return 1;
}

static int moe(Q4Model *model, uint32_t layer,
               const float *input, float *output)
{
    Q4Scratch *s = &model->scratch;
    const Q4LayerWeights *w = &model->layer[layer];
    if (!project(model, s->router, input, Q4_HIDDEN, w->router)) return 0;
    int experts[10];
    float weights[10];
    q4_router_topk(experts, weights, s->router);
    if (getenv("Q4_TRACE_EXPERTS")) {
        fprintf(stderr, "ROUTE token=%u layer=%u", model->position, layer);
        for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot)
            fprintf(stderr, " %d", experts[slot]);
        fputc('\n', stderr);
    }
    Q38GGUFTensor expert_gate[Q4_ACTIVE_EXPERTS];
    Q38GGUFTensor expert_up[Q4_ACTIVE_EXPERTS];
    Q38GGUFTensor expert_down[Q4_ACTIVE_EXPERTS];
    const int prefetch = getenv("Q4_PREFETCH") != NULL;
    for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot) {
        if (!q4_tensor_expert_view(&expert_gate[slot], w->expert_gate,
                                   experts[slot]) ||
            !q4_tensor_expert_view(&expert_up[slot], w->expert_up,
                                   experts[slot]) ||
            !q4_tensor_expert_view(&expert_down[slot], w->expert_down,
                                   experts[slot])) {
            return 0;
        }
        if (prefetch) {
            prefetch_tensor(&expert_gate[slot]);
            prefetch_tensor(&expert_up[slot]);
            prefetch_tensor(&expert_down[slot]);
        }
    }
    if (!q38_quantize_q8_k(s->quantized, input, Q4_HIDDEN)) return 0;
    int success[Q4_ACTIVE_EXPERTS + 1u] = {0};
    double task_time[Q4_ACTIVE_EXPERTS + 1u] = {0};
    int expert_threads = (int)(Q4_ACTIVE_EXPERTS + 1u);
    const char *thread_env = getenv("Q4_EXPERT_THREADS");
    if (thread_env && *thread_env) {
        const int requested = atoi(thread_env);
        if (requested >= 1 && requested <= (int)(Q4_ACTIVE_EXPERTS + 1u))
            expert_threads = requested;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(expert_threads)
#else
    (void)expert_threads;
#endif
    for (uint32_t task = 0; task <= Q4_ACTIVE_EXPERTS; ++task) {
        const double task_started = getenv("Q4_PROFILE_TASK") ? now_seconds() : 0.0;
        if (task == Q4_ACTIVE_EXPERTS) {
            float *gate_out = s->expert_gate;
            float *up_out = s->expert_up;
            float *hidden = s->expert_hidden;
            success[task] = project_prequantized(
                gate_out, input, s->quantized, Q4_HIDDEN, w->shared_gate) &&
                project_prequantized(
                up_out, input, s->quantized, Q4_HIDDEN, w->shared_up);
            if (!success[task]) continue;
            apply_scale(gate_out, Q4_EXPERT_FFN, w->shared_gate_scale, 0u);
            apply_scale(up_out, Q4_EXPERT_FFN, w->shared_up_scale, 0u);
            for (uint32_t i = 0; i < Q4_EXPERT_FFN; ++i)
                hidden[i] = gate_out[i] * sigmoidf_local(gate_out[i]) * up_out[i];
            success[task] = project_prequantized(
                s->shared_output, hidden, NULL, Q4_EXPERT_FFN, w->shared_down);
            if (success[task]) apply_scale(s->shared_output, Q4_HIDDEN,
                                           w->shared_down_scale, 0u);
            if (getenv("Q4_PROFILE_TASK"))
                task_time[task] = now_seconds() - task_started;
            continue;
        }
        float *gate_out = s->routed_gate + (uint64_t)task * Q4_EXPERT_FFN;
        float *up_out = s->routed_up + (uint64_t)task * Q4_EXPERT_FFN;
        float *hidden = s->routed_hidden + (uint64_t)task * Q4_EXPERT_FFN;
        float *expert_out = s->routed_output + (uint64_t)task * Q4_HIDDEN;
        success[task] = project_prequantized(
            gate_out, input, s->quantized, Q4_HIDDEN, &expert_gate[task]) &&
            project_prequantized(
            up_out, input, s->quantized, Q4_HIDDEN, &expert_up[task]);
        if (!success[task]) continue;
        apply_scale(gate_out, Q4_EXPERT_FFN, w->expert_gate_scale,
                    (uint32_t)experts[task]);
        apply_scale(up_out, Q4_EXPERT_FFN, w->expert_up_scale,
                    (uint32_t)experts[task]);
        for (uint32_t i = 0; i < Q4_EXPERT_FFN; ++i)
            hidden[i] = gate_out[i] * sigmoidf_local(gate_out[i]) * up_out[i];
        success[task] = project_prequantized(
            expert_out, hidden, NULL, Q4_EXPERT_FFN, &expert_down[task]);
        if (success[task]) apply_scale(expert_out, Q4_HIDDEN,
                                       w->expert_down_scale,
                                       (uint32_t)experts[task]);
        if (getenv("Q4_PROFILE_TASK"))
            task_time[task] = now_seconds() - task_started;
    }
    for (uint32_t task = 0; task <= Q4_ACTIVE_EXPERTS; ++task) {
        if (!success[task]) return 0;
    }
    if (getenv("Q4_PROFILE_TASK") && layer == 0u) {
        fprintf(stderr, "MOE_TASK L0");
        for (uint32_t task = 0; task <= Q4_ACTIVE_EXPERTS; ++task)
            fprintf(stderr, " %s%u=%.6f", task == Q4_ACTIVE_EXPERTS ? "S" : "E",
                    task == Q4_ACTIVE_EXPERTS ? 0u : (uint32_t)experts[task],
                    task_time[task]);
        fputc('\n', stderr);
    }
    memset(output, 0, Q4_HIDDEN * sizeof(float));
    for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot) {
        const float *expert_out = s->routed_output + (uint64_t)slot * Q4_HIDDEN;
        for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
            output[i] = fmaf(weights[slot], expert_out[i], output[i]);
    }
    float shared_gate = 0.0f;
    if (!dot_vector(&shared_gate, input, w->shared_router, Q4_HIDDEN))
        return 0;
    shared_gate = sigmoidf_local(shared_gate);
    for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
        output[i] = fmaf(shared_gate, s->shared_output[i], output[i]);
    return 1;
}

static int moe_batch_expert(Q4Model *model, const Q4LayerWeights *w,
                            int experts[Q4_BATCH_TILE][Q4_ACTIVE_EXPERTS],
                            uint32_t batch_size, uint32_t expert,
                            uint32_t task)
{
    Q4Scratch *s = &model->scratch;
    uint32_t occurrence_token[Q4_BATCH_TILE];
    uint32_t occurrence_slot[Q4_BATCH_TILE];
    uint32_t count = 0u;
    const uint64_t input_blocks = Q4_HIDDEN / 256u;
    Q38Q8KBlock *compact_quantized = s->union_quantized +
        (uint64_t)task * Q4_BATCH_TILE * input_blocks;
    for (uint32_t token = 0; token < batch_size; ++token) {
        for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot) {
            if ((uint32_t)experts[token][slot] != expert) continue;
            occurrence_token[count] = token;
            occurrence_slot[count] = slot;
            memcpy(compact_quantized + (uint64_t)count * input_blocks,
                   s->batch_quantized + (uint64_t)token * input_blocks,
                   input_blocks * sizeof(*compact_quantized));
            ++count;
            break;
        }
    }
    if (!count) return 0;
    float *gate = s->union_gate +
        (uint64_t)task * Q4_BATCH_TILE * Q4_EXPERT_FFN;
    float *up = s->union_up +
        (uint64_t)task * Q4_BATCH_TILE * Q4_EXPERT_FFN;
    float *hidden = s->union_hidden +
        (uint64_t)task * Q4_BATCH_TILE * Q4_EXPERT_FFN;
    float *expert_output = s->union_output +
        (uint64_t)task * Q4_BATCH_TILE * Q4_HIDDEN;
    Q38GGUFTensor gate_view, up_view, down_view;
    if (!q4_tensor_expert_view(&gate_view, w->expert_gate, (int)expert) ||
        !q4_tensor_expert_view(&up_view, w->expert_up, (int)expert) ||
        !q4_tensor_expert_view(&down_view, w->expert_down, (int)expert) ||
        !q38_tensor_gemm_q8_k(gate, compact_quantized,
                              count, Q4_HIDDEN, &gate_view) ||
        !q38_tensor_gemm_q8_k(up, compact_quantized,
                              count, Q4_HIDDEN, &up_view)) return 0;
    for (uint32_t occurrence = 0; occurrence < count; ++occurrence) {
        float *gate_row = gate + (uint64_t)occurrence * Q4_EXPERT_FFN;
        float *up_row = up + (uint64_t)occurrence * Q4_EXPERT_FFN;
        float *hidden_row = hidden + (uint64_t)occurrence * Q4_EXPERT_FFN;
        apply_scale(gate_row, Q4_EXPERT_FFN, w->expert_gate_scale, expert);
        apply_scale(up_row, Q4_EXPERT_FFN, w->expert_up_scale, expert);
        for (uint32_t i = 0; i < Q4_EXPERT_FFN; ++i)
            hidden_row[i] = gate_row[i] * sigmoidf_local(gate_row[i]) *
                            up_row[i];
    }
    if (count >= 4u) {
        if (!q38_tensor_gemm_f32(expert_output, hidden, count,
                                 Q4_EXPERT_FFN, &down_view)) return 0;
    } else {
        for (uint32_t occurrence = 0; occurrence < count; ++occurrence)
            if (!q38_tensor_gemv_f32(
                    expert_output + (uint64_t)occurrence * Q4_HIDDEN,
                    hidden + (uint64_t)occurrence * Q4_EXPERT_FFN,
                    &down_view)) return 0;
    }
    for (uint32_t occurrence = 0; occurrence < count; ++occurrence) {
        float *output_row = expert_output +
                            (uint64_t)occurrence * Q4_HIDDEN;
        apply_scale(output_row, Q4_HIDDEN, w->expert_down_scale, expert);
        memcpy(s->batch_routed_output +
                ((uint64_t)occurrence_token[occurrence] *
                 Q4_ACTIVE_EXPERTS + occurrence_slot[occurrence]) * Q4_HIDDEN,
               output_row, Q4_HIDDEN * sizeof(float));
    }
    return 1;
}

static int moe_batch_shared(Q4Model *model, const Q4LayerWeights *w,
                            const float *input, uint32_t batch_size)
{
    Q4Scratch *s = &model->scratch;
    if (!project_batch(model, s->batch_moe_gate, input, batch_size,
                       Q4_HIDDEN, w->shared_gate) ||
        !project_batch(model, s->batch_moe_up, input, batch_size,
                       Q4_HIDDEN, w->shared_up)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *gate = s->batch_moe_gate +
                      (uint64_t)token * Q4_EXPERT_FFN;
        float *up = s->batch_moe_up +
                    (uint64_t)token * Q4_EXPERT_FFN;
        float *hidden = s->batch_moe_hidden +
                        (uint64_t)token * Q4_EXPERT_FFN;
        apply_scale(gate, Q4_EXPERT_FFN, w->shared_gate_scale, 0u);
        apply_scale(up, Q4_EXPERT_FFN, w->shared_up_scale, 0u);
        for (uint32_t i = 0; i < Q4_EXPERT_FFN; ++i)
            hidden[i] = gate[i] * sigmoidf_local(gate[i]) * up[i];
    }
    if (!project_batch(model, s->batch_shared_output, s->batch_moe_hidden,
                       batch_size, Q4_EXPERT_FFN, w->shared_down)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token)
        apply_scale(s->batch_shared_output + (uint64_t)token * Q4_HIDDEN,
                    Q4_HIDDEN, w->shared_down_scale, 0u);
    return 1;
}

static int moe_batch(Q4Model *model, uint32_t layer, const float *input,
                     float *output, uint32_t batch_size)
{
    Q4Scratch *s = &model->scratch;
    const Q4LayerWeights *w = &model->layer[layer];
    int experts[Q4_BATCH_TILE][Q4_ACTIVE_EXPERTS];
    float route_weights[Q4_BATCH_TILE][Q4_ACTIVE_EXPERTS];
    if (!project_batch(model, s->batch_router, input, batch_size,
                       Q4_HIDDEN, w->router) ||
        !q38_quantize_q8_k(s->batch_quantized, input,
                           (uint64_t)batch_size * Q4_HIDDEN)) return 0;
    for (uint32_t token = 0; token < batch_size; ++token)
        q4_router_topk(experts[token], route_weights[token],
                       s->batch_router + (uint64_t)token * Q4_EXPERTS);

    uint32_t union_experts[Q4_MOE_UNION_MAX];
    uint32_t union_count = 0u;
    for (uint32_t token = 0; token < batch_size; ++token) {
        for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot) {
            const uint32_t expert = (uint32_t)experts[token][slot];
            uint32_t found = 0u;
            while (found < union_count && union_experts[found] != expert)
                ++found;
            if (found == union_count) union_experts[union_count++] = expert;
        }
    }
    if (!getenv("Q4_BATCH_MOE_SERIAL")) {
        int success[Q4_MOE_UNION_MAX + 1u] = {0};
        int expert_threads = 11;
        const char *thread_env = getenv("Q4_EXPERT_THREADS");
        if (thread_env && *thread_env) {
            const int requested = atoi(thread_env);
            if (requested >= 1 && requested <= 11) expert_threads = requested;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(expert_threads)
#else
        (void)expert_threads;
#endif
        for (uint32_t task = 0; task <= union_count; ++task)
            success[task] = task == union_count
                ? moe_batch_shared(model, w, input, batch_size)
                : moe_batch_expert(model, w, experts, batch_size,
                                   union_experts[task], task);
        for (uint32_t task = 0; task <= union_count; ++task)
            if (!success[task]) return 0;
        goto routed_ready;
    }

    if (!moe_batch_shared(model, w, input, batch_size)) return 0;

    const uint64_t input_blocks = Q4_HIDDEN / 256u;
    Q38Q8KBlock compact_quantized[Q4_BATCH_TILE * (Q4_HIDDEN / 256u)];
    for (uint32_t expert = 0; expert < Q4_EXPERTS; ++expert) {
        uint32_t occurrence_token[Q4_BATCH_TILE];
        uint32_t occurrence_slot[Q4_BATCH_TILE];
        uint32_t count = 0u;
        for (uint32_t token = 0; token < batch_size; ++token) {
            for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot) {
                if ((uint32_t)experts[token][slot] != expert) continue;
                occurrence_token[count] = token;
                occurrence_slot[count] = slot;
                memcpy(compact_quantized + (uint64_t)count * input_blocks,
                       s->batch_quantized + (uint64_t)token * input_blocks,
                       input_blocks * sizeof(*compact_quantized));
                ++count;
                break;
            }
        }
        if (!count) continue;
        Q38GGUFTensor gate_view, up_view, down_view;
        if (!q4_tensor_expert_view(&gate_view, w->expert_gate, (int)expert) ||
            !q4_tensor_expert_view(&up_view, w->expert_up, (int)expert) ||
            !q4_tensor_expert_view(&down_view, w->expert_down, (int)expert) ||
            !q38_tensor_gemm_q8_k(s->batch_moe_gate, compact_quantized,
                                  count, Q4_HIDDEN, &gate_view) ||
            !q38_tensor_gemm_q8_k(s->batch_moe_up, compact_quantized,
                                  count, Q4_HIDDEN, &up_view)) return 0;
        for (uint32_t occurrence = 0; occurrence < count; ++occurrence) {
            float *gate = s->batch_moe_gate +
                          (uint64_t)occurrence * Q4_EXPERT_FFN;
            float *up = s->batch_moe_up +
                        (uint64_t)occurrence * Q4_EXPERT_FFN;
            float *hidden = s->batch_moe_hidden +
                            (uint64_t)occurrence * Q4_EXPERT_FFN;
            apply_scale(gate, Q4_EXPERT_FFN, w->expert_gate_scale, expert);
            apply_scale(up, Q4_EXPERT_FFN, w->expert_up_scale, expert);
            for (uint32_t i = 0; i < Q4_EXPERT_FFN; ++i)
                hidden[i] = gate[i] * sigmoidf_local(gate[i]) * up[i];
        }
        if (count >= 4u) {
            if (!q38_tensor_gemm_f32(s->batch_moe_output,
                                     s->batch_moe_hidden, count,
                                     Q4_EXPERT_FFN, &down_view)) return 0;
        } else {
            for (uint32_t occurrence = 0; occurrence < count; ++occurrence)
                if (!q38_tensor_gemv_f32(
                        s->batch_moe_output +
                            (uint64_t)occurrence * Q4_HIDDEN,
                        s->batch_moe_hidden +
                            (uint64_t)occurrence * Q4_EXPERT_FFN,
                        &down_view)) return 0;
        }
        for (uint32_t occurrence = 0; occurrence < count; ++occurrence) {
            float *expert_output = s->batch_moe_output +
                                   (uint64_t)occurrence * Q4_HIDDEN;
            apply_scale(expert_output, Q4_HIDDEN,
                        w->expert_down_scale, expert);
            memcpy(s->batch_routed_output +
                    ((uint64_t)occurrence_token[occurrence] *
                     Q4_ACTIVE_EXPERTS + occurrence_slot[occurrence]) *
                    Q4_HIDDEN,
                   expert_output, Q4_HIDDEN * sizeof(float));
        }
    }

routed_ready:
    for (uint32_t token = 0; token < batch_size; ++token) {
        float *token_output = output + (uint64_t)token * Q4_HIDDEN;
        memset(token_output, 0, Q4_HIDDEN * sizeof(float));
        for (uint32_t slot = 0; slot < Q4_ACTIVE_EXPERTS; ++slot) {
            const float *expert_output = s->batch_routed_output +
                ((uint64_t)token * Q4_ACTIVE_EXPERTS + slot) * Q4_HIDDEN;
            for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
                token_output[i] = fmaf(route_weights[token][slot],
                                       expert_output[i], token_output[i]);
        }
        float shared_gate = 0.0f;
        if (!dot_vector(&shared_gate,
                        input + (uint64_t)token * Q4_HIDDEN,
                        w->shared_router, Q4_HIDDEN)) return 0;
        shared_gate = sigmoidf_local(shared_gate);
        const float *shared_output = s->batch_shared_output +
                                     (uint64_t)token * Q4_HIDDEN;
        for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
            token_output[i] = fmaf(shared_gate, shared_output[i],
                                   token_output[i]);
    }
    return 1;
}

static int ple(Q4Model *model, uint32_t token, float *residual)
{
    Q4Scratch *s = &model->scratch;
    const Q4PLEWeights *w = &model->layer[1].ple;
    int32_t rows[Q4_PLE_HEADS];
    q4_ple_indices(rows, token, model->previous[0], model->previous[1],
                   model->ple_multipliers, model->ple_offsets,
                   model->ple_sizes);
    for (uint32_t head = 0; head < Q4_PLE_HEADS; ++head) {
        if (!q38_tensor_row_f32(s->ple_embedding +
                (uint64_t)head * Q4_PLE_HEAD_DIM,
                model->ple_embedding, (uint32_t)rows[head])) return 0;
    }
    if (!project(model, s->ple_key, s->ple_embedding, Q4_HIDDEN, w->key) ||
        !project(model, s->ple_value, s->ple_embedding, Q4_HIDDEN, w->value))
        return 0;
    q4_group_rmsnorm(s->ple_key, s->ple_key, (const float *)w->norm_key->data,
                     Q4_HIDDEN, Q4_HC, Q4_RMS_EPS);
    q4_group_rmsnorm(s->hc_norm, residual,
                     (const float *)w->norm_query->data,
                     Q4_HIDDEN, Q4_HC, Q4_RMS_EPS);
    for (uint32_t group = 0; group < Q4_HC; ++group) {
        float score = 0.0f;
        const uint64_t base = (uint64_t)group * Q4_HIDDEN;
        for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
            score = fmaf(s->ple_key[base + i], s->hc_norm[base + i], score);
        score /= sqrtf((float)Q4_HIDDEN);
        const float signed_root = copysignf(
            sqrtf(fmaxf(fabsf(score), 1e-6f)), score);
        const float gate = sigmoidf_local(signed_root);
        for (uint32_t i = 0; i < Q4_HIDDEN; ++i)
            s->ple_gated[base + i] = s->ple_value[i] * gate;
    }
    q4_group_rmsnorm(s->ple_norm, s->ple_gated,
                     (const float *)w->norm_conv->data,
                     Q4_HIDDEN, Q4_HC, Q4_RMS_EPS);
    for (uint32_t channel = 0; channel < Q4_HC_DIM; ++channel) {
        const float *kernel = (const float *)(w->conv->data +
            (uint64_t)channel * 4u * sizeof(float));
        float value = s->ple_norm[channel] * kernel[3];
        value += model->ple_history[channel] * kernel[0];
        value += model->ple_history[(uint64_t)3u * Q4_HC_DIM + channel] *
                 kernel[1];
        value += model->ple_history[(uint64_t)6u * Q4_HC_DIM + channel] *
                 kernel[2];
        const float conv = value * sigmoidf_local(value);
        residual[channel] += s->ple_gated[channel] + conv;
    }
    memmove(model->ple_history,
            model->ple_history + Q4_HC_DIM,
            (uint64_t)(Q4_PLE_HISTORY - 1u) * Q4_HC_DIM * sizeof(float));
    memcpy(model->ple_history +
           (uint64_t)(Q4_PLE_HISTORY - 1u) * Q4_HC_DIM,
           s->ple_norm, Q4_HC_DIM * sizeof(float));
    model->previous[1] = model->previous[0];
    model->previous[0] = token;
    return 1;
}

static int layer_forward(Q4Model *model, uint32_t layer,
                         float *residual, uint32_t position)
{
    Q4Scratch *s = &model->scratch;
    const int profiling = getenv("Q4_PROFILE") != NULL;
    double started = profiling ? now_seconds() : 0.0;
    if (!hc_mix(model, &model->layer[layer].attn_hc, residual,
                s->mixed, s->inject)) return 0;
    if (profiling) {
        model->time_hc += now_seconds() - started;
        started = now_seconds();
    }
    const int ok = ((layer + 1u) % 4u == 0u)
        ? full_attention(model, layer, position, s->mixed, s->branch)
        : linear_attention(model, layer, s->mixed, s->branch);
    if (!ok) return 0;
    if (profiling) {
        model->time_attention += now_seconds() - started;
        started = now_seconds();
    }
    trace_vector("hc_mixed_attn", layer, s->mixed, Q4_HIDDEN);
    q4_hc_combine(residual, s->branch, s->inject,
                  Q4_HIDDEN, Q4_HC);
    trace_vector("hc_combine_attn", layer, residual, Q4_HC_DIM);
    if (!hc_mix(model, &model->layer[layer].ffn_hc, residual,
                s->mixed, s->inject)) return 0;
    if (profiling) {
        model->time_hc += now_seconds() - started;
        started = now_seconds();
    }
    if (!moe(model, layer, s->mixed, s->branch)) return 0;
    if (profiling) {
        model->time_moe += now_seconds() - started;
        started = now_seconds();
    }
    trace_vector("hc_mixed_ffn", layer, s->mixed, Q4_HIDDEN);
    trace_vector("ffn_out", layer, s->branch, Q4_HIDDEN);
    q4_hc_combine(residual, s->branch, s->inject,
                  Q4_HIDDEN, Q4_HC);
    trace_vector("l_last", layer, residual, Q4_HC_DIM);
    if (profiling) model->time_hc += now_seconds() - started;
    return 1;
}

static int layer_forward_batch(Q4Model *model, uint32_t layer,
                               float *const *residual,
                               uint32_t base_position, uint32_t batch_size)
{
    Q4Scratch *s = &model->scratch;
    const int profiling = getenv("Q4_PROFILE_BATCH") != NULL;
    double started = profiling ? now_seconds() : 0.0;
    if (!hc_mix_batch(model, &model->layer[layer].attn_hc,
                      residual, batch_size)) return 0;
    if (profiling) {
        model->time_batch_hc += now_seconds() - started;
        started = now_seconds();
    }
    if ((layer + 1u) % 4u == 0u) {
        if (!full_attention_batch(model, layer, base_position,
                                  s->batch_mixed, s->batch_branch,
                                  batch_size)) return 0;
        for (uint32_t token = 0; token < batch_size; ++token) {
            q4_hc_combine(residual[token],
                          s->batch_branch + (uint64_t)token * Q4_HIDDEN,
                          s->batch_inject + (uint64_t)token * Q4_HC,
                          Q4_HIDDEN, Q4_HC);
        }
    } else {
        if (!linear_attention_batch(model, layer, s->batch_mixed,
                                    s->batch_branch, batch_size)) return 0;
        for (uint32_t token = 0; token < batch_size; ++token)
            q4_hc_combine(residual[token],
                          s->batch_branch + (uint64_t)token * Q4_HIDDEN,
                          s->batch_inject + (uint64_t)token * Q4_HC,
                          Q4_HIDDEN, Q4_HC);
    }
    if (profiling) {
        model->time_batch_attention += now_seconds() - started;
        started = now_seconds();
    }
    if (!hc_mix_batch(model, &model->layer[layer].ffn_hc,
                      residual, batch_size)) return 0;
    if (profiling) model->time_batch_hc += now_seconds() - started;
    started = profiling ? now_seconds() : 0.0;
    if (!moe_batch(model, layer, s->batch_mixed,
                   s->batch_branch, batch_size)) return 0;
    if (profiling) model->time_batch_moe += now_seconds() - started;
    for (uint32_t token = 0; token < batch_size; ++token) {
        q4_hc_combine(residual[token],
                      s->batch_branch + (uint64_t)token * Q4_HIDDEN,
                      s->batch_inject + (uint64_t)token * Q4_HC,
                      Q4_HIDDEN, Q4_HC);
    }
    return 1;
}

static int alloc_state(Q4Model *model)
{
#define ALLOC(ptr, count) do { \
    (ptr) = (float *)calloc((size_t)(count), sizeof(float)); \
    if (!(ptr)) return 0; \
} while (0)
    ALLOC(model->residual, Q4_HC_DIM);
    ALLOC(model->conv_state, 36u * Q4_LINEAR_QKV * 3u);
    ALLOC(model->delta_state, 36u * Q4_LINEAR_HEADS *
          Q4_HEAD_DIM * Q4_HEAD_DIM);
    ALLOC(model->key_cache, (uint64_t)Q4_FULL_LAYERS * model->context_length *
          Q4_ATTN_KV);
    ALLOC(model->value_cache, (uint64_t)Q4_FULL_LAYERS * model->context_length *
          Q4_ATTN_KV);
    ALLOC(model->ple_history, (uint64_t)Q4_PLE_HISTORY * Q4_HC_DIM);
    ALLOC(model->index_tail, (uint64_t)Q4_QSA_LAYERS * Q4_QSA_RATIO *
          Q4_QSA_DIM);
    ALLOC(model->index_blocks, (uint64_t)Q4_QSA_LAYERS *
          ((model->context_length + Q4_QSA_RATIO - 1u) / Q4_QSA_RATIO) *
          Q4_QSA_DIM);
    Q4Scratch *s = &model->scratch;
    ALLOC(s->hc_norm, Q4_HC_DIM); ALLOC(s->hc_low, Q4_HC_RANK);
    ALLOC(s->hc_gate, Q4_HC_DIM); ALLOC(s->mixed, Q4_HIDDEN);
    ALLOC(s->inject, Q4_HC); ALLOC(s->branch, Q4_HIDDEN);
    ALLOC(s->wide0, Q4_ATTN_Q_GATE); ALLOC(s->wide1, Q4_LINEAR_VALUE);
    ALLOC(s->beta, Q4_LINEAR_HEADS); ALLOC(s->alpha, Q4_LINEAR_HEADS);
    ALLOC(s->attention, Q4_ATTN_OUT); ALLOC(s->query, Q4_ATTN_OUT);
    ALLOC(s->gate, Q4_ATTN_OUT); ALLOC(s->key, Q4_ATTN_KV);
    ALLOC(s->value, Q4_ATTN_KV);
    ALLOC(s->scores, (uint64_t)Q4_ATTN_HEADS * model->context_length);
    ALLOC(s->index_query, Q4_QSA_HEADS * Q4_QSA_DIM);
    ALLOC(s->index_key, Q4_QSA_DIM);
    ALLOC(s->index_scores,
          (model->context_length + Q4_QSA_RATIO - 1u) / Q4_QSA_RATIO);
    s->index_heap = (uint32_t *)calloc(Q4_QSA_HEAP_BLOCKS,
                                       sizeof(*s->index_heap));
    s->selected = (uint8_t *)calloc(model->context_length,
                                    sizeof(*s->selected));
    if (!s->index_heap || !s->selected) return 0;
    ALLOC(s->batch_hc_norm, Q4_BATCH_TILE * Q4_HC_DIM);
    ALLOC(s->batch_hc_low, Q4_BATCH_TILE * Q4_HC_RANK);
    ALLOC(s->batch_hc_gate, Q4_BATCH_TILE * Q4_HC_DIM);
    ALLOC(s->batch_mixed, Q4_BATCH_TILE * Q4_HIDDEN);
    ALLOC(s->batch_inject, Q4_BATCH_TILE * Q4_HC);
    ALLOC(s->batch_wide0, Q4_BATCH_TILE * Q4_ATTN_Q_GATE);
    ALLOC(s->batch_wide1, Q4_BATCH_TILE * Q4_LINEAR_VALUE);
    ALLOC(s->batch_beta, Q4_BATCH_TILE * Q4_LINEAR_HEADS);
    ALLOC(s->batch_alpha, Q4_BATCH_TILE * Q4_LINEAR_HEADS);
    ALLOC(s->batch_attention, Q4_BATCH_TILE * Q4_LINEAR_VALUE);
    ALLOC(s->batch_key, Q4_BATCH_TILE * Q4_ATTN_KV);
    ALLOC(s->batch_value, Q4_BATCH_TILE * Q4_ATTN_KV);
    ALLOC(s->batch_branch, Q4_BATCH_TILE * Q4_HIDDEN);
    ALLOC(s->batch_router, Q4_BATCH_TILE * Q4_EXPERTS);
    ALLOC(s->batch_moe_gate, Q4_BATCH_TILE * Q4_EXPERT_FFN);
    ALLOC(s->batch_moe_up, Q4_BATCH_TILE * Q4_EXPERT_FFN);
    ALLOC(s->batch_moe_hidden, Q4_BATCH_TILE * Q4_EXPERT_FFN);
    ALLOC(s->batch_moe_output, Q4_BATCH_TILE * Q4_HIDDEN);
    ALLOC(s->batch_routed_output, Q4_BATCH_TILE * Q4_ACTIVE_EXPERTS *
          Q4_HIDDEN);
    ALLOC(s->batch_shared_output, Q4_BATCH_TILE * Q4_HIDDEN);
    ALLOC(s->batch_logits, Q4_BATCH_TILE * Q4_VOCAB);
    ALLOC(s->union_gate, Q4_MOE_UNION_MAX * Q4_BATCH_TILE * Q4_EXPERT_FFN);
    ALLOC(s->union_up, Q4_MOE_UNION_MAX * Q4_BATCH_TILE * Q4_EXPERT_FFN);
    ALLOC(s->union_hidden, Q4_MOE_UNION_MAX * Q4_BATCH_TILE * Q4_EXPERT_FFN);
    ALLOC(s->union_output, Q4_MOE_UNION_MAX * Q4_BATCH_TILE * Q4_HIDDEN);
    s->union_quantized = (Q38Q8KBlock *)calloc(
        Q4_MOE_UNION_MAX * Q4_BATCH_TILE * (Q4_HIDDEN / 256u),
        sizeof(*s->union_quantized));
    if (!s->union_quantized) return 0;
    s->batch_quantized = (Q38Q8KBlock *)calloc(
        Q4_BATCH_TILE * (Q4_HC_DIM / 256u),
        sizeof(*s->batch_quantized));
    if (!s->batch_quantized) return 0;
    ALLOC(s->router, Q4_EXPERTS); ALLOC(s->expert_gate, Q4_EXPERT_FFN);
    ALLOC(s->expert_up, Q4_EXPERT_FFN); ALLOC(s->expert_hidden, Q4_EXPERT_FFN);
    ALLOC(s->expert_output, Q4_HIDDEN); ALLOC(s->shared_output, Q4_HIDDEN);
    ALLOC(s->routed_gate, Q4_ACTIVE_EXPERTS * Q4_EXPERT_FFN);
    ALLOC(s->routed_up, Q4_ACTIVE_EXPERTS * Q4_EXPERT_FFN);
    ALLOC(s->routed_hidden, Q4_ACTIVE_EXPERTS * Q4_EXPERT_FFN);
    ALLOC(s->routed_output, Q4_ACTIVE_EXPERTS * Q4_HIDDEN);
    ALLOC(s->ple_embedding, Q4_HIDDEN); ALLOC(s->ple_key, Q4_HC_DIM);
    ALLOC(s->ple_value, Q4_HIDDEN); ALLOC(s->ple_gated, Q4_HC_DIM);
    ALLOC(s->ple_norm, Q4_HC_DIM); ALLOC(s->logits, Q4_VOCAB);
    s->quantized = (Q38Q8KBlock *)calloc(Q4_HC_DIM / 256u,
                                         sizeof(*s->quantized));
    if (!s->quantized) return 0;
#undef ALLOC
    return 1;
}

Q4Model *q4_model_open_gguf(const char *first_shard, uint32_t context_length)
{
    if (context_length == 0u || context_length > Q4_MODEL_CONTEXT) {
        fprintf(stderr, "qwen4: context must be between 1 and %u tokens\n",
                Q4_MODEL_CONTEXT);
        return NULL;
    }
    Q4Model *model = (Q4Model *)calloc(1u, sizeof(*model));
    if (!model) return NULL;
    if (!q4_gguf_set_open(&model->gguf, first_shard, 0)) goto fail;
    if (!getenv("Q4_DISABLE_Q8_REPACK")) {
        int repacked = 1;
        for (size_t shard = 0; shard < model->gguf.shard_count; ++shard)
            repacked &= q38_prepare_q8_0_repacks(&model->gguf.shards[shard]);
        if (!repacked) {
            for (size_t shard = 0; shard < model->gguf.shard_count; ++shard)
                q38_release_q8_0_repacks(&model->gguf.shards[shard]);
            fprintf(stderr,
                    "qwen4: Q8_0 repack unavailable; using mapped weights\n");
        }
    }
    if (getenv("Q4_F32_REPACK")) {
        int repacked = 1;
        for (size_t shard = 0; shard < model->gguf.shard_count; ++shard)
            repacked &= q38_prepare_f32_repacks(&model->gguf.shards[shard]);
        if (!repacked) {
            for (size_t shard = 0; shard < model->gguf.shard_count; ++shard)
                q38_release_f32_repacks(&model->gguf.shards[shard]);
            fprintf(stderr,
                    "qwen4: F32 repack unavailable; using mapped weights\n");
        }
    }
    const Q38GGUF *meta = q4_gguf_metadata(&model->gguf);
    uint64_t layers = 0, hidden = 0, experts = 0, active = 0;
    Q38GGUFString arch;
    if (!q38_gguf_meta_string(meta, "general.architecture", &arch) ||
        arch.length != 8u || memcmp(arch.data, "qwen4exp", 8u) != 0 ||
        !q38_gguf_meta_u64(meta, "qwen4exp.block_count", &layers) ||
        !q38_gguf_meta_u64(meta, "qwen4exp.embedding_length", &hidden) ||
        !q38_gguf_meta_u64(meta, "qwen4exp.expert_count", &experts) ||
        !q38_gguf_meta_u64(meta, "qwen4exp.expert_used_count", &active) ||
        layers != Q4_LAYERS || hidden != Q4_HIDDEN || experts != Q4_EXPERTS ||
        active != Q4_ACTIVE_EXPERTS) {
        fprintf(stderr, "qwen4: unsupported Qwen3.8-Flash-Next contract\n");
        goto fail;
    }
    model->layers = (uint32_t)layers;
    model->hidden = (uint32_t)hidden;
    model->experts = (uint32_t)experts;
    model->active_experts = (uint32_t)active;
    model->context_length = context_length;
    model->embedding = weight(&model->gguf, "token_embd.weight");
    model->ple_embedding = weight(&model->gguf, "per_layer_token_embd.weight");
    model->output_hc.norm = weight(&model->gguf, "output_hc_norm.weight");
    model->output_hc.down = weight(&model->gguf, "output_hc_down.weight");
    model->output_hc.up = weight(&model->gguf, "output_hc_up.weight");
    model->output = weight(&model->gguf, "output.weight");
    model->output_scale = optional_weight(&model->gguf, "output.scale");
    int ok = shape(model->embedding, "token_embd", 2, Q4_HIDDEN,
                   Q4_VOCAB, 0) &&
             shape(model->ple_embedding, "ple_embedding", 2, 160,
                   320001536u, 0) &&
             shape(model->output_hc.norm, "output_hc_norm", 1, Q4_HC_DIM,
                   0, 0) &&
             shape(model->output_hc.down, "output_hc_down", 2, Q4_HC_DIM,
                   Q4_HC_RANK, 0) &&
             shape(model->output_hc.up, "output_hc_up", 2, Q4_HC_RANK,
                   Q4_HC_DIM, 0) &&
             shape(model->output, "output", 2, Q4_HIDDEN, Q4_VOCAB, 0);
    for (uint32_t layer = 0; layer < Q4_LAYERS; ++layer)
        ok &= bind_layer(model, &model->gguf, layer);
    const Q38GGUFMeta *multipliers = q38_gguf_find_meta(
        meta, "qwen4exp.ple.layer_multipliers");
    const Q38GGUFMeta *offsets = q38_gguf_find_meta(
        meta, "qwen4exp.ple.head_offsets");
    const Q38GGUFMeta *sizes = q38_gguf_find_meta(
        meta, "qwen4exp.ple.head_vocab_sizes");
    for (uint32_t i = 0; i < 3u; ++i)
        ok &= q38_gguf_meta_array_u64(multipliers, i,
                                      &model->ple_multipliers[i]);
    for (uint32_t i = 0; i < Q4_PLE_HEADS; ++i) {
        ok &= q38_gguf_meta_array_u64(offsets, i, &model->ple_offsets[i]);
        ok &= q38_gguf_meta_array_u64(sizes, i, &model->ple_sizes[i]);
    }
    if (!ok || !alloc_state(model)) goto fail;
    q4_model_reset(model);
    return model;
fail:
    q4_model_close(model);
    return NULL;
}

void q4_model_close(Q4Model *model)
{
    if (!model) return;
    for (size_t shard = 0; shard < model->gguf.shard_count; ++shard)
        q38_release_q8_0_repacks(&model->gguf.shards[shard]);
    for (size_t shard = 0; shard < model->gguf.shard_count; ++shard)
        q38_release_f32_repacks(&model->gguf.shards[shard]);
    q4_gguf_set_close(&model->gguf);
#define FREE(ptr) free(ptr)
    FREE(model->residual);
    FREE(model->conv_state); FREE(model->delta_state);
    FREE(model->key_cache); FREE(model->value_cache); FREE(model->ple_history);
    FREE(model->index_tail); FREE(model->index_blocks);
    Q4Scratch *s = &model->scratch;
    FREE(s->hc_norm); FREE(s->hc_low); FREE(s->hc_gate); FREE(s->mixed);
    FREE(s->inject); FREE(s->branch); FREE(s->wide0); FREE(s->wide1);
    FREE(s->beta); FREE(s->alpha); FREE(s->attention); FREE(s->query);
    FREE(s->gate); FREE(s->key); FREE(s->value); FREE(s->scores);
    FREE(s->index_query); FREE(s->index_key); FREE(s->index_scores);
    free(s->index_heap); free(s->selected);
    FREE(s->batch_hc_norm); FREE(s->batch_hc_low); FREE(s->batch_hc_gate);
    FREE(s->batch_mixed); FREE(s->batch_inject); free(s->batch_quantized);
    FREE(s->batch_wide0); FREE(s->batch_wide1);
    FREE(s->batch_beta); FREE(s->batch_alpha);
    FREE(s->batch_attention); FREE(s->batch_key); FREE(s->batch_value);
    FREE(s->batch_branch);
    FREE(s->batch_router); FREE(s->batch_moe_gate); FREE(s->batch_moe_up);
    FREE(s->batch_moe_hidden); FREE(s->batch_moe_output);
    FREE(s->batch_routed_output); FREE(s->batch_shared_output);
    FREE(s->batch_logits);
    FREE(s->union_gate); FREE(s->union_up); FREE(s->union_hidden);
    FREE(s->union_output); free(s->union_quantized);
    FREE(s->router); FREE(s->expert_gate); FREE(s->expert_up);
    FREE(s->expert_hidden); FREE(s->expert_output); FREE(s->shared_output);
    FREE(s->routed_gate); FREE(s->routed_up); FREE(s->routed_hidden);
    FREE(s->routed_output);
    FREE(s->ple_embedding); FREE(s->ple_key); FREE(s->ple_value);
    FREE(s->ple_gated); FREE(s->ple_norm); FREE(s->logits);
    free(s->quantized);
#undef FREE
    free(model);
}

void q4_model_reset(Q4Model *model)
{
    if (!model) return;
    model->position = 0u;
    model->previous[0] = Q4_EOS;
    model->previous[1] = Q4_EOS;
    model->time_ple = model->time_hc = model->time_attention = 0.0;
    model->time_moe = model->time_head = 0.0;
    model->time_gdn_input = model->time_gdn_state = 0.0;
    model->time_gdn_output = model->time_full_attention = 0.0;
    model->time_batch_hc = model->time_batch_attention = 0.0;
    model->time_batch_moe = 0.0;
    memset(model->residual, 0, Q4_HC_DIM * sizeof(float));
    memset(model->conv_state, 0,
           (uint64_t)36u * Q4_LINEAR_QKV * 3u * sizeof(float));
    memset(model->delta_state, 0,
           (uint64_t)36u * Q4_LINEAR_HEADS * Q4_HEAD_DIM * Q4_HEAD_DIM *
           sizeof(float));
    memset(model->key_cache, 0, (uint64_t)Q4_FULL_LAYERS *
           model->context_length * Q4_ATTN_KV * sizeof(float));
    memset(model->value_cache, 0, (uint64_t)Q4_FULL_LAYERS *
           model->context_length * Q4_ATTN_KV * sizeof(float));
    memset(model->ple_history, 0,
           (uint64_t)Q4_PLE_HISTORY * Q4_HC_DIM * sizeof(float));
    memset(model->index_tail, 0,
           (uint64_t)Q4_QSA_LAYERS * Q4_QSA_RATIO * Q4_QSA_DIM *
           sizeof(float));
}

Q4ModelState *q4_model_state_create(void)
{
    Q4ModelState *state = (Q4ModelState *)calloc(1u, sizeof(*state));
    if (!state) return NULL;
#define STATE_ALLOC(field, count) do { \
    state->field = (float *)malloc((size_t)(count) * sizeof(float)); \
    if (!state->field) { q4_model_state_destroy(state); return NULL; } \
} while (0)
    STATE_ALLOC(residual, Q4_HC_DIM);
    STATE_ALLOC(conv_state, (uint64_t)36u * Q4_LINEAR_QKV * 3u);
    STATE_ALLOC(delta_state, (uint64_t)36u * Q4_LINEAR_HEADS *
                Q4_HEAD_DIM * Q4_HEAD_DIM);
    STATE_ALLOC(ple_history, (uint64_t)Q4_PLE_HISTORY * Q4_HC_DIM);
    STATE_ALLOC(index_tail, (uint64_t)Q4_QSA_LAYERS * Q4_QSA_RATIO *
                Q4_QSA_DIM);
    STATE_ALLOC(logits, Q4_VOCAB);
#undef STATE_ALLOC
    return state;
}

void q4_model_state_destroy(Q4ModelState *state)
{
    if (!state) return;
    free(state->residual);
    free(state->conv_state);
    free(state->delta_state);
    free(state->key_cache);
    free(state->value_cache);
    free(state->ple_history);
    free(state->index_tail);
    free(state->index_blocks);
    free(state->logits);
    free(state);
}

static int state_cache_capacity(Q4ModelState *state, uint32_t tokens)
{
    if (tokens <= state->cache_capacity) return 1;
    const uint64_t count = (uint64_t)Q4_FULL_LAYERS * tokens * Q4_ATTN_KV;
    if (count > SIZE_MAX / sizeof(float)) return 0;
    float *key = (float *)malloc((size_t)count * sizeof(float));
    float *value = (float *)malloc((size_t)count * sizeof(float));
    const uint64_t blocks = (uint64_t)Q4_QSA_LAYERS *
        ((tokens + Q4_QSA_RATIO - 1u) / Q4_QSA_RATIO) * Q4_QSA_DIM;
    float *index = (float *)malloc((size_t)blocks * sizeof(float));
    if (!key || !value || !index) {
        free(key); free(value); free(index); return 0;
    }
    free(state->key_cache);
    free(state->value_cache);
    free(state->index_blocks);
    state->key_cache = key;
    state->value_cache = value;
    state->index_blocks = index;
    state->cache_capacity = tokens;
    return 1;
}

int q4_model_state_save(const Q4Model *model, Q4ModelState *state)
{
    if (!model || !state || !state_cache_capacity(state, model->position))
        return 0;
    state->position = model->position;
    state->previous[0] = model->previous[0];
    state->previous[1] = model->previous[1];
    memcpy(state->residual, model->residual,
           Q4_HC_DIM * sizeof(float));
    memcpy(state->conv_state, model->conv_state,
           (uint64_t)36u * Q4_LINEAR_QKV * 3u * sizeof(float));
    memcpy(state->delta_state, model->delta_state,
           (uint64_t)36u * Q4_LINEAR_HEADS * Q4_HEAD_DIM * Q4_HEAD_DIM *
           sizeof(float));
    const uint64_t layer_count = (uint64_t)model->position * Q4_ATTN_KV;
    for (uint32_t layer = 0; layer < Q4_FULL_LAYERS; ++layer) {
        const uint64_t source = (uint64_t)layer * model->context_length *
                                Q4_ATTN_KV;
        const uint64_t target = (uint64_t)layer * layer_count;
        memcpy(state->key_cache + target, model->key_cache + source,
               (size_t)layer_count * sizeof(float));
        memcpy(state->value_cache + target, model->value_cache + source,
               (size_t)layer_count * sizeof(float));
    }
    memcpy(state->ple_history, model->ple_history,
           (uint64_t)Q4_PLE_HISTORY * Q4_HC_DIM * sizeof(float));
    memcpy(state->index_tail, model->index_tail,
           (uint64_t)Q4_QSA_LAYERS * Q4_QSA_RATIO * Q4_QSA_DIM *
           sizeof(float));
    const uint64_t block_count = (uint64_t)Q4_QSA_LAYERS *
        (model->position / Q4_QSA_RATIO) * Q4_QSA_DIM;
    if (block_count && !state->index_blocks) return 0;
    const uint32_t model_blocks =
        (model->context_length + Q4_QSA_RATIO - 1u) / Q4_QSA_RATIO;
    const uint32_t saved_blocks = model->position / Q4_QSA_RATIO;
    for (uint32_t layer = 0; layer < Q4_QSA_LAYERS; ++layer)
        memcpy(state->index_blocks + (uint64_t)layer * saved_blocks *
               Q4_QSA_DIM,
               model->index_blocks + (uint64_t)layer * model_blocks *
               Q4_QSA_DIM,
               (size_t)saved_blocks * Q4_QSA_DIM * sizeof(float));
    (void)block_count;
    memcpy(state->logits, model->scratch.logits, Q4_VOCAB * sizeof(float));
    return 1;
}

int q4_model_state_restore(Q4Model *model, const Q4ModelState *state,
                           const float **logits)
{
    if (!model || !state || !logits || state->position == 0u ||
        state->position > model->context_length ||
        state->position > state->cache_capacity) return 0;
    model->position = state->position;
    model->previous[0] = state->previous[0];
    model->previous[1] = state->previous[1];
    memcpy(model->residual, state->residual,
           Q4_HC_DIM * sizeof(float));
    memcpy(model->conv_state, state->conv_state,
           (uint64_t)36u * Q4_LINEAR_QKV * 3u * sizeof(float));
    memcpy(model->delta_state, state->delta_state,
           (uint64_t)36u * Q4_LINEAR_HEADS * Q4_HEAD_DIM * Q4_HEAD_DIM *
           sizeof(float));
    const uint64_t layer_count = (uint64_t)state->position * Q4_ATTN_KV;
    for (uint32_t layer = 0; layer < Q4_FULL_LAYERS; ++layer) {
        const uint64_t source = (uint64_t)layer * layer_count;
        const uint64_t target = (uint64_t)layer * model->context_length *
                                Q4_ATTN_KV;
        memcpy(model->key_cache + target, state->key_cache + source,
               (size_t)layer_count * sizeof(float));
        memcpy(model->value_cache + target, state->value_cache + source,
               (size_t)layer_count * sizeof(float));
    }
    memcpy(model->ple_history, state->ple_history,
           (uint64_t)Q4_PLE_HISTORY * Q4_HC_DIM * sizeof(float));
    memcpy(model->index_tail, state->index_tail,
           (uint64_t)Q4_QSA_LAYERS * Q4_QSA_RATIO * Q4_QSA_DIM *
           sizeof(float));
    const uint32_t model_blocks =
        (model->context_length + Q4_QSA_RATIO - 1u) / Q4_QSA_RATIO;
    const uint32_t saved_blocks = state->position / Q4_QSA_RATIO;
    for (uint32_t layer = 0; layer < Q4_QSA_LAYERS; ++layer)
        memcpy(model->index_blocks + (uint64_t)layer * model_blocks *
               Q4_QSA_DIM,
               state->index_blocks + (uint64_t)layer * saved_blocks *
               Q4_QSA_DIM,
               (size_t)saved_blocks * Q4_QSA_DIM * sizeof(float));
    memcpy(model->scratch.logits, state->logits, Q4_VOCAB * sizeof(float));
    model->time_ple = model->time_hc = model->time_attention = 0.0;
    model->time_moe = model->time_head = 0.0;
    *logits = model->scratch.logits;
    return 1;
}

int q4_model_forward_token(Q4Model *model, uint32_t token_id,
                           const float **logits)
{
    if (!model || !logits || token_id >= Q4_VOCAB ||
        model->position >= model->context_length) return 0;
    if (!q38_tensor_row_f32(model->scratch.mixed, model->embedding, token_id))
        return 0;
    trace_vector("model.input_embed", UINT32_MAX,
                 model->scratch.mixed, Q4_HIDDEN);
    for (uint32_t group = 0; group < Q4_HC; ++group)
        memcpy(model->residual + (uint64_t)group * Q4_HIDDEN,
               model->scratch.mixed, Q4_HIDDEN * sizeof(float));
    for (uint32_t layer = 0; layer < Q4_LAYERS; ++layer) {
        if (layer == 1u) {
            const double started = getenv("Q4_PROFILE") ? now_seconds() : 0.0;
            if (!ple(model, token_id, model->residual)) return 0;
            if (getenv("Q4_PROFILE")) model->time_ple += now_seconds() - started;
        }
        if (!layer_forward(model, layer, model->residual,
                           model->position)) return 0;
    }
    const double head_started = getenv("Q4_PROFILE") ? now_seconds() : 0.0;
    if (!hc_mix(model, &model->output_hc, model->residual,
                model->scratch.mixed, NULL) ||
        !project(model, model->scratch.logits, model->scratch.mixed,
                 Q4_HIDDEN, model->output)) return 0;
    apply_scale(model->scratch.logits, Q4_VOCAB, model->output_scale, 0u);
    trace_vector("result_norm", UINT32_MAX,
                 model->scratch.mixed, Q4_HIDDEN);
    if (getenv("Q4_PROFILE")) {
        model->time_head += now_seconds() - head_started;
        fprintf(stderr,
                "PROFILE tokens=%u ple=%.6f hc=%.6f attn=%.6f moe=%.6f head=%.6f gdn_in=%.6f gdn_state=%.6f gdn_out=%.6f full=%.6f\n",
                model->position + 1u, model->time_ple, model->time_hc,
                model->time_attention, model->time_moe, model->time_head,
                model->time_gdn_input, model->time_gdn_state,
                model->time_gdn_output, model->time_full_attention);
    }
    ++model->position;
    *logits = model->scratch.logits;
    return 1;
}

int q4_model_prefill(Q4Model *model, const uint32_t *tokens,
                     uint32_t token_count, const float **logits)
{
    if (!model || !tokens || !token_count || !logits ||
        token_count > model->context_length - model->position) return 0;
    if (token_count < Q4_BATCH_TILE || getenv("Q4_TOKEN_MAJOR")) {
        for (uint32_t i = 0; i < token_count; ++i)
            if (!q4_model_forward_token(model, tokens[i], logits)) return 0;
        return 1;
    }
    float *residuals = (float *)malloc(
        (uint64_t)token_count * Q4_HC_DIM * sizeof(float));
    if (!residuals) return 0;
    for (uint32_t token = 0; token < token_count; ++token) {
        float *residual = residuals + (uint64_t)token * Q4_HC_DIM;
        if (!q38_tensor_row_f32(model->scratch.mixed, model->embedding,
                                tokens[token])) { free(residuals); return 0; }
        for (uint32_t group = 0; group < Q4_HC; ++group)
            memcpy(residual + (uint64_t)group * Q4_HIDDEN,
                   model->scratch.mixed, Q4_HIDDEN * sizeof(float));
    }
    const uint32_t base_position = model->position;
    for (uint32_t layer = 0; layer < Q4_LAYERS; ++layer) {
        for (uint32_t first = 0; first < token_count;
             first += Q4_BATCH_TILE) {
            const uint32_t count = token_count - first < Q4_BATCH_TILE
                                 ? token_count - first : Q4_BATCH_TILE;
            float *chunk[Q4_BATCH_TILE];
            for (uint32_t token = 0; token < count; ++token) {
                chunk[token] = residuals +
                    (uint64_t)(first + token) * Q4_HC_DIM;
                if (layer == 1u &&
                    !ple(model, tokens[first + token], chunk[token])) {
                    free(residuals); return 0;
                }
            }
            if (!layer_forward_batch(model, layer, chunk,
                                     base_position + first, count)) {
                free(residuals); return 0;
            }
        }
    }
    float *last = residuals + (uint64_t)(token_count - 1u) * Q4_HC_DIM;
    const int ok = hc_mix(model, &model->output_hc, last,
                          model->scratch.mixed, NULL) &&
        project(model, model->scratch.logits, model->scratch.mixed,
                Q4_HIDDEN, model->output);
    if (ok) apply_scale(model->scratch.logits, Q4_VOCAB,
                        model->output_scale, 0u);
    free(residuals);
    if (!ok) return 0;
    model->position += token_count;
    if (getenv("Q4_PROFILE_BATCH"))
        fprintf(stderr,
                "BATCH_PROFILE tokens=%u hc=%.6f attention=%.6f moe=%.6f\n",
                token_count, model->time_batch_hc,
                model->time_batch_attention, model->time_batch_moe);
    *logits = model->scratch.logits;
    return 1;
}

int q4_model_verify_greedy(Q4Model *model, const uint32_t *tokens,
                           uint32_t token_count, uint32_t *next_ids,
                           const float **logits)
{
    if (!model || !tokens || !token_count || token_count > Q4_BATCH_TILE ||
        !next_ids || !logits ||
        token_count > model->context_length - model->position) return 0;
    float *residuals = (float *)malloc(
        (uint64_t)token_count * Q4_HC_DIM * sizeof(float));
    if (!residuals) return 0;
    for (uint32_t token = 0; token < token_count; ++token) {
        float *residual = residuals + (uint64_t)token * Q4_HC_DIM;
        if (!q38_tensor_row_f32(model->scratch.mixed, model->embedding,
                                tokens[token])) {
            free(residuals);
            return 0;
        }
        for (uint32_t group = 0; group < Q4_HC; ++group)
            memcpy(residual + (uint64_t)group * Q4_HIDDEN,
                   model->scratch.mixed, Q4_HIDDEN * sizeof(float));
    }
    const uint32_t base_position = model->position;
    for (uint32_t layer = 0; layer < Q4_LAYERS; ++layer) {
        float *chunk[Q4_BATCH_TILE];
        for (uint32_t token = 0; token < token_count; ++token) {
            chunk[token] = residuals + (uint64_t)token * Q4_HC_DIM;
            if (layer == 1u && !ple(model, tokens[token], chunk[token])) {
                free(residuals);
                return 0;
            }
        }
        if (!layer_forward_batch(model, layer, chunk,
                                 base_position, token_count)) {
            fprintf(stderr, "qwen4: batch verifier failed at layer %u\n", layer);
            free(residuals);
            return 0;
        }
    }
    float *chunk[Q4_BATCH_TILE];
    for (uint32_t token = 0; token < token_count; ++token)
        chunk[token] = residuals + (uint64_t)token * Q4_HC_DIM;
    int ok = hc_mix_batch(model, &model->output_hc, chunk, token_count) &&
             project_batch(model, model->scratch.batch_logits,
                           model->scratch.batch_mixed, token_count,
                           Q4_HIDDEN, model->output);
    if (!ok) fprintf(stderr, "qwen4: batch verifier output head failed\n");
    for (uint32_t token = 0; ok && token < token_count; ++token) {
        float *row = model->scratch.batch_logits +
                     (uint64_t)token * Q4_VOCAB;
        apply_scale(row, Q4_VOCAB, model->output_scale, 0u);
        uint32_t best = 0u;
        for (uint32_t id = 1u; id < Q4_VOCAB; ++id)
            if (row[id] > row[best]) best = id;
        next_ids[token] = best;
        if (token + 1u == token_count)
            memcpy(model->scratch.logits, row,
                   Q4_VOCAB * sizeof(float));
    }
    free(residuals);
    if (!ok) return 0;
    model->position += token_count;
    *logits = model->scratch.logits;
    return 1;
}

uint32_t q4_model_vocab_size(const Q4Model *model)
{
    return model ? Q4_VOCAB : 0u;
}

uint32_t q4_model_layer_count(const Q4Model *model)
{
    return model ? model->layers : 0u;
}

uint32_t q4_model_hidden_size(const Q4Model *model)
{
    return model ? model->hidden : 0u;
}

uint32_t q4_model_expert_count(const Q4Model *model)
{
    return model ? model->experts : 0u;
}

uint32_t q4_model_active_expert_count(const Q4Model *model)
{
    return model ? model->active_experts : 0u;
}

uint32_t q4_model_position(const Q4Model *model)
{
    return model ? model->position : 0u;
}

uint32_t q4_model_context_length(const Q4Model *model)
{
    return model ? model->context_length : 0u;
}
