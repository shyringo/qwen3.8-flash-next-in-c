#ifndef QWEN4_MODEL_H
#define QWEN4_MODEL_H

#include <stdint.h>

typedef struct Q4Model Q4Model;
typedef struct Q4ModelState Q4ModelState;

Q4Model *q4_model_open_gguf(const char *first_shard, uint32_t context_length);
void q4_model_close(Q4Model *model);
void q4_model_reset(Q4Model *model);
Q4ModelState *q4_model_state_create(void);
void q4_model_state_destroy(Q4ModelState *state);
int q4_model_state_save(const Q4Model *model, Q4ModelState *state);
int q4_model_state_restore(Q4Model *model, const Q4ModelState *state,
                           const float **logits);
int q4_model_forward_token(Q4Model *model, uint32_t token_id,
                           const float **logits);
int q4_model_prefill(Q4Model *model, const uint32_t *tokens,
                     uint32_t token_count, const float **logits);

uint32_t q4_model_vocab_size(const Q4Model *model);
uint32_t q4_model_layer_count(const Q4Model *model);
uint32_t q4_model_hidden_size(const Q4Model *model);
uint32_t q4_model_expert_count(const Q4Model *model);
uint32_t q4_model_active_expert_count(const Q4Model *model);
uint32_t q4_model_position(const Q4Model *model);
uint32_t q4_model_context_length(const Q4Model *model);

#endif
