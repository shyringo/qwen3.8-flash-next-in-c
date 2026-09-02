CC       ?= cc
BUILD    ?= build
BIN      ?= bin

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH ?= -mcpu=native
  else
    ARCH ?= -march=native
  endif
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp
else
  ARCH ?= -march=native
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

WARN := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla
CFLAGS ?= -O3 -std=c99 $(WARN) $(ARCH) $(OMP_CFLAGS) -ffp-contract=off
LDFLAGS ?= -lm -pthread $(OMP_LDFLAGS)
INCLUDES := -Iinclude -Iinclude/qwen38 -Iinclude/qwen4 -Ithird_party -Isrc/io -Isrc/cli

GGUF_OBJ := $(BUILD)/src/io/qwen38_gguf.o
Q4_GGUF_OBJ := $(BUILD)/src/io/qwen4_gguf.o
Q4_MODEL_OBJ := $(BUILD)/src/qwen4/qwen4_model.o
Q4_OPS_OBJ := $(BUILD)/src/qwen4/qwen4_ops.o
QUANT_OBJ := $(BUILD)/src/qwen38/qwen38_quant.o
TOKENIZER_OBJ := $(BUILD)/src/qwen38/qwen38_tokenizer.o
SAMPLER_OBJ := $(BUILD)/src/qwen38/qwen38_sampler.o
HTTP_OBJ := $(BUILD)/src/cli/qwen38_http.o
TOOL_OBJ := $(BUILD)/src/cli/qwen38_tool.o
TEST_BINS := $(BIN)/test_qwen38_gguf $(BIN)/test_qwen38_quant \
	$(BIN)/test_qwen38_sampler $(BIN)/test_qwen38_nfc \
	$(BIN)/test_qwen38_http $(BIN)/test_qwen38_tool \
	$(BIN)/test_qwen4_gguf $(BIN)/test_qwen4_ops
TOOL_BINS := $(BIN)/qwen4-meta-inspect \
	$(BIN)/qwen4-tensor-inspect \
	$(BIN)/qwen4-contract-probe \
	$(BIN)/qwen4-logits-probe \
	$(BIN)/qwen4-state-probe \
	$(BIN)/qwen4-qsa-probe

.PHONY: all test tools strict portable sanitize clean

all: $(BIN)/qwen4

tools: $(TOOL_BINS)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(GGUF_OBJ): include/qwen38/qwen38_gguf.h
$(QUANT_OBJ): include/qwen38/qwen38_quant.h include/qwen38/qwen38_gguf.h \
	third_party/ggml-common.h
$(TOKENIZER_OBJ): include/qwen38/qwen38_tokenizer.h \
	include/qwen38/qwen38_gguf.h third_party/tok.h third_party/tok_unicode.h \
	third_party/tok_nfc.h third_party/tok_nfc_data.h
$(SAMPLER_OBJ): include/qwen38/qwen38_sampler.h

$(BIN):
	@mkdir -p $@

$(BIN)/test_qwen38_gguf: tests/unit/test_qwen38_gguf.c $(TOKENIZER_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_quant: tests/unit/test_qwen38_quant.c $(QUANT_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_sampler: tests/unit/test_qwen38_sampler.c $(SAMPLER_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_nfc: tests/unit/test_qwen38_nfc.c \
	third_party/tok_nfc.h third_party/tok_nfc_data.h | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_http: tests/unit/test_qwen38_http.c $(HTTP_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen38_tool: tests/unit/test_qwen38_tool.c $(TOOL_OBJ) $(HTTP_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen4_gguf: tests/unit/test_qwen4_gguf.c $(Q4_GGUF_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_qwen4_ops: tests/unit/test_qwen4_ops.c $(Q4_OPS_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4-meta-inspect: src/cli/qwen4_meta_inspect.c $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4-tensor-inspect: src/cli/qwen4_tensor_inspect.c $(Q4_GGUF_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4-contract-probe: src/cli/qwen4_contract_probe.c $(Q4_MODEL_OBJ) $(Q4_OPS_OBJ) $(QUANT_OBJ) $(Q4_GGUF_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4-logits-probe: src/cli/qwen4_logits_probe.c $(Q4_MODEL_OBJ) $(Q4_OPS_OBJ) $(QUANT_OBJ) $(Q4_GGUF_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4-state-probe: src/cli/qwen4_state_probe.c $(Q4_MODEL_OBJ) $(Q4_OPS_OBJ) $(QUANT_OBJ) $(Q4_GGUF_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4-qsa-probe: src/cli/qwen4_qsa_probe.c $(Q4_MODEL_OBJ) $(Q4_OPS_OBJ) $(QUANT_OBJ) $(Q4_GGUF_OBJ) $(GGUF_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/qwen4: src/cli/qwen4_main.c $(Q4_MODEL_OBJ) $(Q4_OPS_OBJ) $(QUANT_OBJ) $(TOKENIZER_OBJ) $(SAMPLER_OBJ) $(Q4_GGUF_OBJ) $(GGUF_OBJ) $(HTTP_OBJ) $(TOOL_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

test: $(TEST_BINS) $(BIN)/qwen4
	./$(BIN)/test_qwen38_gguf
	./$(BIN)/test_qwen38_quant
	./$(BIN)/test_qwen38_sampler
	./$(BIN)/test_qwen38_nfc
	./$(BIN)/test_qwen38_http
	./$(BIN)/test_qwen38_tool
	./$(BIN)/test_qwen4_gguf
	./$(BIN)/test_qwen4_ops
	./$(BIN)/qwen4 --help >/dev/null 2>&1

strict:
	$(MAKE) BUILD=build/strict BIN=build/strict/bin \
		CFLAGS="-O3 -std=c99 $(WARN) -Werror $(ARCH) $(OMP_CFLAGS) -ffp-contract=off" all tools test

portable:
	$(MAKE) BUILD=build/portable BIN=build/portable/bin \
		ARCH= OMP_CFLAGS= OMP_LDFLAGS= all tools test

sanitize:
	$(MAKE) BUILD=build/sanitize BIN=build/sanitize/bin \
		ARCH= OMP_CFLAGS= OMP_LDFLAGS= \
		CFLAGS="-O1 -g -std=c99 $(WARN) -fsanitize=address,undefined -fno-omit-frame-pointer -ffp-contract=off" \
		LDFLAGS="-lm -pthread -fsanitize=address,undefined" all tools test

clean:
	rm -rf $(BUILD) $(BIN)
