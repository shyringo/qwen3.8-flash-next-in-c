# Optimizations and provenance

The implementation is classified into reused code/data, adaptations of public
formats or ideas, and project-specific engineering.

## Reused code and data

- ByteLevel BPE and Unicode normalization originate from the tokenizer lineage
  documented in `NOTICE`.
- GGUF quantization layouts and fixed IQ codebooks follow ggml's published MIT
  formats and tables.
- The bounded HTTP JSON parser uses the included MIT-licensed jsmn source.

## Adapted public ideas

- Packed-weight computation, activation quantization and multi-row CPU kernels
  extend the native Qwen3.8-27B runtime in this project family.
- Expert routing, bounded caching/prefetch concepts and storage-aware MoE
  scheduling adapt experience from the public
  `deepseek-v4-flash-0731-in-c` project.
- Qwen4Exp tensor meanings and equations follow the official model metadata,
  chat template and technical report. A fixed llama.cpp build is an external
  correctness oracle only; it is not linked or called at runtime.

## Project-specific engineering

- **Metadata-first split GGUF loading.** Treat a zero-tensor first shard as the
  authoritative tokenizer/configuration source and resolve tensors across all
  read-only shard mappings without concatenation.
- **Bounded-memory weight streaming.** Under an 8 GiB budget, skip optional
  Q8_0 repacks and discard clean mapped pages after each layer. The kernel math
  and selected GGUF stay unchanged while resident memory remains bounded.
- **On-demand PLE row decoding.** Compute sixteen 64-bit n-gram hashes and read
  only their 160-value IQ4_NL rows from a 26.82 GiB table.
- **Incremental QSA block indexing.** Keep only each layer's four raw tail
  keys. Once a block completes, pool, normalize and rotate it exactly once;
  later tokens score the compact block cache with a bounded 513-entry heap.
- **Native four-stream residual execution.** Implement low-rank gated mixing,
  per-stream grouped RMSNorm and stable scatter directly in C.
- **Expert-level MoE parallelism.** Run ten routed experts and the shared
  expert as one outer task group, eliminating thousands of small nested
  OpenMP regions while preserving accumulation order.
- **Layout-aware Q8_0 row groups.** Repack only high-compression-ratio Q8_0
  matrices into block-major groups of eight rows. Continuous int8 expansion
  and batched FP16 scale conversion retain each row's original FMA sequence.
- **Vector-table IQ4_NL expert output.** Evaluate eight output rows together,
  use `vpshufb` for sixteen-value codebook lookup, and preserve the original
  interleaved `0,16,1,17,...` accumulation order.
- **Cross-row F32 routing.** Keep eight independent router accumulators in one
  AVX2 register, removing scalar dependency chains without transposing the
  mapped model.
- **Transposed DeltaNet state.** Store independent state columns contiguously
  across SIMD lanes, then fuse decay with prediction and update eight columns
  at a time. State snapshots remain exact byte copies.
- **Recurrence-safe batch-4 prefill.** Batch Hyper-Connection and GDN
  projections while convolution, DeltaNet, PLE and causal attention advance
  in token order. Group routed tokens by expert, evaluate the expert union in
  parallel, then accumulate results in each token's original route-slot order.
- **Cross-token GDN state traversal.** Keep recurrence sequential inside each
  independent channel and head while one OpenMP team advances four positions,
  removing six short parallel-region boundaries per GDN layer.
- **Fused batch full attention.** Quantize four inputs once, share Q/K/V/O
  weight reads, write all causal KV rows, then let each head advance the four
  positions in order inside one parallel region.
- **Q8_0 row-and-token reuse.** A block-major Q8_0 load updates eight output
  rows for four token lanes with separate accumulators. Each result retains the
  original column order and is byte-identical to the scalar-layout path.
- **Measured prefetch gating.** Immediate expert read-ahead is available for
  cold-storage experiments through `Q4_PREFETCH=1`, but stays off by default
  because paired hot-page runs showed that syscall overhead outweighed its
  lack of pipeline distance.
- **Quantized-input reuse.** Quantize one activation once and reuse it across
  all compatible projections and selected experts.
- **Native resident chat and tool API.** Keep the 67.56 GiB mapping and runtime
  state resident behind a loopback endpoint with OpenAI-compatible messages,
  function tools, matched result replay, SSE token deltas and keep-alives.
- **Exact cross-turn prefix reuse.** Snapshot the post-prefill recurrent state,
  packed per-layer attention prefix, PLE history and next-token logits. A
  subsequent request that begins with the exact same token sequence restores
  that boundary and computes only newly appended conversation tokens.

On the reference laptop, a resident API request with 23 prompt tokens and 16
output tokens reached TTFT 3.731 s and TPOT 0.199 s/token (5.03 token/s) after
a different representative warm-up request. A fixed 16-position token-major
forward reached 4.98 positions/s. The optimized batch-4 verifier reached 9.89
positions/s and the fixed 16-position batch prefill reached 8.94 positions/s.
Performance remains sensitive to temperature, background load and page-cache
state.
