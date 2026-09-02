# Qwen3.8-Flash-Next inference architecture

The runtime directly maps the three pinned Unsloth Dynamic `UD-IQ1_S` GGUF
shards. The first shard contains model metadata and the complete tokenizer but
no tensors; the other two contain 1,224 tensors. The files are never joined or
converted.

## Model contract

| component | checkpoint value |
|---|---:|
| main parameters | 125B |
| activated main parameters | 6B per token |
| n-gram embedding parameters | 51B |
| decoder layers | 48 |
| hidden width | 2,560 |
| residual streams | 4 |
| residual mixer rank | 320 |
| experts / routed experts | 512 / 10 |
| routed and shared expert width | 640 |
| Gated DeltaNet / QSA layers | 36 / 12 |
| vocabulary | 248,320 |

Every layer first mixes the four residual streams into one 2,560-wide token
state, executes either Gated DeltaNet or Qwen Sparse Attention, and scatters
the result back into all four streams. A second mixer feeds the MoE block. The
output head uses another mixer instead of a separate final RMSNorm.

## PLE n-gram embedding

Layer 1 receives sixteen 160-wide rows from a shared 320,001,536-row IQ4_NL
table. Eight rows represent a token bigram and eight represent a trigram. Row
IDs are deterministic 64-bit hashes of the current and preceding tokens. Only
the selected rows are decoded; the 26.82 GiB table is never expanded or read
sequentially in full.

PLE projects the gathered 2,560 values into a four-stream key and a shared
value. Per-stream similarity gates the value, followed by a four-tap depthwise
causal convolution with dilation three.

## Attention and recurrent state

Three of every four layers use Gated DeltaNet with 16 query/key heads, 48 value
heads and 128-wide state. The implementation retains convolution and recurrent
state between tokens.

Every fourth layer uses 24 query heads, two key/value heads and a 256-wide head.
Within 2,048 tokens, attention remains dense. Beyond that budget, the four-head
QSA indexer mean-pools raw keys in four-token blocks, normalizes and rotates
each completed block, and selects up to 2,051 token positions from the best
blocks. The incomplete current block is always visible. Attention still
accumulates selected tokens in causal position order. Configured context may
reach the checkpoint's 262,144-token limit when the machine has enough memory;
the user-facing default is 8,192.
Configured capacity adds approximately 51 KiB of runtime state per token,
mostly for the twelve K/V cache slots. Physical model pages and
the recurrent state are separate from this context-dependent allocation.

## MoE execution

Router logits use a 512-way softmax. The best ten probabilities are selected
and renormalized, then their packed gate/up/down planes are evaluated together
with one gated shared expert. Eleven expert tasks run concurrently, while the
final routed accumulation retains a stable slot order.

For prefill, groups of four tokens share Hyper-Connection and Gated DeltaNet
weight reads. Their routed expert IDs are merged into one union of at most 40
experts. Union entries run concurrently, but each result is written back to
its token's original route slot before stable accumulation. PLE, convolution,
DeltaNet and causal attention state still advance in token order.

The GGUF mapping is read-only. The operating system manages mapped model pages.
Immediate `POSIX_MADV_WILLNEED` hints are available through `Q4_PREFETCH=1`
for cold-storage experiments, but are disabled by default because they slowed
paired hot-page runs without enough pipeline distance.
