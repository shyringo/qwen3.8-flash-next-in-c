<h1 align="center">Qwen3.8-Flash-Next in C: The Best Model Under 200B, Near 10 token/s on a Laptop CPU</h1>

<p align="center">
  <strong>Run the best model under 200B near 10 token/s on a single laptop CPU, with exact batch verification.</strong><br>
  Run the 125B-A6B + 51B PLE model in native C: no GPU, CUDA, Python, PyTorch, model conversion, or external inference runtime.<br>
  Chat in the terminal or connect your apps through a resident OpenAI-compatible API.
</p>

<table align="center">
  <tr>
    <td align="center"><strong>9.89 token/s</strong><br>exact batch-4<br>verification throughput</td>
    <td align="center"><strong>Best under 200B</strong><br>125B-A6B main model<br>+ 51B PLE</td>
    <td align="center"><strong>5.03 token/s</strong><br>0.199 s/token<br>resident chat TPOT</td>
    <td align="center"><strong>8 GB RAM</strong><br>minimum path<br>selected automatically</td>
    <td align="center"><strong>No added approximation</strong><br>optimized paths preserve<br>the selected GGUF's results</td>
  </tr>
</table>

Qwen3.8-Flash-Next scores 56 on the current
[Artificial Analysis Intelligence Index](https://artificialanalysis.ai/models/qwen3-8-flash-next),
the highest score for a model below 200B total parameters.

<p align="center">
  <a href="https://github.com/shyringo/qwen3.8-flash-next-in-c/actions/workflows/ci.yml"><img src="https://github.com/shyringo/qwen3.8-flash-next-in-c/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/shyringo/qwen3.8-flash-next-in-c/releases"><img src="https://img.shields.io/github/v/release/shyringo/qwen3.8-flash-next-in-c" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/shyringo/qwen3.8-flash-next-in-c" alt="Code license"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a><br>
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#environment-requirements">Requirements</a> ·
  <a href="#measured-performance">Performance</a> ·
  <a href="#inference-accuracy">Accuracy</a> ·
  <a href="#inference-optimizations-implemented-in-this-project">Optimizations</a>
</p>

## Quick start

On Ubuntu, Debian, or Windows with WSL2:

```bash
sudo apt update
sudo apt install -y build-essential curl git
git clone https://github.com/shyringo/qwen3.8-flash-next-in-c.git
cd qwen3.8-flash-next-in-c
./qwen4.sh
```

On macOS, install Apple's command-line tools and OpenMP first:

```bash
xcode-select --install
brew install libomp
git clone https://github.com/shyringo/qwen3.8-flash-next-in-c.git
cd qwen3.8-flash-next-in-c
./qwen4.sh
```

The launcher compiles the engine, downloads and verifies the pinned 67.56 GiB
Unsloth `UD-IQ1_S` split GGUF from ModelScope with resume support, then opens
an interactive conversation. No weight conversion is needed. Enter `/reset`
for a new conversation and `/exit` to quit.

Run one request:

```bash
./qwen4.sh --prompt "Where are the boundaries of technology?" --max-tokens 256
```

Choose direct answers or visible reasoning:

```bash
./qwen4.sh --no-thinking
./qwen4.sh --thinking
./qwen4.sh --system "You are the only interface humanity needs."
```

Keep the model loaded behind a loopback OpenAI-compatible API:

```bash
./qwen4.sh --server 8080 --no-thinking
```

Use `http://127.0.0.1:8080/v1` as the base URL and
`qwen3.8-flash-next-in-c` as the model. No API key is required. Streaming,
function tools, parallel tool calls, and matched tool-result replay are
supported. See [Local API](docs/QWEN4_API.md).

Change context capacity or CPU threads without rebuilding:

```bash
QWEN4_CONTEXT=16384 ./qwen4.sh
QWEN4_THREADS=8 ./qwen4.sh
```

Set the available memory budget explicitly when needed:

```bash
./qwen4.sh --memory-gib 8
```

Machines with about 8 GB RAM automatically use the same bounded-memory weight
streaming path. It preserves the selected GGUF's results while trading speed
for lower residency.

The default context is 8,192 tokens. QSA keeps attention work bounded beyond
2,048 tokens; configured capacity may reach the model's 262,144-token limit
when enough memory is available.

## Environment requirements

- A 64-bit POSIX system, a C compiler, `make`, and `curl`. Windows users should
  use WSL2.
- At least 75 GB of free disk space for the pinned split GGUF and download
  headroom.
- 8 GB RAM is the minimum supported path. It is selected automatically on
  low-memory machines or explicitly with `--memory-gib 8`; a 2,048-context,
  16-token run measured 596 MiB peak RSS. At least 12 GB is recommended for
  the faster default path. Configured capacity adds about 51 KiB of runtime
  state per token.
- An x86-64 CPU with AVX2 is recommended for the fastest kernels. A tested
  portable scalar build is available for other 64-bit POSIX CPUs.

Under WSL2, the default model cache is inside the Linux filesystem at
`~/.cache/qwen3.8-flash-next-in-c/model`, avoiding the slower `/mnt/c` and
`/mnt/d` path for the 67.56 GiB weight set. Set `QWEN4_MODEL_DIR` to choose
another location.

This release supports text chat, text generation and function tools. It does
not accept image or video input.

## Measured performance

These are observed wall-clock values, not estimates:

| workload | TTFT / total | TPOT | throughput | peak RSS |
|---|---:|---:|---:|---:|
| resident API: 23-token prompt, 16-token output | **3.731 s** | **0.199 s/token** | **5.03 token/s** | - |
| 8 GB path: 19-token prompt, 16-token output | **4.156 s** | **0.611 s/token** | **1.64 token/s** | **596 MiB** |
| exact batch-4 verification, 4 positions | **0.405 s total** | - | **9.89 positions/s** | **6.55 GiB** |
| fixed 16-position token-major forward | 3.216 s total | - | **4.98 positions/s** | - |
| batch-4 prefill, fixed 16 positions | 1.789 s total | - | **8.94 positions/s** | - |
| 16-token one-shot, context 2,048 | - | - | - | **8.99 GiB** |

Reference environment: Intel Core i5-1340P laptop, 32 GB host memory, Windows
11 with WSL2 Ubuntu 22.04, GCC 11.4, 11 workers, pinned weights on the
WSL2 ext4 filesystem, warm model pages, `OMP_WAIT_POLICY=ACTIVE` and greedy
inference. Power mode,
temperature, page-cache state and background programs materially affect CPU
results.

The near-10 result is aggregate throughput for four positions evaluated in one
exact batch, not single-conversation TPOT. The benchmark first computes a
token-by-token reference, resets the model, and reports speed only when all
four greedy IDs and the complete final logits are byte-identical. It uses a
256-token capacity and a larger Q8_0 repack arena; normal chat keeps the more
memory-efficient defaults.

In a representative component profile, each token-major position used about
0.077 s MoE, 0.087 s attention/GDN, 0.053 s hyper-connection and 0.010 s
output-head time. Thermal state changes the absolute values.
The resident API excludes model download, process startup and model mapping
from TTFT. It also reuses an exact tokenized conversation prefix between
successive requests, reducing multi-turn prefill without changing logits.

Run the fixed reproducible workload:

```bash
./scripts/benchmark-qwen4.sh
```

Run the correctness-gated batch throughput benchmark:

```bash
./scripts/benchmark-qwen4-batch.sh
```

Share results from another CPU in the
[community benchmark thread](https://github.com/shyringo/qwen3.8-flash-next-in-c/discussions/1).

The benchmark uses active OpenMP waiting while a request is running. Normal
interactive chat and server launchers use passive waiting so an idle resident
model does not keep the laptop CPU at 100%.

## Inference optimizations implemented in this project

The complete implementation is classified as reused code and data,
adaptations of published formats or ideas, and project-specific engineering in
[Optimizations and provenance](docs/QWEN4_OPTIMIZATIONS.md). The principal
project-specific work includes:

- **Metadata-first split GGUF loading.** A metadata-only first shard supplies
  configuration and tokenizer data while tensors resolve across read-only
  shard mappings without concatenation or conversion.
- **Bounded-memory weight streaming.** The 8 GB path skips optional weight
  repacks and releases clean mapped pages after each layer, keeping only the
  active working set resident without changing model arithmetic.
- **On-demand 51B PLE decoding.** Sixteen n-gram hashes select only the needed
  IQ4_NL rows from the 26.82 GiB per-layer embedding table.
- **Incremental QSA block indexing.** Four-token key blocks are pooled,
  normalized and rotated once; a bounded 513-entry heap keeps long-context
  attention to the best blocks plus the current tail.
- **Native four-stream residual execution.** Low-rank gated hyper-connections,
  grouped RMSNorm and stable scatter run directly in C.
- **Expert-union MoE batching.** Token-major decode runs ten routed experts and
  one shared expert in parallel. Batch-4 prefill deduplicates experts across
  tokens, evaluates the union concurrently, then restores route-slot order.
- **Layout-aware low-bit SIMD.** Selective block-major Q8_0 row groups,
  cross-row F32 routing and `vpshufb` IQ4_NL lookup preserve each output's
  accumulation order while removing scalar decode bottlenecks.
- **Transposed DeltaNet state.** Independent recurrent columns are contiguous
  SIMD lanes; decay, prediction and update process eight columns at a time.
- **Recurrence-safe batch-4 prefill.** Hyper-Connection and GDN projections
  share weight reads while PLE, convolution, DeltaNet and attention state
  advance in exact token order.
- **Cross-token GDN state traversal.** One OpenMP team advances four tokens in
  sequence inside each independent channel/head, replacing hundreds of short
  fork/barrier cycles without reordering recurrent updates.
- **Fused batch full attention.** Q/K/V/O weights serve four causal positions
  per pass; KV writes happen first, then each head advances the four positions
  in exact order inside one parallel region.
- **Two-dimensional Q8_0 reuse.** A block-major weight load updates eight
  output rows across four token lanes, retaining an independent accumulator
  and the original FMA order for every result.
- **Quantized-input reuse and measured prefetch gating.** An activation is
  quantized once for compatible projections; immediate expert read-ahead is
  disabled by default after paired hot-page measurements showed a regression.
- **Exact cross-turn prefix reuse.** The resident API snapshots recurrent,
  PLE, attention and QSA state after prefill, then computes only appended
  messages when the next request has the exact same token prefix.
- **Native chat and function-tool API.** Bounded C HTTP/JSON, UTF-8-safe SSE,
  parallel calls, result replay, readiness and model discovery require no
  Python service or external inference engine.

The native model graph and runtime are not a wrapper around another inference
engine. Code and idea boundaries, upstream revisions and licenses are recorded
in [NOTICE](NOTICE).

## Inference accuracy

The engine introduces **no additional approximation beyond the selected
quantized GGUF**. This is a runtime-fidelity claim: the 1-bit-weight GGUF is
itself a lossy quantization of the original BF16/FP8 checkpoint.

For the pinned model and fixed 16-token prompt, all optimized native paths
preserve the same 248,320-dimensional full-logit SHA-256:

```text
34db8f9d482429a2a3a10bbe600a59b7ecf6813cf9c3fcd6f356430d19ac3e36
```

A fixed CPU-only llama.cpp reference returns the same top-five token IDs in
the same order. Across the complete vocabulary, cosine similarity is
0.994556324. Token-major/batch-4 execution, activation reuse, expert-union
parallelism, low-bit SIMD, transposed recurrent state, 2,048/4,096 context
maintenance, and saved/restored conversation state are checked independently. See
[Inference accuracy evidence](docs/QWEN4_CORRECTNESS.md).

```bash
make strict
make portable
make sanitize
```

## License and acknowledgements

Repository source code is licensed under Apache License 2.0. Model weights use
the separate [Qwen Community License 1.0](docs/MODEL_LICENSE.md); review it
before downloading or using the model.

The launcher downloads pinned GGUF shards from
[Unsloth's Qwen3.8-Flash-Next collection](https://www.modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF)
on ModelScope, with a Hugging Face fallback. The official model and
architecture are published by
[Qwen](https://github.com/QwenLM/Qwen3.8-Flash-Next).

Low-bit CPU and tokenizer foundations were carried forward from
[qwen3.8-27b-in-c](https://github.com/shyringo/qwen3.8-27b-in-c). Storage-aware
MoE ideas were adapted from
[deepseek-v4-flash-0731-in-c](https://github.com/shyringo/deepseek-v4-flash-0731-in-c),
whose tokenizer lineage credits
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c). GGUF/IQ
formats and codebooks follow published [ggml](https://github.com/ggml-org/ggml)
work. Full attribution is in [NOTICE](NOTICE).
