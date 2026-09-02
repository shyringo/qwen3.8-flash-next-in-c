# Qwen3.8-Flash-Next correctness evidence

Correctness comparisons always use the exact same pinned `UD-IQ1_S` GGUF
files. They measure runtime fidelity, not the quality difference between this
low-bit checkpoint and BF16 or FP8 weights.

## Pinned model

ModelScope revision:

```text
a25519c666383714aad56d7916060aa9b315dc0e
```

| shard | bytes | SHA-256 |
|---|---:|---|
| 00001 | 10,946,624 | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd` |
| 00002 | 49,990,818,368 | `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` |
| 00003 | 22,544,696,352 | `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` |

## Fixed prompt

`tests/fixtures/qwen4_oracle_prompt.txt` encodes to:

```text
248045 846 198 116348 111764 109994 10992 248046
198 248045 74455 198 248068 271 248069 271
```

The native runtime returns this top five:

```text
2005:21.995903
109455:18.5840473
332:15.1209002
97785:14.7734909
116348:13.5392303
```

A CPU-only llama.cpp evaluator fixed at commit
`3173a56471c1753650cd806694145ffd6dcace67` returns the same five token IDs in
the same order. Across all 248,320 logits, maximum absolute difference is
0.983716965, mean absolute difference is 0.135533601, RMSE is 0.171638146 and
cosine similarity is 0.994556324. The native full-logit SHA-256 is:

```text
34db8f9d482429a2a3a10bbe600a59b7ecf6813cf9c3fcd6f356430d19ac3e36
```

The difference comes from independent low-bit dot-product and reduction
grouping. Embedding rows match exactly, and layer-boundary comparisons were
used to identify the Qwen4-specific sigmoid GDN output gate.

## Internal invariants

- Sequential and default batch-4 layer-major evaluation produce byte-identical
  full logits for the fixed 16-token prompt.
- Reusing one Q8_K activation across projections is byte-identical to repeated
  quantization.
- Expert-level dispatch, expert-union batching, selective Q8_0 repacking,
  F32 routing, IQ4_NL vector-table lookup and transposed DeltaNet state all
  preserve the native full-logit SHA byte for byte.
- Continuing with the same token after sequential and batch-4 prefill also
  produces byte-identical full logits, covering recurrent and KV state rather
  than only the last prefill output.
- A request that reaches `max_tokens` does not advance the final visible token
  solely to compute unused next-token logits. The response text and usage are
  unchanged; cross-turn replay starts from the exact post-prefill snapshot.
- Saving recurrent, PLE and attention state, advancing one token, restoring
  the state and advancing the same token again produces byte-identical logits
  across the complete 248,320-token vocabulary.
- The fixed 16-token prompt produces the same complete-logit SHA with a 2,048
  context and with a 4,096 context that actively maintains QSA index blocks.
  Saving and restoring after a completed four-token QSA block is also
  byte-exact.
- A 2,053-token continuous full-model run crosses the QSA sparse boundary in
  all twelve main attention layers. Positions 2,048 through 2,052 select
  2,049, 2,050, 2,051, 2,051 and 2,051 causal token positions respectively,
  matching the 2,048-token budget plus the four-token tail rule.
- ASan/UBSan, strict warnings-as-errors, OpenMP and portable scalar builds are
  part of release validation.
