# Contributing

Bug fixes, portability work, measured kernel improvements and reproducible
hardware results are welcome.

Before opening a pull request:

1. Keep changes focused and preserve the native model equations.
2. Run `make test`, `make strict` and `make portable`.
3. Run `make sanitize` for parser, memory-management or state changes.
4. For numerical changes, report the fixed-prompt full-logit SHA and explain
   why any difference is expected.
5. For performance changes, include the exact command, model SHA, CPU, memory,
   operating system, compiler, thread settings, context, page-cache state,
   TTFT, TPOT and peak RSS. Compare repeated runs under the same conditions.

Do not commit model weights, generated binaries, private prompts, local paths
or credentials. Code contributions must be compatible with Apache-2.0 and
must preserve third-party notices.

