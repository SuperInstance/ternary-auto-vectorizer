# ternary-auto-vectorizer

Automatically lifts scalar Z₃ ternary operations to warp-level parallel versions. Proves vectorized output is equivalent to scalar input — a real compiler optimization pass for ternary GPU code.

## Overview

# Experiment M — Ternary Auto-Vectorizer

Novel compiler feature: automatically lifts a scalar Z₃ ternary dot-threshold

## Stats

- **Tests**: 22
- **LOC**: 662
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use ternary_auto_vectorizer::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
