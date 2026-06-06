# ternary-auto-vectorizer

*Automatically lift scalar Z₃ ternary kernels to warp-parallel GPU kernels, then formally prove equivalence by exhaustive enumeration. The compiler doesn't guess — it verifies.*

## Why This Exists

Writing GPU kernels by hand is error-prone. Writing ternary GPU kernels is even more error-prone because Z₃ arithmetic (where 1+1=-1) is unintuitive. This crate does something unusual: it takes a scalar ternary function, automatically generates a warp-parallel version, and then *proves the two are equivalent* by testing every possible input combination.

For ternary operations on N inputs, exhaustive verification is feasible because there are only 3^N combinations. For N=8, that's 6,561 test cases — trivial. For N=16, it's 43 million — still feasible. This is a luxury that float operations don't have (you'd need uncountably infinite tests).

## Architecture

```
Scalar Kernel: f(trit_0, trit_1, ..., trit_N) → trit
         ↓ auto-vectorize
Warp Kernel: f::<WARP_SIZE>(packed_input) → packed_output
         ↓ formal verification
Exhaustive 3^N test: scalar_output == warp_output for ALL inputs
         ↓ if pass
Guaranteed equivalent — ship it
```

### Key Types

- **`ScalarKernel`** — A scalar ternary function (closure or function pointer)
- **`VecKernel`** — Auto-vectorized version operating on packed trit arrays
- **`VectorizationProof`** — Result of exhaustive equivalence checking
- **`ProofResult`** — Verified (all 3^N pass) or Counterexample (show failing input pair)

## Usage

```rust
use ternary_auto_vectorizer::*;

// Define a scalar ternary operation
let scalar = |inputs: &[i8]| -> i8 {
    // Some Z₃ computation
    inputs.iter().fold(0, |acc, &x| tadd(acc, x))
};

// Auto-vectorize to warp-parallel
let vec_kernel = auto_vectorize(&scalar, 8); // 8 inputs

// Formally prove equivalence
let proof = verify_equivalence(&scalar, &vec_kernel, 8);
match proof {
    ProofResult::Verified => println!("Proved equivalent over all 3^8 = 6,561 inputs"),
    ProofResult::Counterexample(inputs) => {
        panic!("Bug! Scalar and vector disagree on input: {:?}", inputs);
    }
}
```

## The Deeper Idea

This crate exploits a unique property of ternary systems: the input space is small enough to enumerate exhaustively. This means ternary compiler optimizations can be *formally verified* — not tested, not fuzzed, verified. Every single input is checked.

This is the same principle that makes `ternary-proof` (zero-knowledge proofs) tractable — the small alphabet makes exhaustive reasoning practical.

## Related Crates

- `ternary-compiler` — The broader ternary compilation pipeline
- `ternary-kernel-launch` — GPU kernel launch infrastructure
- `ternary-warp-block` — Warp-level parallel ternary operations
- `ternary-proof` — Zero-knowledge proofs using the same exhaustive enumeration
