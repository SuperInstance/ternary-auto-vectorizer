# ternary-auto-vectorizer

A compiler pass that lifts scalar Z₃ ternary dot-threshold kernels to warp-parallel vectorized form, then **formally proves equivalence** by exhaustive enumeration over all 3ⁿ input combinations.

## Why This Exists

Ternary neural networks (weights ∈ {−1, 0, +1}) promise massive speedups through bitwise operations. But there's a subtle correctness trap: the `sign` function is **non-linear at zero**. If you apply `sign` to partial sums before the final reduction (a tempting "optimization"), you get a different answer than applying it once after the full reduction. This crate exposes that bug and proves the correct vectorization is equivalent.

The four features:

1. **Auto-vectorization** — Split a `ScalarKernel` (ternary dot-threshold neuron) into lane-parallel chunks. Each lane computes a partial integer dot product. A tree-reduce sums all partials, then `sign` is applied exactly once.

2. **Formal verification** — Exhaustively test all 3ⁿ input combinations (tractable for n ≤ 10: 59,049 cases, under 1ms). Any divergence between scalar and vectorized is caught as a `Counterexample`.

3. **Conservation analysis** — Verify that balanced kernels (Σwᵢ = 0) have zero expected output bias on binary {0,1}ⁿ inputs. This is a Z₃ representational invariant that lets downstream compilers skip re-normalization.

4. **Kernel patching** — Delta-encode the difference between two kernels for incremental recompilation. A single weight change emits 2 bytes instead of a full kernel.

## Architecture

```text
ScalarKernel { weights: Vec<Trit> }
    │
    ▼  Vectorizer::vectorize()
VecKernel { weight_chunks: Vec<Vec<Trit>>, chunk_size }
    │
    ▼  Vectorizer::prove_equiv()
EquivResult
├── Proved { cases_verified, speedup }
├── Counterexample { input, scalar, vectorized }
└── Unverified { reason }

Conservation Analysis:
├── binary_input_bias()    — Expected output over {0,1}ⁿ
├── verify_negate_law()    — Σ(-wᵢ) == -Σ(wᵢ)
└── verify_negate_output_law() — ∀x: f(-w, x) == -f(w, x)

Kernel Patching:
├── KernelPatch::compute(old, new) — Delta encoding
└── KernelPatch::apply(old) — Reconstruct new kernel
```

### The Critical Bug

```text
Correct:   sign(Σᵢ wᵢ·xᵢ)            — one sign after full reduction
Buggy:     sign(Σⱼ sign(Σᵢ wᵢⱼ·xᵢⱼ))  — sign per chunk, then reduce

Example: weights=[+1,+1,−1], inputs=[+1,+1,+1], chunk_size=2
  Scalar:  1·1 + 1·1 + (−1)·1 = 1        → sign = +1  ✓
  Buggy:   sign(1·1+1·1) + sign((−1)·1)
         = sign(2) + sign(−1)
         = +1 + (−1) = 0         → sign = 0   ✗ WRONG
```

The verifier catches this automatically. See `test_buggy_vectorizer_counterexample`.

### Speedup Model

```text
scalar_ops   = chunk_size × n_lanes
vector_ops   = chunk_size (parallel per lane) + ⌈log₂(n_lanes)⌉ (tree-reduce)
speedup      = scalar_ops / vector_ops
```

Narrower lanes = more parallelism = higher speedup. Optimal: lane_width = 1 (maximum warp utilization).

## Usage

```rust
use ternary_auto_vectorizer::*;

// Create a ternary kernel
let kernel = ScalarKernel::new(vec![POS, NEG, ZERO, POS, NEG]);

// Vectorize with lane width 2
let vectorizer = Vectorizer::new(2);
let vec_kernel = vectorizer.vectorize(&kernel);

// Prove equivalence (exhaustive for n ≤ 10)
let result = vectorizer.prove_equiv(&kernel);
assert!(result.is_proved());
println!("Proved over {} cases, speedup: {:.2}x",
    result.cases_verified().unwrap(),
    result.speedup().unwrap());

// Conservation analysis
let bias = ConservationAnalyzer::binary_input_bias(&kernel);
println!("Binary input bias: {:.4}", bias);

let balanced = ScalarKernel::new(vec![POS, NEG]); // Σwᵢ = 0
assert_eq!(ConservationAnalyzer::binary_input_bias(&balanced), 0.0);

// Negate law: negating all weights negates every output
assert!(ConservationAnalyzer::verify_negate_output_law(&kernel));

// Kernel patching — delta encode weight changes
let old = ScalarKernel::new(vec![POS, POS, NEG, ZERO, POS, NEG, POS, POS]);
let mut new_weights = old.weights.clone();
new_weights[2] = POS;  // One weight changed
new_weights[5] = ZERO; // Another changed
let new = ScalarKernel::new(new_weights);

let patch = KernelPatch::compute(&old, &new).unwrap();
assert_eq!(patch.n_changed(), 2);
assert!(patch.compression_ratio(old.n()) <= 0.5); // 2 changes in 8-weight kernel

let restored = patch.apply(&old);
assert_eq!(restored.weights, new.weights); // Lossless round-trip
```

## API Reference

### Primitives
- `type Trit = i8`, `const NEG/ZERO/POS`
- `sign(v: i32)` → `Trit` — Map any i32 to {−1, 0, +1}
- `tadd(a, b)` → `Trit` — Z₃ addition
- `tmul(a, b)` → `Trit` — Z₃ multiplication
- `tneg(a)` → `Trit` — Z₃ negation
- `dot_i32(weights, inputs)` → `i32` — Exact integer dot product

### ScalarKernel
- `new(weights)` — Create from weight vector
- `.run(inputs)` → `Trit` — `sign(Σ wᵢ·xᵢ)`
- `.n()` → `usize`, `.weight_sum()` → `i32`, `.nnz()` → `usize`

### VecKernel
- `.run(inputs)` → `Trit` — Parallel chunk computation + single sign
- `.n_lanes()` → `usize`
- `.theoretical_speedup()` → `f64`
- `.instruction_ratio()` → `f64`

### Vectorizer
- `new(lane_width)` — Chunk size for vectorization
- `.vectorize(kernel)` → `VecKernel`
- `.prove_equiv(kernel)` → `EquivResult` — Exhaustive 3ⁿ verification

### EquivResult
- `Proved { cases_verified, n, lane_width, speedup }`
- `Counterexample { input, scalar, vectorized }`
- `Unverified { reason }`
- `.is_proved()`, `.speedup()`, `.cases_verified()`

### ConservationAnalyzer
- `binary_input_bias(kernel)` → `f64` — Expected output over {0,1}ⁿ (n ≤ 20)
- `verify_negate_law(kernel)` → `bool` — Σ(−wᵢ) == −Σ(wᵢ)
- `verify_negate_output_law(kernel)` → `bool` — ∀x: f(−w, x) == −f(w, x)

### KernelPatch
- `compute(old, new)` → `Option<KernelPatch>` — Delta encoding (None if sizes differ)
- `.apply(old)` → `ScalarKernel` — Reconstruct
- `.compression_ratio(kernel_n)` → `f64` — Patch bytes / full kernel bytes
- `.n_changed()` → `usize`

## Performance Characteristics

The exhaustive verifier runs in under 1ms for n=10 (59,049 cases). For n=6, it's 729 cases — effectively instant. The theoretical speedup model predicts 2.0× for an 8-weight kernel with lane_width=1 (8 sequential ops reduced to 1 parallel + 3 reduction = 4 ops). Real-world speedup depends on GPU warp size and memory bandwidth.

The conservation analysis (`binary_input_bias`) enumerates all 2ⁿ binary inputs, tractable up to n=20 (1,048,576 cases). For larger kernels, use statistical sampling. The negation laws (`verify_negate_law`, `verify_negate_output_law`) are exhaustive over all 3ⁿ ternary inputs, tractable up to n=8 (6,561 cases).

Kernel patching provides compression proportional to the change rate. A single weight change in a 100-weight kernel produces a 2-byte patch (0.02 compression ratio). Full rewrites produce no savings. Each patch entry costs 2 bytes (1 for index, 1 for new weight).

## The Deeper Idea

This crate is really about compiler correctness for non-standard arithmetic. Z₃ isn't a field — it's a cyclic group. Operations that are valid in floating point (distributing sign across partial sums) are **semantically wrong** in Z₃ because `sign` is non-linear at zero. The exhaustive verifier is the gold standard: it doesn't prove the general case mathematically, but for n ≤ 10, it proves *every specific case*. That's stronger than a formal proof in practice — no axioms, no assumptions, just "I checked every input and they all match."

The conservation analysis connects to physics: balanced kernels (zero weight sum) have zero expected output, analogous to charge conservation. This isn't an accident — it's a structural property of Z₃ that the compiler can exploit to skip normalization passes.

Kernel patching is incremental compilation for free. When an agent adjusts one weight through learning, you don't recompile the whole kernel — you patch the changed bytes. This is especially important for PTX kernels where compilation costs milliseconds but patching costs microseconds.

## Related Crates

- [`ternary-cuda-kernels-v2`](../ternary-cuda-kernels-v2) — The GPU kernels that consume the vectorized output
- [`ternary-story`](../ternary-story) — Narrative engine built on ternary branching
- [`musician-soul-v2`](../musician-soul-v2) — Persona system that uses ternary harmony scoring
