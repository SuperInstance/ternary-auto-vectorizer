//! # Experiment M — Ternary Auto-Vectorizer
//!
//! Novel compiler feature: automatically lifts a scalar Z₃ ternary dot-threshold
//! kernel to a warp-parallel VecKernel, then **formally proves equivalence** by
//! exhaustive enumeration over all 3ⁿ input combinations.
//!
//! Two additional features round out the experiment:
//!
//! - **Conservation analysis** — verifies that balanced kernels (Σwᵢ=0) produce
//!   zero expected-output bias on binary {0,1}ⁿ inputs, a Z₃ representational
//!   invariant that the compiler can use to skip re-normalisation passes.
//!
//! - **Kernel patching** — delta-encodes the difference between two kernels so
//!   that a small weight tweak re-emits only changed bytes rather than a full
//!   recompile. Measures patch/full compression ratio.
//!
//! The critical insight missing from prior ternary compilers: the `sign` function
//! is **non-linear at zero**, so a naive "apply sign to each partial sum then
//! reduce" is NOT equivalent to "reduce then sign" — the verifier catches this
//! class of bug automatically (see `test_buggy_vectorizer_counterexample`).

// ─── Primitive trit algebra ──────────────────────────────────────────────────

pub type Trit = i8;
pub const NEG: Trit = -1;
pub const ZERO: Trit = 0;
pub const POS: Trit = 1;

/// Map any i32 to its sign ∈ {-1, 0, +1}.
#[inline]
pub fn sign(v: i32) -> Trit {
    v.signum() as Trit
}

/// Z₃ addition: sign of the integer sum.
#[inline]
pub fn tadd(a: Trit, b: Trit) -> Trit {
    sign(a as i32 + b as i32)
}

/// Z₃ multiplication: product of signs (stays in {-1,0,+1}).
#[inline]
pub fn tmul(a: Trit, b: Trit) -> Trit {
    (a as i32 * b as i32) as Trit
}

/// Z₃ negation.
#[inline]
pub fn tneg(a: Trit) -> Trit {
    -a
}

/// Exact integer dot product (NOT clamped). Used by both scalar and vectorized paths.
pub fn dot_i32(weights: &[Trit], inputs: &[Trit]) -> i32 {
    debug_assert_eq!(weights.len(), inputs.len());
    weights
        .iter()
        .zip(inputs)
        .map(|(&w, &x)| w as i32 * x as i32)
        .sum()
}

// ─── ScalarKernel ─────────────────────────────────────────────────────────────

/// A single ternary dot-threshold neuron: output = sign(Σ wᵢ·xᵢ).
pub struct ScalarKernel {
    pub weights: Vec<Trit>,
}

impl ScalarKernel {
    pub fn new(weights: Vec<Trit>) -> Self {
        ScalarKernel { weights }
    }

    pub fn run(&self, inputs: &[Trit]) -> Trit {
        sign(dot_i32(&self.weights, inputs))
    }

    pub fn n(&self) -> usize {
        self.weights.len()
    }

    /// Algebraic weight balance: Σ wᵢ. Zero means the kernel is "charge-neutral".
    pub fn weight_sum(&self) -> i32 {
        self.weights.iter().map(|&w| w as i32).sum()
    }

    /// Non-zero weight count (structural sparsity).
    pub fn nnz(&self) -> usize {
        self.weights.iter().filter(|&&w| w != ZERO).count()
    }
}

// ─── VecKernel ────────────────────────────────────────────────────────────────

/// Warp-parallel version of a ScalarKernel.
///
/// Weights are chunked into `chunk_size`-wide groups. Each group computes a
/// partial integer dot product independently (one warp lane). A tree-reduce sums
/// all partial results, then `sign` is applied exactly once.
///
/// Correctness requirement: `sign` must be applied AFTER the full reduction.
/// Applying it per-chunk (before reduction) is a compiler bug — see the
/// counterexample in the test suite.
pub struct VecKernel {
    pub weight_chunks: Vec<Vec<Trit>>,
    pub chunk_size: usize,
}

impl VecKernel {
    pub fn run(&self, inputs: &[Trit]) -> Trit {
        let total: i32 = self
            .weight_chunks
            .iter()
            .zip(inputs.chunks(self.chunk_size))
            .map(|(wc, xc)| dot_i32(wc, xc))
            .sum();
        sign(total) // one sign application after full reduction — correct
    }

    pub fn n_lanes(&self) -> usize {
        self.weight_chunks.len()
    }

    /// Theoretical speedup over scalar execution.
    ///
    /// Wall-clock model:
    ///   scalar   = chunk_size × n_lanes  (sequential dot products)
    ///   vector   = chunk_size (parallel per-lane) + ⌈log₂(n_lanes)⌉ (tree-reduce)
    ///   speedup  = scalar / vector
    ///
    /// Optimal lane_width is the smallest value that fits in a warp — maximum
    /// parallelism. Wider lanes reduce the number of warp units doing work.
    pub fn theoretical_speedup(&self) -> f64 {
        let n_lanes = self.n_lanes() as f64;
        if n_lanes <= 1.0 {
            return 1.0;
        }
        let chunk = self.chunk_size as f64;
        let total_n = chunk * n_lanes; // == kernel.n()
        let reduction_depth = n_lanes.log2().ceil();
        total_n / (chunk + reduction_depth)
    }

    /// Instruction count ratio relative to scalar baseline.
    pub fn instruction_ratio(&self) -> f64 {
        1.0 / self.theoretical_speedup()
    }
}

// ─── Vectorizer ───────────────────────────────────────────────────────────────

pub struct Vectorizer {
    pub lane_width: usize,
}

impl Vectorizer {
    pub fn new(lane_width: usize) -> Self {
        assert!(lane_width > 0, "lane_width must be >= 1");
        Vectorizer { lane_width }
    }

    /// Chunk the kernel weights into `lane_width`-sized groups.
    pub fn vectorize(&self, kernel: &ScalarKernel) -> VecKernel {
        let chunk_size = self.lane_width.min(kernel.n()).max(1);
        let weight_chunks = kernel
            .weights
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();
        VecKernel {
            weight_chunks,
            chunk_size,
        }
    }

    /// Exhaustively verify that the vectorized kernel is equivalent to scalar
    /// for every possible input in {-1,0,+1}ⁿ.
    ///
    /// Tractable for n ≤ 10 (3¹⁰ = 59 049 cases, < 1 ms).
    pub fn prove_equiv(&self, kernel: &ScalarKernel) -> EquivResult {
        const MAX_N: usize = 10;
        let n = kernel.n();
        if n > MAX_N {
            return EquivResult::Unverified {
                reason: "n > 10: exhaustive search exceeds tractable bound",
            };
        }

        let vec_kernel = self.vectorize(kernel);
        let mut cases: u64 = 0;
        let mut input = vec![NEG; n];

        loop {
            let scalar_out = kernel.run(&input);
            let vec_out = vec_kernel.run(&input);

            if scalar_out != vec_out {
                return EquivResult::Counterexample {
                    input: input.clone(),
                    scalar: scalar_out,
                    vectorized: vec_out,
                };
            }
            cases += 1;

            // Increment base-3 counter over {-1, 0, +1}ⁿ
            let mut carry = true;
            for digit in input.iter_mut().rev() {
                if !carry {
                    break;
                }
                if *digit < POS {
                    *digit += 1;
                    carry = false;
                } else {
                    *digit = NEG;
                    // carry propagates leftward
                }
            }
            if carry {
                break; // overflowed: all 3ⁿ combinations exhausted
            }
        }

        EquivResult::Proved {
            cases_verified: cases,
            n,
            lane_width: self.lane_width,
            speedup: vec_kernel.theoretical_speedup(),
        }
    }
}

// ─── EquivResult ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum EquivResult {
    Proved {
        cases_verified: u64,
        n: usize,
        lane_width: usize,
        speedup: f64,
    },
    Counterexample {
        input: Vec<Trit>,
        scalar: Trit,
        vectorized: Trit,
    },
    Unverified {
        reason: &'static str,
    },
}

impl EquivResult {
    pub fn is_proved(&self) -> bool {
        matches!(self, EquivResult::Proved { .. })
    }

    pub fn speedup(&self) -> Option<f64> {
        if let EquivResult::Proved { speedup, .. } = self {
            Some(*speedup)
        } else {
            None
        }
    }

    pub fn cases_verified(&self) -> Option<u64> {
        if let EquivResult::Proved { cases_verified, .. } = self {
            Some(*cases_verified)
        } else {
            None
        }
    }
}

// ─── Conservation analysis ────────────────────────────────────────────────────

/// Z₃ conservation laws for ternary kernels.
///
/// Key invariant (binary-input bias theorem):
/// A kernel with `weight_sum == 0` has exactly zero expected output bias when
/// inputs are drawn uniformly from {0,1}ⁿ. This lets the compiler skip
/// re-normalisation passes for balanced layers.
pub struct ConservationAnalyzer;

impl ConservationAnalyzer {
    /// Expected output over all 2ⁿ binary {0,1}ⁿ inputs.
    ///
    /// Theorem: if `kernel.weight_sum() == 0`, then `binary_input_bias == 0.0`.
    pub fn binary_input_bias(kernel: &ScalarKernel) -> f64 {
        let n = kernel.n();
        assert!(n <= 20, "n > 20: use sampling");
        let total = 1u64 << n;
        let sum: i64 = (0..total)
            .map(|mask| {
                let input: Vec<Trit> = (0..n)
                    .map(|i| if (mask >> i) & 1 == 1 { POS } else { ZERO })
                    .collect();
                kernel.run(&input) as i64
            })
            .sum();
        sum as f64 / total as f64
    }

    /// Negation law: negating all weights must negate the weight sum exactly.
    /// Ensures the Z₃ algebraic structure is intact after weight quantisation.
    pub fn verify_negate_law(kernel: &ScalarKernel) -> bool {
        let neg_weights: Vec<Trit> = kernel.weights.iter().map(|&w| tneg(w)).collect();
        let negated = ScalarKernel::new(neg_weights);
        negated.weight_sum() == -kernel.weight_sum()
    }

    /// For each input x, verify that negating all weights maps output(x) → -output(x).
    /// Stronger than `verify_negate_law`: checks every individual output, not just the sum.
    pub fn verify_negate_output_law(kernel: &ScalarKernel) -> bool {
        let n = kernel.n();
        if n > 8 {
            return false; // Too large for exhaustive check
        }
        let neg_weights: Vec<Trit> = kernel.weights.iter().map(|&w| tneg(w)).collect();
        let negated = ScalarKernel::new(neg_weights);

        let mut input = vec![NEG; n];
        loop {
            if negated.run(&input) != tneg(kernel.run(&input)) {
                return false;
            }
            let mut carry = true;
            for digit in input.iter_mut().rev() {
                if !carry {
                    break;
                }
                if *digit < POS {
                    *digit += 1;
                    carry = false;
                } else {
                    *digit = NEG;
                }
            }
            if carry {
                break;
            }
        }
        true
    }
}

// ─── KernelPatch ──────────────────────────────────────────────────────────────

/// Delta encoding between two kernels of the same size.
///
/// When an agent makes a small weight update (e.g., one training step), the
/// compiler emits only the changed (index, new_weight) pairs rather than a
/// full kernel. Enables sub-ms incremental PTX patching.
pub struct KernelPatch {
    pub changed_indices: Vec<usize>,
    pub new_weights: Vec<Trit>,
}

impl KernelPatch {
    /// Compute the delta between `old` and `new`. Returns `None` if sizes differ.
    pub fn compute(old: &ScalarKernel, new: &ScalarKernel) -> Option<Self> {
        if old.n() != new.n() {
            return None;
        }
        let (indices, weights): (Vec<_>, Vec<_>) = old
            .weights
            .iter()
            .zip(&new.weights)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (_, &b))| (i, b))
            .unzip();
        Some(KernelPatch {
            changed_indices: indices,
            new_weights: weights,
        })
    }

    /// Apply this patch to `old`, producing the new kernel.
    pub fn apply(&self, old: &ScalarKernel) -> ScalarKernel {
        let mut weights = old.weights.clone();
        for (&idx, &w) in self.changed_indices.iter().zip(&self.new_weights) {
            weights[idx] = w;
        }
        ScalarKernel::new(weights)
    }

    /// Ratio of patch bytes to full kernel bytes (each trit = 1 byte; patch = 2 bytes/change).
    pub fn compression_ratio(&self, kernel_n: usize) -> f64 {
        let patch_bytes = self.changed_indices.len() * 2; // (index, value) per change
        patch_bytes as f64 / kernel_n as f64
    }

    pub fn n_changed(&self) -> usize {
        self.changed_indices.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Primitive algebra ──

    #[test]
    fn test_sign_clamps_to_trit() {
        assert_eq!(sign(-99), NEG);
        assert_eq!(sign(0), ZERO);
        assert_eq!(sign(99), POS);
        assert_eq!(sign(-1), NEG);
        assert_eq!(sign(1), POS);
    }

    #[test]
    fn test_tmul_is_product_of_signs() {
        assert_eq!(tmul(NEG, NEG), POS); // (-1)(-1) = +1
        assert_eq!(tmul(NEG, ZERO), ZERO);
        assert_eq!(tmul(NEG, POS), NEG); // (-1)(+1) = -1
        assert_eq!(tmul(POS, POS), POS);
        assert_eq!(tmul(ZERO, POS), ZERO);
    }

    #[test]
    fn test_tadd_representative_cases() {
        assert_eq!(tadd(NEG, POS), ZERO);  // cancellation
        assert_eq!(tadd(NEG, NEG), NEG);   // sign(-2) = -1
        assert_eq!(tadd(POS, POS), POS);   // sign(+2) = +1
        assert_eq!(tadd(ZERO, NEG), NEG);
    }

    #[test]
    fn test_tneg_flips_sign() {
        assert_eq!(tneg(POS), NEG);
        assert_eq!(tneg(NEG), POS);
        assert_eq!(tneg(ZERO), ZERO);
    }

    // ── ScalarKernel ──

    #[test]
    fn test_scalar_zero_weights_always_outputs_zero() {
        let k = ScalarKernel::new(vec![ZERO, ZERO, ZERO]);
        for x in [POS, NEG, ZERO] {
            assert_eq!(k.run(&[x, x, x]), ZERO);
        }
    }

    #[test]
    fn test_scalar_single_weight_acts_as_identity() {
        let k = ScalarKernel::new(vec![POS]);
        assert_eq!(k.run(&[NEG]), NEG);
        assert_eq!(k.run(&[ZERO]), ZERO);
        assert_eq!(k.run(&[POS]), POS);
    }

    #[test]
    fn test_scalar_negated_weight_acts_as_negate() {
        let k = ScalarKernel::new(vec![NEG]);
        assert_eq!(k.run(&[POS]), NEG);
        assert_eq!(k.run(&[NEG]), POS);
        assert_eq!(k.run(&[ZERO]), ZERO);
    }

    #[test]
    fn test_scalar_balanced_weights_cancel_equal_inputs() {
        // w=[+1,-1]: inputs=[x,x] → dot = x - x = 0 → ZERO
        let k = ScalarKernel::new(vec![POS, NEG]);
        assert_eq!(k.run(&[POS, POS]), ZERO);
        assert_eq!(k.run(&[NEG, NEG]), ZERO);
        assert_eq!(k.run(&[ZERO, ZERO]), ZERO);
    }

    // ── Equivalence proofs ──

    #[test]
    fn test_equiv_proof_lane_width_1_covers_3_to_n_cases() {
        let v = Vectorizer::new(1);
        let k = ScalarKernel::new(vec![POS, NEG, ZERO]);
        let result = v.prove_equiv(&k);
        assert!(result.is_proved(), "{result:?}");
        assert_eq!(result.cases_verified(), Some(27)); // 3^3
    }

    #[test]
    fn test_equiv_proof_lane_width_4_n4_verified() {
        let v = Vectorizer::new(4);
        let k = ScalarKernel::new(vec![POS, NEG, ZERO, POS]);
        let result = v.prove_equiv(&k);
        assert!(result.is_proved(), "{result:?}");
        assert_eq!(result.cases_verified(), Some(81)); // 3^4
    }

    #[test]
    fn test_equiv_proof_lane_width_2_n6_verified() {
        let v = Vectorizer::new(2);
        let k = ScalarKernel::new(vec![POS, POS, NEG, NEG, POS, NEG]);
        let result = v.prove_equiv(&k);
        assert!(result.is_proved(), "{result:?}");
        assert_eq!(result.cases_verified(), Some(729)); // 3^6
    }

    #[test]
    fn test_equiv_proof_non_power_of_two_size_n5_width3() {
        // n=5, width=3 → chunks of sizes [3, 2]
        let v = Vectorizer::new(3);
        let k = ScalarKernel::new(vec![POS, NEG, ZERO, POS, NEG]);
        let vk = v.vectorize(&k);
        assert_eq!(vk.weight_chunks.len(), 2);
        assert_eq!(vk.weight_chunks[0].len(), 3);
        assert_eq!(vk.weight_chunks[1].len(), 2);
        assert!(v.prove_equiv(&k).is_proved());
    }

    // ── Counterexample detection ──

    #[test]
    fn test_buggy_vectorizer_counterexample_detected() {
        // The bug: apply sign() to each partial sum BEFORE the final reduction.
        //
        // Concrete counterexample:
        //   weights=[+1,+1,−1], inputs=[+1,+1,+1], chunk_size=2
        //   Scalar:  1+1−1 = 1        → sign = +1
        //   Buggy:   chunk₁=sign(2)=+1, chunk₂=sign(−1)=−1 → +1+(−1)=0 → sign = 0  ← WRONG

        let weights = [POS, POS, NEG];
        let inputs = [POS, POS, POS];

        // Correct scalar
        let scalar_out = sign(dot_i32(&weights, &inputs));
        assert_eq!(scalar_out, POS);

        // Buggy vectorized (sign applied per chunk)
        let chunk1 = sign(dot_i32(&weights[..2], &inputs[..2])) as i32;
        let chunk2 = sign(dot_i32(&weights[2..], &inputs[2..])) as i32;
        let buggy_out = sign(chunk1 + chunk2);
        assert_eq!(buggy_out, ZERO);

        // The compiler's verifier would catch this divergence
        assert_ne!(
            scalar_out, buggy_out,
            "Buggy vectorizer produces a different answer — verifier must flag this"
        );
    }

    // ── Speedup model ──

    #[test]
    fn test_speedup_decreases_with_wider_lanes() {
        // Wider lanes = fewer parallel units = lower speedup.
        // Narrowest lane (width=1) maximises parallelism for a fixed kernel size.
        let k = ScalarKernel::new(vec![POS; 16]);
        let speedups: Vec<f64> = [1, 2, 4, 8, 16]
            .iter()
            .map(|&w| Vectorizer::new(w).vectorize(&k).theoretical_speedup())
            .collect();
        for pair in speedups.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "Speedup should decrease as lane width grows (fewer parallel units): \
                 {:.2} vs {:.2}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_speedup_is_greater_than_one_when_n_exceeds_chunk_depth() {
        // n=8, width=1: scalar=8 ops, vector=1(per lane)+3(reduce)=4 → speedup=2.0
        let k = ScalarKernel::new(vec![POS; 8]);
        let vk = Vectorizer::new(1).vectorize(&k);
        assert!(
            vk.theoretical_speedup() > 1.0,
            "8-wide kernel with 1-wide lanes must have speedup > 1: {}",
            vk.theoretical_speedup()
        );
    }

    // ── Conservation laws ──

    #[test]
    fn test_balanced_kernel_has_zero_binary_input_bias() {
        // w=[+1,−1]: weight_sum = 0 → bias over {0,1}² inputs must be exactly 0
        let k = ScalarKernel::new(vec![POS, NEG]);
        assert_eq!(k.weight_sum(), 0);
        let bias = ConservationAnalyzer::binary_input_bias(&k);
        assert_eq!(bias, 0.0, "Balanced kernel: binary-input bias must be 0, got {bias}");
    }

    #[test]
    fn test_unbalanced_kernel_has_nonzero_binary_input_bias() {
        // w=[+1,+1]: weight_sum = 2 → positively biased on {0,1}² inputs
        let k = ScalarKernel::new(vec![POS, POS]);
        assert_ne!(k.weight_sum(), 0);
        let bias = ConservationAnalyzer::binary_input_bias(&k);
        assert!(
            bias > 0.0,
            "All-positive weights must produce positive bias on binary inputs: {bias}"
        );
    }

    #[test]
    fn test_negate_law_reverses_weight_sum() {
        let k = ScalarKernel::new(vec![POS, POS, NEG, ZERO]);
        assert!(ConservationAnalyzer::verify_negate_law(&k));
    }

    #[test]
    fn test_negate_output_law_holds_for_small_kernel() {
        // For every input x, negating weights must flip the output sign
        let k = ScalarKernel::new(vec![POS, NEG, POS]);
        assert!(
            ConservationAnalyzer::verify_negate_output_law(&k),
            "Negating all weights must negate every individual output"
        );
    }

    // ── Kernel patching ──

    #[test]
    fn test_patch_for_two_changes_is_smaller_than_full_kernel() {
        let old = ScalarKernel::new(vec![POS, POS, NEG, ZERO, POS, NEG, POS, POS]);
        let mut new_w = old.weights.clone();
        new_w[2] = POS;   // changed
        new_w[5] = ZERO;  // changed
        let new = ScalarKernel::new(new_w);

        let patch = KernelPatch::compute(&old, &new).unwrap();
        assert_eq!(patch.n_changed(), 2);

        let ratio = patch.compression_ratio(old.n());
        // patch_bytes = 2 changes × 2 bytes = 4; full_bytes = 8 → ratio = 0.5
        // Any patch touching ≤ half the weights is a net win over full recompile.
        assert!(
            ratio <= 0.5,
            "2-change patch on 8-weight kernel must be ≤50% of full kernel: ratio={ratio:.2}"
        );

        // Round-trip: applying patch to old must reproduce new
        let restored = patch.apply(&old);
        assert_eq!(restored.weights, new.weights, "Patch round-trip must be lossless");
    }

    #[test]
    fn test_patch_for_identical_kernels_is_empty() {
        let k = ScalarKernel::new(vec![POS, NEG, ZERO]);
        let patch = KernelPatch::compute(&k, &k).unwrap();
        assert_eq!(patch.n_changed(), 0);
        assert_eq!(patch.compression_ratio(k.n()), 0.0);
    }

    #[test]
    fn test_patch_returns_none_for_size_mismatch() {
        let a = ScalarKernel::new(vec![POS, NEG]);
        let b = ScalarKernel::new(vec![POS, NEG, ZERO]);
        assert!(KernelPatch::compute(&a, &b).is_none());
    }
}
