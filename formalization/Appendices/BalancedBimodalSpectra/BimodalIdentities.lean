/-
Appendix "Balanced bimodal spectra", Lemma (bimodal identities)
(label `lem:bimodal-identities`), together with the note following it.

For `λ ∈ Λ_m`, abbreviating `ℐ = ℐ(λ)` and `C = C(λ)`:

1. `‖λ₊‖² = ½(1+ℐ)` and `‖λ₋‖² = ½(1-ℐ)`;
2. `2‖λ₊‖‖λ₋‖ = √(1-ℐ²)`;
3. `‖λ₊-λ₋‖² = 1 - 2C`, and hence `𝒫(λ) = 1 - √(1-2C)`;
4. `Σ_j ((λ₊)_j + i(λ₋)_j)² = ℐ + 2iC`.

Note following the lemma: `0 ≤ 2C ≤ 1` (`C ≥ 0` since the entries of
`λ±` are nonnegative, and `2C ≤ 1` by part (3)), so `0 ≤ 𝒫 ≤ 1`.

Each statement carries the paper's hypothesis `λ ∈ Λ_m`; where a part
is an identity that holds without the hypothesis, the hypothesis is
named `_hx`.
-/
import Appendices.BalancedBimodalSpectra.Defs

namespace Appendices

open BigOperators

variable {m : ℕ} {x : Fin (2 * m) → ℝ}

/-! ### Index bookkeeping

The index set of `ℝ^{2m}` splits into the first `m` indices and their
reversals; the entries of `λ₋` and `λ₊` are, up to sign, the entries
of `λ` at these two blocks. -/

/-- Splitting a sum over `Fin (2m)` into the first `m` indices and the
reversed first `m` indices. -/
lemma sum_split {m : ℕ} (g : Fin (2 * m) → ℝ) :
    ∑ i, g i
      = (∑ j : Fin m, g ⟨j, by omega⟩)
        + ∑ j : Fin m, g (Fin.rev ⟨j, by omega⟩) := by
  have hmm : m + m = 2 * m := by omega
  rw [← Fintype.sum_equiv (finCongr hmm) (fun i => g (finCongr hmm i)) g
    (fun i => rfl), Fin.sum_univ_add]
  congr 1
  refine Fintype.sum_equiv Fin.revPerm _ _ fun j => ?_
  have hj := j.isLt
  congr 1
  ext
  simp only [finCongr_apply, Fin.val_cast, Fin.val_natAdd, Fin.revPerm_apply,
    Fin.val_rev]
  omega

/-- `‖λ₊‖² + ‖λ₋‖² = ‖λ‖²`: the entries of `λ₊` and `λ₋` are, up to
sign, the entries of `λ`. -/
lemma sum_sq_lamPlus_add_sum_sq_lamMinus (x : Fin (2 * m) → ℝ) :
    (∑ j, lamPlus x j ^ 2) + ∑ j, lamMinus x j ^ 2 = ∑ i, x i ^ 2 := by
  rw [sum_split (fun i => x i ^ 2), add_comm]
  congr 1
  exact Finset.sum_congr rfl fun j _ => neg_sq _

/-! ### Entry signs on `Λ_m` -/

/-- Entries at indices `< m` are negative. -/
lemma MemLambdaSet.entry_neg (hx : MemLambdaSet m x) (i : Fin (2 * m))
    (hi : (i : ℕ) < m) : x i < 0 := by
  have hm := hx.m_pos
  refine lt_of_le_of_lt (hx.mono ?_) (hx.middle_neg hm)
  rw [Fin.le_def]
  show (i : ℕ) ≤ m - 1
  omega

/-- Entries at indices `≥ m` are positive. -/
lemma MemLambdaSet.entry_pos (hx : MemLambdaSet m x) (i : Fin (2 * m))
    (hi : m ≤ (i : ℕ)) : 0 < x i := by
  have hm := hx.m_pos
  refine lt_of_lt_of_le (hx.middle_pos hm) (hx.mono ?_)
  rw [Fin.le_def]
  show m ≤ (i : ℕ)
  exact hi

/-- The entries of `λ₊` are positive. -/
lemma MemLambdaSet.lamPlus_pos (hx : MemLambdaSet m x) (j : Fin m) :
    0 < lamPlus x j := by
  have hj := j.isLt
  refine hx.entry_pos _ ?_
  rw [Fin.val_rev]
  show m ≤ 2 * m - ((j : ℕ) + 1)
  omega

/-- The entries of `λ₋` are positive. -/
lemma MemLambdaSet.lamMinus_pos (hx : MemLambdaSet m x) (j : Fin m) :
    0 < lamMinus x j := by
  have hj := j.isLt
  have h := hx.entry_neg ⟨j, by omega⟩ (by show (j : ℕ) < m; exact hj)
  exact neg_pos.mpr h

/-! ### Lemma (bimodal identities) -/

/-- Part (1): `‖λ₊‖² = ½(1+ℐ)`. -/
theorem sum_sq_lamPlus_eq (hx : MemLambdaSet m x) :
    ∑ j, lamPlus x j ^ 2 = (1 + imbalance x) / 2 := by
  have h := sum_sq_lamPlus_add_sum_sq_lamMinus x
  rw [hx.unit] at h
  rw [imbalance] at *
  linarith

/-- Part (1): `‖λ₋‖² = ½(1-ℐ)`. -/
theorem sum_sq_lamMinus_eq (hx : MemLambdaSet m x) :
    ∑ j, lamMinus x j ^ 2 = (1 - imbalance x) / 2 := by
  have h := sum_sq_lamPlus_add_sum_sq_lamMinus x
  rw [hx.unit] at h
  rw [imbalance] at *
  linarith

/-- Part (2): `2‖λ₊‖‖λ₋‖ = √(1-ℐ²)`. -/
theorem two_mul_norm_lamPlus_mul_norm_lamMinus (hx : MemLambdaSet m x) :
    2 * Real.sqrt (∑ j, lamPlus x j ^ 2)
      * Real.sqrt (∑ j, lamMinus x j ^ 2)
      = Real.sqrt (1 - imbalance x ^ 2) := by
  have hp : (0 : ℝ) ≤ ∑ j, lamPlus x j ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  have hm : (0 : ℝ) ≤ ∑ j, lamMinus x j ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  have key : (2 * Real.sqrt (∑ j, lamPlus x j ^ 2)
      * Real.sqrt (∑ j, lamMinus x j ^ 2)) ^ 2 = 1 - imbalance x ^ 2 := by
    have hsq : (2 * Real.sqrt (∑ j, lamPlus x j ^ 2)
        * Real.sqrt (∑ j, lamMinus x j ^ 2)) ^ 2
        = 4 * (∑ j, lamPlus x j ^ 2) * ∑ j, lamMinus x j ^ 2 := by
      rw [mul_pow, mul_pow, Real.sq_sqrt hp, Real.sq_sqrt hm]
      ring
    rw [hsq, sum_sq_lamPlus_eq hx, sum_sq_lamMinus_eq hx]
    ring
  calc 2 * Real.sqrt (∑ j, lamPlus x j ^ 2)
        * Real.sqrt (∑ j, lamMinus x j ^ 2)
      = Real.sqrt ((2 * Real.sqrt (∑ j, lamPlus x j ^ 2)
          * Real.sqrt (∑ j, lamMinus x j ^ 2)) ^ 2) :=
        (Real.sqrt_sq (by positivity)).symm
    _ = Real.sqrt (1 - imbalance x ^ 2) := by rw [key]

/-- Part (3): `‖λ₊-λ₋‖² = 1 - 2C`. -/
theorem sum_sq_lamPlus_sub_lamMinus (hx : MemLambdaSet m x) :
    ∑ j, (lamPlus x j - lamMinus x j) ^ 2 = 1 - 2 * statC x := by
  have hsum : (∑ j, lamPlus x j ^ 2) + ∑ j, lamMinus x j ^ 2 = 1 := by
    rw [sum_sq_lamPlus_add_sum_sq_lamMinus x, hx.unit]
  have expand : ∑ j, (lamPlus x j - lamMinus x j) ^ 2
      = ((∑ j, lamPlus x j ^ 2) + ∑ j, lamMinus x j ^ 2)
        - 2 * ∑ j, lamPlus x j * lamMinus x j := by
    rw [← Finset.sum_add_distrib, Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [expand, hsum, statC]

/-- Part (3), continued: `𝒫(λ) = 1 - √(1-2C)`. -/
theorem pairingScore_eq (hx : MemLambdaSet m x) :
    pairingScore x = 1 - Real.sqrt (1 - 2 * statC x) := by
  rw [pairingScore, sum_sq_lamPlus_sub_lamMinus hx]

/-- Part (4): `Σ_j ((λ₊)_j + i(λ₋)_j)² = ℐ + 2iC`.  (The identity
holds without the hypothesis `λ ∈ Λ_m`, which the paper's statement
carries.) -/
theorem sum_sq_complex (_hx : MemLambdaSet m x) :
    ∑ j, ((lamPlus x j : ℂ) + lamMinus x j * Complex.I) ^ 2
      = (imbalance x : ℂ) + 2 * Complex.I * (statC x : ℂ) := by
  have hterm : ∀ j : Fin m,
      ((lamPlus x j : ℂ) + lamMinus x j * Complex.I) ^ 2
        = ((lamPlus x j ^ 2 - lamMinus x j ^ 2 : ℝ) : ℂ)
          + ((2 * (lamPlus x j * lamMinus x j) : ℝ) : ℂ) * Complex.I := by
    intro j
    apply Complex.ext
    · simp [pow_two, Complex.mul_re, Complex.mul_im, Complex.add_re,
        Complex.add_im]
    · simp [pow_two, Complex.mul_re, Complex.mul_im, Complex.add_re,
        Complex.add_im]
      ring
  have h1 : ∑ j, (lamPlus x j ^ 2 - lamMinus x j ^ 2) = imbalance x := by
    rw [imbalance, Finset.sum_sub_distrib]
  have h2 : ∑ j, 2 * (lamPlus x j * lamMinus x j) = 2 * statC x := by
    rw [statC, Finset.mul_sum]
  simp_rw [hterm]
  rw [Finset.sum_add_distrib, ← Complex.ofReal_sum, ← Finset.sum_mul,
    ← Complex.ofReal_sum, h1, h2]
  push_cast
  ring

/-! ### The note after the lemma: `0 ≤ 2C ≤ 1`, so `0 ≤ 𝒫 ≤ 1`. -/

/-- `C ≥ 0`, since the entries of `λ±` are nonnegative. -/
theorem statC_nonneg (hx : MemLambdaSet m x) : 0 ≤ statC x :=
  Finset.sum_nonneg fun j _ =>
    mul_nonneg (hx.lamPlus_pos j).le (hx.lamMinus_pos j).le

/-- `2C ≤ 1`, by part (3). -/
theorem two_mul_statC_le_one (hx : MemLambdaSet m x) : 2 * statC x ≤ 1 := by
  have h := sum_sq_lamPlus_sub_lamMinus hx
  have h0 : (0 : ℝ) ≤ ∑ j, (lamPlus x j - lamMinus x j) ^ 2 :=
    Finset.sum_nonneg fun j _ => sq_nonneg _
  linarith

/-- `0 ≤ 𝒫`. -/
theorem pairingScore_nonneg (hx : MemLambdaSet m x) : 0 ≤ pairingScore x := by
  rw [pairingScore_eq hx]
  have h1 : 1 - 2 * statC x ≤ 1 := by
    have := statC_nonneg hx
    linarith
  have := Real.sqrt_le_one.mpr h1
  linarith

/-- `𝒫 ≤ 1`. -/
theorem pairingScore_le_one (hx : MemLambdaSet m x) : pairingScore x ≤ 1 := by
  rw [pairingScore_eq hx]
  have := Real.sqrt_nonneg (1 - 2 * statC x)
  linarith

end Appendices
